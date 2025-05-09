import io
import json
import sys
from typing import Any, Optional

import httpx
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastmcp import Context, FastMCP
from fastmcp.utilities.http import RequestMiddleware
from mcp.server.sse import SseServerTransport
from starlette.routing import Mount

# Initialize FastMCP server
mcp = FastMCP("nekt")

# Constants
NEKT_API_BASE = "https://api.nekt.ai"
USER_AGENT = "nekt-mcp-server/1.0"
# Constants for database explorer
PAGE_SIZE = 100

# Dict to track application status
application_status_cache = {}

print("MCP server initialized", file=sys.stderr)


async def make_nekt_request(
    url: str, api_key: str, method: str = "GET", json_data: dict = None, timeout: float = 30.0
) -> dict[str, Any] | None:
    """Make a request to the Nekt API with the provided API key.

    Args:
        url: The URL to make the request to
        api_key: The API key to use for authentication
        method: HTTP method (GET or POST)
        json_data: JSON data to send for POST requests
        timeout: Request timeout in seconds

    Returns:
        JSON response as dict or None if request failed
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json", "x-api-key": api_key}

    print(f"Making {method} request to {url} with provided API key", file=sys.stderr)

    async with httpx.AsyncClient() as client:
        try:
            if method.upper() == "GET":
                response = await client.get(url, headers=headers, timeout=timeout)
            elif method.upper() == "POST":
                response = await client.post(url, headers=headers, json=json_data, timeout=timeout)
            else:
                print(f"Unsupported HTTP method: {method}", file=sys.stderr)
                return None

            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API request error: {str(e)}", file=sys.stderr)
            return None


async def process_parquet_in_memory(presigned_url: str) -> pd.DataFrame:
    """Download and process a parquet file from a presigned URL asynchronously.

    Args:
        presigned_url: URL to download the parquet file from

    Returns:
        pandas DataFrame containing the data from the parquet file or None on failure
    """
    print("Downloading parquet file...", file=sys.stderr)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(presigned_url)
            if response.status_code == 200:
                print("Download successful, processing data...", file=sys.stderr)
                # Read the parquet file directly from memory
                buffer = io.BytesIO(response.content)
                df = pd.read_parquet(buffer)
                print(f"DataFrame shape: {df.shape}", file=sys.stderr)
                return df
            else:
                print(f"Failed to download file. Status code: {response.status_code}", file=sys.stderr)
                return None
        except Exception as e:
            print(f"Error downloading or processing parquet: {str(e)}", file=sys.stderr)
            return None


def build_sql_query(layer: str, table: str, limit: Optional[int] = None) -> str:
    """Build a SQL query string from layer, table, and optional limit.

    Args:
        layer: The database layer/schema
        table: The table name
        limit: Optional row limit

    Returns:
        SQL query string
    """
    # Ensure layer and table are properly quoted
    quoted_layer = f'"{layer}"'
    quoted_table = f'"{table}"'

    # Build the base query
    sql_query = f"SELECT\n\t*\nFROM\n\t{quoted_layer}.{quoted_table}"

    # Add limit if provided
    if limit is not None:
        sql_query += f"\nLIMIT {limit}"

    return sql_query


@mcp.tool()
async def check_application_status(ctx: Context = None) -> str:
    """Check if the Nekt Explorer application is running.

    This tool checks whether the Nekt Explorer application is currently active and
    ready to execute SQL queries. This should be called before attempting to create
    or execute queries.

    Possible status values:
    - active: Application is fully running and ready to use
    - activating: Application is in the process of starting up
    - inactive: Application is not running

    Returns:
        A JSON string with the application status details.
    """
    print("Checking application status", file=sys.stderr)

    # Check for API key in context headers
    request = ctx.get_http_request()
    api_key = request.headers.get("x-api-key")
    if not api_key:
        return '{"error": "API key not provided in request headers. Please include x-api-key in your request."}'

    # Check if we have a cached status
    if api_key in application_status_cache and application_status_cache[api_key] == "active":
        # Verify it's still active
        response = await make_nekt_request(f"{NEKT_API_BASE}/api/v1/explorer/application/", api_key)
        if response and response.get("status") == "active":
            return json.dumps({"status": "active", "message": "Application is running and ready to execute queries."})
        # Reset cache if not active
        application_status_cache.pop(api_key, None)

    # Get current status
    response = await make_nekt_request(f"{NEKT_API_BASE}/api/v1/explorer/application/", api_key)
    if not response:
        return '{"error": "Failed to check application status - API call failed"}'

    status = response.get("status", "unknown")
    application_status_cache[api_key] = status

    # Return appropriate message based on status
    if status == "active":
        return json.dumps(
            {
                "status": "active",
                "display_status": "Active",
                "message": "Application is running and ready to execute queries.",
                "next_step": "You can now create and execute queries.",
            }
        )
    elif status == "activating":
        return json.dumps(
            {
                "status": "activating",
                "display_status": "Activating",
                "message": "Application is currently starting up. Please wait until it's active.",
                "next_step": "Check status again in a few moments using check_application_status.",
            }
        )
    elif status == "inactive":
        return json.dumps(
            {
                "status": "inactive",
                "display_status": "Inactive",
                "message": "Application is not active. Use the start_application tool to start it.",
                "next_step": "Call start_application to begin the startup process.",
            }
        )
    else:
        return json.dumps(
            {
                "status": status,
                "display_status": "Unknown",
                "message": f"Application has an unexpected status: {status}.",
                "next_step": "Try using start_application to restart the application.",
                "details": response,
            }
        )


@mcp.tool()
async def start_application(ctx: Context = None) -> str:
    """Start the Nekt Explorer application if it's not already running.

    This tool initiates the startup of the Nekt Explorer application. The application
    startup process is asynchronous and may take time to complete. This tool only
    initiates the process and returns immediately - use check_application_status to
    monitor its progress.

    Possible application status values after calling this tool:
    - active: Application is already running and ready to use
    - activating: Application is in the process of starting up
    - inactive: Application failed to start

    Returns:
        A JSON string with the start request result.
    """
    print("Starting application", file=sys.stderr)

    # Check for API key in context headers
    request = ctx.get_http_request()
    api_key = request.headers.get("x-api-key")
    if not api_key:
        return '{"error": "API key not provided in request headers. Please include x-api-key in your request."}'

    # Check current status first
    status_response = await make_nekt_request(f"{NEKT_API_BASE}/api/v1/explorer/application/", api_key)
    if status_response and status_response.get("status") == "active":
        application_status_cache[api_key] = "active"
        return json.dumps(
            {
                "status": "active",
                "display_status": "Active",
                "message": "Application is already running and ready to use.",
                "next_step": "You can now create and execute queries.",
            }
        )

    # Start application
    response = await make_nekt_request(f"{NEKT_API_BASE}/api/v1/explorer/application/start/", api_key, method="POST")

    if not response:
        return '{"error": "Failed to start application - API call failed"}'

    status = response.get("status", "unknown")
    application_status_cache[api_key] = status

    # Return appropriate message based on status after start attempt
    if status == "active":
        return json.dumps(
            {
                "status": "active",
                "display_status": "Active",
                "message": "Application is now active and ready to use.",
                "next_step": "You can now create and execute queries.",
            }
        )
    elif status == "activating":
        return json.dumps(
            {
                "status": "activating",
                "display_status": "Activating",
                "message": "Application startup process initiated. The application is now activating.",
                "next_step": "Use check_application_status to monitor progress until status is 'Active'.",
                "details": response,
            }
        )
    elif status == "inactive":
        return json.dumps(
            {
                "status": "inactive",
                "display_status": "Inactive",
                "message": "Application failed to start and remains inactive.",
                "next_step": "Check system logs or try again later.",
                "details": response,
            }
        )
    else:
        return json.dumps(
            {
                "status": status,
                "display_status": "Unknown",
                "message": f"Application returned an unexpected status: {status}.",
                "next_step": "Check status using check_application_status or try again later.",
                "details": response,
            }
        )


@mcp.tool()
async def create_sql_query(ctx: Context = None, sql_query: str = None) -> str:
    """Create a new SQL query in the Nekt Explorer system.

    This tool creates a new query using the provided SQL statement. Before using this tool,
    ensure that the application is running by using check_application_status or start_application.

    Args:
        sql_query: The SQL query to execute (required)

    Returns:
        A JSON string with the created query details, including the query slug that can be used
        for execution.
    """
    print("Creating SQL query", file=sys.stderr)

    # Check for API key in context headers
    request = ctx.get_http_request()
    api_key = request.headers.get("x-api-key")
    if not api_key:
        return '{"error": "API key not provided in request headers. Please include x-api-key in your request."}'

    # Validate required parameters
    if not sql_query:
        return '{"error": "Missing required parameter: sql_query"}'

    # Check if application is active
    if api_key not in application_status_cache or application_status_cache[api_key] != "active":
        status_response = await make_nekt_request(f"{NEKT_API_BASE}/api/v1/explorer/application/", api_key)
        if not status_response or status_response.get("status") != "active":
            return json.dumps(
                {
                    "error": "Application is not active",
                    "message": "Start the application first using the start_application tool and wait until it is active",
                }
            )
        application_status_cache[api_key] = "active"

    # Create the query
    response = await make_nekt_request(
        f"{NEKT_API_BASE}/api/v1/explorer/queries/", api_key, method="POST", json_data={"sql_query": sql_query}
    )

    if not response:
        return '{"error": "Failed to create query - API call failed"}'

    query_slug = response.get("slug")
    if not query_slug:
        return json.dumps({"error": "Failed to create query - no slug returned", "details": response})

    return json.dumps(
        {"status": "success", "message": "Query created successfully", "query_slug": query_slug, "sql_query": sql_query}
    )


@mcp.tool()
async def create_table_query(ctx: Context = None, layer: str = None, table: str = None, limit: int = None) -> str:
    """Create a new SQL query for a specific table in the Nekt Explorer system.

    This tool creates a new query to select data from a specified table within a layer/schema.
    Before using this tool, ensure that the application is running by using check_application_status
    or start_application.

    Args:
        layer: The database layer/schema name (required)
        table: The table name (required)
        limit: Optional limit on the number of rows to return

    Returns:
        A JSON string with the created query details, including the query slug that can be used
        for execution.
    """
    print(f"Creating table query for {layer}.{table}", file=sys.stderr)

    # Check for API key in context headers
    request = ctx.get_http_request()
    api_key = request.headers.get("x-api-key")
    if not api_key:
        return '{"error": "API key not provided in request headers. Please include x-api-key in your request."}'

    # Validate required parameters
    if not layer:
        return '{"error": "Missing required parameter: layer"}'
    if not table:
        return '{"error": "Missing required parameter: table"}'

    # Validate limit if provided
    if limit is not None and limit <= 0:
        return '{"error": "Limit must be a positive integer"}'

    # Build the SQL query
    sql_query = build_sql_query(layer, table, limit)

    # Reuse the create_sql_query tool logic
    return await create_sql_query(ctx, sql_query)


@mcp.tool()
async def start_query_execution(ctx: Context = None, query_slug: str = None, page_number: int = 1) -> str:
    """Start the execution of a previously created SQL query.

    This tool initiates the execution of a SQL query identified by its slug. The query
    execution process is asynchronous and may take time to complete. This tool only
    initiates the process and returns immediately - use check_query_execution to monitor
    its progress.

    Args:
        query_slug: The slug of the query to execute (required)
        page_number: The page number to fetch (default: 1)

    Returns:
        A JSON string with the execution details including the execution ID needed to
        check status and get results.
    """
    print(f"Starting execution of query {query_slug} for page {page_number}", file=sys.stderr)

    # Check for API key in context headers
    request = ctx.get_http_request()
    api_key = request.headers.get("x-api-key")
    if not api_key:
        return '{"error": "API key not provided in request headers. Please include x-api-key in your request."}'

    # Validate required parameters
    if not query_slug:
        return '{"error": "Missing required parameter: query_slug"}'

    if page_number < 1:
        return '{"error": "page_number must be a positive integer"}'

    # Check if application is active
    if api_key not in application_status_cache or application_status_cache[api_key] != "active":
        status_response = await make_nekt_request(f"{NEKT_API_BASE}/api/v1/explorer/application/", api_key)
        if not status_response or status_response.get("status") != "active":
            return json.dumps(
                {
                    "error": "Application is not active",
                    "message": "Start the application first using the start_application tool and wait until it is active",
                }
            )
        application_status_cache[api_key] = "active"

    # Execute the query for the requested page
    response = await make_nekt_request(
        f"{NEKT_API_BASE}/api/v1/explorer/queries/{query_slug}/execution/",
        api_key,
        method="POST",
        json_data={"page_number": page_number},
    )

    if not response:
        return '{"error": "Failed to start query execution - API call failed"}'

    execution_id = response.get("id")
    if not execution_id:
        return json.dumps({"error": "Failed to start query execution - no execution ID returned", "details": response})

    return json.dumps(
        {
            "status": "success",
            "message": "Query execution started successfully",
            "query_slug": query_slug,
            "execution_id": execution_id,
            "page_number": page_number,
            "execution_status": response.get("status", "unknown"),
            "next_step": "Use check_query_execution to monitor the execution status",
        }
    )


@mcp.tool()
async def check_query_execution(ctx: Context = None, query_slug: str = None, execution_id: str = None) -> str:
    """Check the status of a running query execution.

    This tool checks whether a previously started query execution has completed.
    If the execution is complete, you can proceed to fetch the results using
    get_query_results.

    Args:
        query_slug: The slug of the query (required)
        execution_id: The execution ID returned from start_query_execution (required)

    Returns:
        A JSON string with the current execution status.
    """
    print(f"Checking execution status of query {query_slug}, execution ID {execution_id}", file=sys.stderr)

    # Check for API key in context headers
    request = ctx.get_http_request()
    api_key = request.headers.get("x-api-key")
    if not api_key:
        return '{"error": "API key not provided in request headers. Please include x-api-key in your request."}'

    # Validate required parameters
    if not query_slug:
        return '{"error": "Missing required parameter: query_slug"}'
    if not execution_id:
        return '{"error": "Missing required parameter: execution_id"}'

    # Check execution status
    response = await make_nekt_request(
        f"{NEKT_API_BASE}/api/v1/explorer/queries/{query_slug}/execution/{execution_id}/", api_key
    )

    if not response:
        return '{"error": "Failed to check execution status - API call failed"}'

    status = response.get("status", "unknown")

    if status == "complete":
        return json.dumps(
            {
                "status": "complete",
                "message": "Query execution completed successfully",
                "query_slug": query_slug,
                "execution_id": execution_id,
                "next_step": "Use get_query_results to fetch the results",
            }
        )
    elif status == "failed":
        error_message = "Query execution failed"
        if "error" in response:
            error_message = response["error"]
        elif "message" in response:
            error_message = response["message"]

        return json.dumps(
            {
                "status": "failed",
                "message": "Query execution failed",
                "error": error_message,
                "query_slug": query_slug,
                "execution_id": execution_id,
                "details": response,
            }
        )
    else:
        return json.dumps(
            {
                "status": status,
                "message": f"Query execution is still in progress with status: {status}",
                "query_slug": query_slug,
                "execution_id": execution_id,
                "next_step": "Check again later using check_query_execution",
            }
        )


@mcp.tool()
async def get_query_results(ctx: Context = None, query_slug: str = None, execution_id: str = None) -> str:
    """Fetch the results of a completed query execution.

    This tool retrieves the results of a successfully completed query execution.
    It should only be called after check_query_execution confirms the execution
    has completed successfully.

    Args:
        query_slug: The slug of the query (required)
        execution_id: The execution ID returned from start_query_execution (required)

    Returns:
        A JSON string with the query results data.
    """
    print(f"Getting results for query {query_slug}, execution ID {execution_id}", file=sys.stderr)

    # Check for API key in context headers
    request = ctx.get_http_request()
    api_key = request.headers.get("x-api-key")
    if not api_key:
        return '{"error": "API key not provided in request headers. Please include x-api-key in your request."}'

    # Validate required parameters
    if not query_slug:
        return '{"error": "Missing required parameter: query_slug"}'
    if not execution_id:
        return '{"error": "Missing required parameter: execution_id"}'

    # Verify execution is complete
    status_response = await make_nekt_request(
        f"{NEKT_API_BASE}/api/v1/explorer/queries/{query_slug}/execution/{execution_id}/", api_key
    )

    if not status_response:
        return '{"error": "Failed to verify execution status - API call failed"}'

    status = status_response.get("status")
    if status != "complete":
        return json.dumps(
            {
                "error": "Query execution is not complete",
                "status": status,
                "message": "Cannot get results until execution is complete",
                "query_slug": query_slug,
                "execution_id": execution_id,
            }
        )

    # Get results
    response = await make_nekt_request(
        f"{NEKT_API_BASE}/api/v1/explorer/queries/{query_slug}/execution/{execution_id}/results/", api_key
    )

    if not response:
        return '{"error": "Failed to get query results - API call failed"}'

    presigned_url = response.get("presigned_url")
    if not presigned_url:
        return json.dumps({"error": "No presigned URL for results found", "details": response})

    # Process the parquet file
    df = await process_parquet_in_memory(presigned_url)
    if df is None:
        return '{"error": "Failed to process parquet file"}'

    # Convert to JSON with records orientation
    try:
        # Limit records if there are too many to avoid excessive token usage
        max_records = 100
        actual_rows = len(df)
        if len(df) > max_records:
            df = df.head(max_records)

        data_json = df.to_json(orient="records", date_format="iso")

        result = {
            "status": "success",
            "query_slug": query_slug,
            "execution_id": execution_id,
            "total_rows": actual_rows,
            "returned_rows": len(df),
            "truncated": actual_rows > max_records,
            "data": json.loads(data_json),
        }

        return json.dumps(result)
    except Exception as e:
        error_msg = str(e)
        print(f"Error converting dataframe to JSON: {error_msg}", file=sys.stderr)
        return json.dumps(
            {
                "error": "Failed to convert results to JSON",
                "message": error_msg,
                "query_slug": query_slug,
                "execution_id": execution_id,
            }
        )


@mcp.tool()
async def get_sources(ctx: Context = None, page: int = 1, page_size: int = 20) -> str:
    """Retrieve a comprehensive list of all data sources available in the Nekt platform.

    This tool returns all data sources the user has access to, including details like name,
    description, pipeline status, and more. Use this tool whenever the user asks to see,
    list, or know about their available data sources.

    Args:
        page: Page number to fetch (default: 1)
        page_size: Number of results per page (default: 20)

    Returns:
        A JSON-formatted string containing data sources information and pagination details,
        optimized for consumption by an LLM.
    """
    print(f"get_sources called with page={page}, page_size={page_size}", file=sys.stderr)

    # Check for API key in context headers
    request = ctx.get_http_request()
    api_key = request.headers.get("x-api-key")
    if not api_key:
        return '{"error": "API key not provided in request headers. Please include x-api-key in your request."}'

    url = f"{NEKT_API_BASE}/api/v1/sources/?page={page}&page_size={page_size}&expand[]=last_run&expand[]=connector_version.connector"
    data = await make_nekt_request(url, api_key)

    if not data:
        return '{"error": "Unable to fetch sources - API call failed or returned no data"}'

    # Extract pagination information
    total_count = data.get("count", 0)
    next_url = data.get("next")
    previous_url = data.get("previous")

    # Calculate total pages
    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 0

    results = data.get("results", [])
    if not results:
        return (
            '{"sources": [], "pagination": {"total": 0, "current_page": '
            + str(page)
            + ', "total_pages": 0}, "message": "No sources found"}'
        )

    # Format the sources with relevant information for LLM consumption
    formatted_sources = []
    for source in results:
        # Extract status information
        active = source.get("active", False)
        status_text = source.get("status", "unknown")

        # Format last run information
        last_run = source.get("last_run", {})
        last_run_info = {}
        if last_run:
            # Calculate duration if we have start and end times
            started_at = last_run.get("started_at", "")
            ended_at = last_run.get("ended_at", "")

            duration = ""
            if started_at and ended_at:
                try:
                    from datetime import datetime

                    start_time = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                    end_time = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
                    duration_seconds = (end_time - start_time).total_seconds()

                    # Format duration in a readable way
                    minutes, seconds = divmod(duration_seconds, 60)
                    hours, minutes = divmod(minutes, 60)

                    if hours > 0:
                        duration = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                    elif minutes > 0:
                        duration = f"{int(minutes)}m {int(seconds)}s"
                    else:
                        duration = f"{int(seconds)}s"
                except Exception as e:
                    print(f"Error calculating duration: {e}", file=sys.stderr)

            last_run_info = {
                "status": last_run.get("status", ""),
                "ended_at": ended_at,
                "started_at": started_at,
                "duration": duration,
            }

        # Create structured source entry
        source_entry = {
            "name": source["connector_version"]["connector"]["name"],
            "slug": source["slug"],
            "active": active,
            "status": status_text,
            "description": source.get("description", "No description"),
            "last_run": last_run_info,
        }
        formatted_sources.append(source_entry)

    # Create a structured response
    response = {
        "sources": formatted_sources,
        "pagination": {
            "total": total_count,
            "current_page": page,
            "total_pages": total_pages,
            "has_next": next_url is not None,
            "has_previous": previous_url is not None,
            "next_page": page + 1 if next_url else None,
            "previous_page": page - 1 if previous_url else None,
        },
    }

    return json.dumps(response)


@mcp.tool()
async def query_status_summary(ctx: Context = None) -> str:
    """Get a summary of the database explorer application and any active query executions.

    This tool provides a quick overview of the application status and any previously started
    query executions that the system is aware of. Use this to get a high-level understanding
    of what's happening in the database explorer system.

    Possible application status values:
    - active: Application is fully running and ready to use
    - activating: Application is in the process of starting up
    - inactive: Application is not running

    Returns:
        A JSON string with the current status summary.
    """
    print("Getting query status summary", file=sys.stderr)

    # Check for API key in context headers
    request = ctx.get_http_request()
    api_key = request.headers.get("x-api-key")
    if not api_key:
        return '{"error": "API key not provided in request headers. Please include x-api-key in your request."}'

    # Get application status
    app_status = "unknown"
    if api_key in application_status_cache:
        app_status = application_status_cache[api_key]
        # Verify it's still correct
        try:
            status_response = await make_nekt_request(f"{NEKT_API_BASE}/api/v1/explorer/application/", api_key)
            if status_response:
                app_status = status_response.get("status", "unknown")
                application_status_cache[api_key] = app_status
        except Exception:
            pass

    summary = {
        "application_status": app_status,
        "display_status": get_display_status(app_status),
        "is_ready": app_status == "active",
        "message": "This is a summary of the database explorer system status.",
    }

    # Add appropriate next steps based on status
    if app_status == "active":
        summary["next_step"] = "Use create_sql_query or create_table_query to create a new query"
    elif app_status == "activating":
        summary["next_step"] = (
            "Wait for application to finish starting up and check status again using check_application_status"
        )
    elif app_status == "inactive":
        summary["next_step"] = "Use start_application to start the database explorer application"
    else:
        summary["next_step"] = "Check application status with check_application_status for more details"

    return json.dumps(summary)


def get_display_status(status: str) -> str:
    """Convert internal status values to display-friendly versions.

    Args:
        status: The internal status string

    Returns:
        A display-friendly version of the status
    """
    status_map = {"active": "Active", "activating": "Activating", "inactive": "Inactive"}
    return status_map.get(status, "Unknown")


app = FastAPI(title="Nekt MCP Server", docs_url=None, redoc_url=None)
sse = SseServerTransport("/messages/")
app.router.routes.append(Mount("/messages", app=sse.handle_post_message))

# Add the RequestMiddleware to the app to make the request available in the context var
app.add_middleware(RequestMiddleware)


@app.get("/sse", tags=["MCP"])
async def handle_sse(request: Request, x_api_key: str = Header(None)):
    """Handle SSE requests.

    Args:
        request: The incoming request
        x_api_key: The API key from the x-api-key header
    """
    print("SSE connection requested", file=sys.stderr)

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key in x-api-key header")

    try:
        async with sse.connect_sse(request.scope, request.receive, request._send) as (read_stream, write_stream):
            init_options = mcp._mcp_server.create_initialization_options()

            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                init_options,
            )
    except Exception as e:
        print(f"Error in SSE connection: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    import os

    import uvicorn

    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"Starting nekt server with SSE on {host}:{port}...", file=sys.stderr)
    uvicorn.run(app, host=host, port=port)
