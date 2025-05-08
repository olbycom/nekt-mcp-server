import sys
from typing import Any, Optional

import httpx
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

print("MCP server initialized", file=sys.stderr)


async def make_nekt_request(url: str, api_key: str) -> dict[str, Any] | None:
    """Make a request to the Nekt API with the provided API key.

    Args:
        url: The URL to make the request to
        api_key: The API key to use for authentication
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json", "x-api-key": api_key}

    print(f"Making request to {url} with provided API key", file=sys.stderr)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API request error: {str(e)}", file=sys.stderr)
            return None


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

    import json

    return json.dumps(response)


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
