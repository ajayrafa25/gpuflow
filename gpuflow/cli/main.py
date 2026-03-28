from __future__ import annotations

import os
import sys
from typing import Optional

import click
import httpx
from rich.console import Console
from rich.table import Table

console = Console()


def get_client(server: str, api_key: str) -> httpx.Client:
    return httpx.Client(
        base_url=server,
        headers={"X-API-Key": api_key},
        timeout=30,
    )


def handle_error(response: httpx.Response):
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        console.print(f"[red]Error {response.status_code}:[/red] {detail}")
        sys.exit(1)


@click.group()
@click.option("--server", default=lambda: os.environ.get("GPUFLOW_SERVER_URL", "http://localhost:8000"), show_default=True)
@click.option("--api-key", default=lambda: os.environ.get("API_KEY", "dev-change-me"), show_default=True)
@click.pass_context
def cli(ctx: click.Context, server: str, api_key: str):
    """GPUFlow - Simple GPU scheduling for ML teams."""
    ctx.ensure_object(dict)
    ctx.obj["server"] = server
    ctx.obj["api_key"] = api_key


@cli.command()
@click.argument("entrypoint")
@click.option("--gpus", default=1, show_default=True, help="Number of GPUs")
@click.option("--nodes", default=1, show_default=True, help="Number of nodes")
@click.option("--name", default=None, help="Job name (defaults to entrypoint basename)")
@click.option("--image", default=None, help="Docker image")
@click.option("--command", default=None, help="Override the launch command")
@click.pass_context
def run(ctx: click.Context, entrypoint: str, gpus: int, nodes: int, name: Optional[str], image: Optional[str], command: Optional[str]):
    """Submit a training job."""
    if not name:
        name = os.path.basename(entrypoint).replace(".py", "")

    payload = {
        "entrypoint": entrypoint,
        "name": name,
        "requested_gpus": gpus,
        "requested_nodes": nodes,
    }
    if image:
        payload["docker_image"] = image
    if command:
        payload["command"] = command

    with get_client(ctx.obj["server"], ctx.obj["api_key"]) as client:
        resp = client.post("/api/v1/jobs", json=payload)
    handle_error(resp)
    job = resp.json()
    console.print(f"[green]Submitted[/green] job [bold]{job['id']}[/bold] ({job['name']}) — status: {job['status']}")


@cli.command()
@click.option("--status", "status_filter", default=None, help="Filter by status (queued/running/completed/failed/cancelled)")
@click.pass_context
def status(ctx: click.Context, status_filter: Optional[str]):
    """List all jobs."""
    params = {}
    if status_filter:
        params["status_filter"] = status_filter

    with get_client(ctx.obj["server"], ctx.obj["api_key"]) as client:
        resp = client.get("/api/v1/jobs", params=params)
    handle_error(resp)
    jobs = resp.json()

    if not jobs:
        console.print("No jobs found.")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", width=12)
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("GPUs", justify="right")
    table.add_column("Image")
    table.add_column("Created")

    status_colors = {
        "queued": "yellow",
        "running": "cyan",
        "completed": "green",
        "failed": "red",
        "cancelled": "dim",
    }

    for job in jobs:
        color = status_colors.get(job["status"], "white")
        table.add_row(
            job["id"][:12],
            job["name"],
            f"[{color}]{job['status']}[/{color}]",
            str(job["requested_gpus"]),
            job["docker_image"].split("/")[-1][:30],
            job["created_at"][:19].replace("T", " "),
        )

    console.print(table)


@cli.command()
@click.argument("job_id")
@click.option("--follow", "-f", is_flag=True, help="Stream logs as they arrive")
@click.pass_context
def logs(ctx: click.Context, job_id: str, follow: bool):
    """Show logs for a job."""
    params = {"follow": "true"} if follow else {}
    url = f"/api/v1/jobs/{job_id}/logs"

    if follow:
        with get_client(ctx.obj["server"], ctx.obj["api_key"]) as client:
            with client.stream("GET", url, params=params) as resp:
                if resp.status_code >= 400:
                    console.print(f"[red]Error {resp.status_code}[/red]")
                    sys.exit(1)
                for chunk in resp.iter_text():
                    console.print(chunk, end="")
    else:
        with get_client(ctx.obj["server"], ctx.obj["api_key"]) as client:
            resp = client.get(url)
        handle_error(resp)
        console.print(resp.text)


@cli.command()
@click.argument("job_id")
@click.pass_context
def cancel(ctx: click.Context, job_id: str):
    """Cancel a queued or running job."""
    with get_client(ctx.obj["server"], ctx.obj["api_key"]) as client:
        resp = client.delete(f"/api/v1/jobs/{job_id}")
    if resp.status_code == 204:
        console.print(f"[green]Cancelled[/green] job {job_id}")
    else:
        handle_error(resp)


if __name__ == "__main__":
    cli()
