"""CLI entry point for WrinkleFree Inference Engine."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option()
def main():
    """WrinkleFree Inference Engine - BitNet model serving."""
    pass


@main.command()
@click.option(
    "--hf-repo",
    default="microsoft/BitNet-b1.58-2B-4T",
    help="HuggingFace repository ID",
)
@click.option(
    "--quant-type",
    default="i2_s",
    type=click.Choice(["i2_s", "tl1", "tl2"]),
    help="Quantization type",
)
@click.option(
    "--bitnet-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to BitNet installation",
)
def convert(hf_repo: str, quant_type: str, bitnet_path: Optional[Path]):
    """Convert a HuggingFace model to GGUF format."""
    from wrinklefree_inference.converter.hf_to_gguf import (
        ConversionConfig,
        HFToGGUFConverter,
    )

    console.print(f"[bold]Converting {hf_repo}[/bold]")
    console.print(f"Quantization type: {quant_type}")

    try:
        converter = HFToGGUFConverter(bitnet_path)
        config = ConversionConfig(hf_repo=hf_repo, quant_type=quant_type)

        def progress(msg: str):
            console.print(f"  {msg}")

        gguf_path = converter.convert(config, progress_callback=progress)
        console.print(f"\n[green]Success![/green] Model saved to: {gguf_path}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to GGUF model file",
)
@click.option("--port", "-p", default=8080, help="Server port")
@click.option("--host", "-h", default="0.0.0.0", help="Server host")
@click.option("--threads", "-t", default=0, help="Number of threads (0=auto)")
@click.option("--context-size", "-c", default=4096, help="Context size (KV cache)")
@click.option(
    "--bitnet-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to BitNet installation",
)
def serve(
    model: Path,
    port: int,
    host: str,
    threads: int,
    context_size: int,
    bitnet_path: Optional[Path],
):
    """Start the BitNet inference server."""
    from wrinklefree_inference.server.bitnet_server import (
        BitNetServer,
        get_default_bitnet_path,
    )

    if bitnet_path is None:
        bitnet_path = get_default_bitnet_path()

    console.print(f"[bold]Starting inference server[/bold]")
    console.print(f"Model: {model}")
    console.print(f"Port: {port}")
    console.print(f"Context size: {context_size}")

    server = BitNetServer(
        bitnet_path=bitnet_path,
        model_path=model,
        port=port,
        host=host,
        num_threads=threads,
        context_size=context_size,
        continuous_batching=True,
    )

    try:
        console.print("\n[yellow]Starting server...[/yellow]")
        server.start(wait_for_ready=True, timeout=120)
        console.print(f"[green]Server running at http://{host}:{port}[/green]")
        console.print("Press Ctrl+C to stop")

        # Keep running
        import time
        while server.is_running():
            time.sleep(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    finally:
        server.stop()
        console.print("[green]Server stopped[/green]")


@main.command()
@click.option(
    "--url",
    default="http://localhost:8080",
    help="Inference server URL",
)
@click.option("--prompt", "-p", required=True, help="Prompt to generate from")
@click.option("--max-tokens", "-n", default=128, help="Maximum tokens to generate")
@click.option("--temperature", "-t", default=0.7, help="Sampling temperature")
@click.option("--stream/--no-stream", default=True, help="Stream output")
def generate(url: str, prompt: str, max_tokens: int, temperature: float, stream: bool):
    """Generate text from a prompt."""
    from wrinklefree_inference.client.bitnet_client import BitNetClient

    # Parse URL
    url_clean = url.replace("http://", "").replace("https://", "")
    if ":" in url_clean:
        host, port_str = url_clean.split(":")
        port = int(port_str.split("/")[0])
    else:
        host = url_clean.split("/")[0]
        port = 8080

    client = BitNetClient(host=host, port=port)

    if not client.health_check():
        console.print(f"[red]Error:[/red] Server not available at {url}")
        sys.exit(1)

    if stream:
        for chunk in client.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            console.print(chunk, end="")
        console.print()
    else:
        response = client.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        console.print(response)


@main.command()
@click.option(
    "--url",
    default="http://localhost:8080",
    help="Inference server URL",
)
@click.option("--timeout", default=60, help="Request timeout in seconds")
def validate(url: str, timeout: int):
    """Validate KV cache behavior."""
    from wrinklefree_inference.kv_cache.validator import run_kv_cache_validation

    console.print(f"[bold]Validating KV cache at {url}[/bold]\n")

    metrics = run_kv_cache_validation(url, timeout)

    table = Table(title="KV Cache Validation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Prefix Speedup", f"{metrics.prefix_speedup:.2f}x")
    table.add_row("First Request Latency", f"{metrics.first_request_latency_ms:.1f}ms")
    table.add_row("Second Request Latency", f"{metrics.second_request_latency_ms:.1f}ms")
    table.add_row("Context Limit Handled", str(metrics.context_limit_handled))
    table.add_row("Concurrent Success Rate", f"{metrics.concurrent_success_rate*100:.0f}%")

    console.print(table)

    if metrics.errors:
        console.print("\n[red]Errors:[/red]")
        for error in metrics.errors:
            console.print(f"  - {error}")
        sys.exit(1)

    # Check pass criteria
    if metrics.concurrent_success_rate < 0.8:
        console.print("\n[red]FAIL:[/red] Concurrent success rate too low")
        sys.exit(1)

    console.print("\n[green]PASS:[/green] KV cache validation successful")


@main.command()
@click.option(
    "--bitnet-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to BitNet installation",
)
def list_models(bitnet_path: Optional[Path]):
    """List available GGUF models."""
    from wrinklefree_inference.converter.hf_to_gguf import HFToGGUFConverter

    try:
        converter = HFToGGUFConverter(bitnet_path)
        models = converter.list_available_models()

        if not models:
            console.print("[yellow]No models found[/yellow]")
            console.print("Run 'wrinklefree-inference convert' to download a model")
            return

        table = Table(title="Available Models")
        table.add_column("Model", style="cyan")
        table.add_column("Size", style="green")

        for model_path in models:
            size_mb = model_path.stat().st_size / (1024 * 1024)
            table.add_row(str(model_path.name), f"{size_mb:.1f} MB")

        console.print(table)

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
