"""Optional evaluator commands; importing the base CLI never loads benchmark dependencies."""
import json
from pathlib import Path

import typer

app = typer.Typer(help='Run the pinned public benchmark suite against a served checkpoint.')


@app.command('config')
def config_command():
    """Print the public suite, runtime pins and protocol source hashes without a GPU."""
    from .runner import configuration
    typer.echo(json.dumps(configuration(), indent=2))


@app.command('run')
def run_command(endpoint: str = typer.Option(..., help='OpenAI-compatible vLLM endpoint, ending in /v1'),
                model: str = typer.Option('candidate', help='Name of the checkpoint served at the endpoint'),
                tokenizer: Path = typer.Option(..., help='Matching local checkpoint/tokenizer directory'),
                content_hash: str = typer.Option(..., help='Immutable content hash of the served checkpoint'),
                output: Path = typer.Option(Path('public-results'), help='New output directory'),
                manifest: Path | None = typer.Option(None, help='Downloaded pinned validation.json; bundled by default'),
                concurrency: int = typer.Option(32, min=1, max=256)):
    """Run all 1,110 public items; save an advisory report, never an official submission."""
    from .runner import evaluate_public
    try:
        result = evaluate_public(endpoint=endpoint, model=model, tokenizer=tokenizer, content_hash=content_hash,
                                 output=output, manifest_path=manifest, concurrency=concurrency,
                                 progress=lambda done, total: typer.echo(f'Public validation: {done}/{total}', err=True))
    except ImportError:
        typer.echo('Install the pinned public evaluator dependencies in a separate environment; see PUBLIC_EVALUATION.md.', err=True)
        raise typer.Exit(1)
    except Exception as exc:
        typer.echo(f'Public evaluation failed ({type(exc).__name__}); no complete score was accepted. '
                   'Check runtime pins, suite identity and the endpoint. Existing output directories cannot be reused.', err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps({'output': str(output), 'local_only': True,
                          'validation_score': result['validation_score'],
                          'validation_benchmarks': result['validation_benchmarks']}, indent=2))
