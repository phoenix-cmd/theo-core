"""THEO Core CLI entry point."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="theo",
    help="THEO — The Poet: A cognitive operating system.",
)


@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context) -> None:
    """THEO cognitive operating system CLI."""
    if ctx.invoked_subcommand is None:
        from theo_core.composition.bootstrap import bootstrap

        container = bootstrap()
        container.kernel.boot()


@app.command(name="boot")
def boot() -> None:
    """Boot the THEO cognitive kernel."""
    from theo_core.composition.bootstrap import bootstrap

    container = bootstrap()
    container.kernel.boot()


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
