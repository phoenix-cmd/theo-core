"""THEO Core CLI entry point with interactive 'theo chat' REPL."""

from __future__ import annotations

import sys

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


@app.command(name="chat")
def chat() -> None:
    """Start an interactive cognitive session with THEO."""
    from theo_core.composition.bootstrap import bootstrap

    container = bootstrap()
    container.kernel.boot()

    typer.echo("\n==================================================")
    typer.echo("      THEO v0.2.0 — Deterministic Cognitive REPL  ")
    typer.echo("==================================================")
    typer.echo(
        "Commands: /explain, /trace, /memory, /knowledge, /goals, /context, /replay <id>, /exit\n"
    )

    while True:
        try:
            user_input = typer.prompt("You").strip()
            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                typer.echo("Shutting down THEO cognitive runtime. Goodbye!")
                break

            if user_input.startswith("/"):
                parts = user_input.split()
                cmd = parts[0].lower()

                if cmd == "/context":
                    typer.echo(f"\n[Active Context]: {container.context_manager.snapshot()}\n")
                    continue
                if cmd == "/goals":
                    active_goals = [
                        g.description for g in container.goal_manager.get_active_goals()
                    ]
                    typer.echo(f"\n[Active Goals]: {active_goals}\n")
                    continue
                if cmd == "/memory":
                    active_memories = [
                        m.model_dump(mode="json") for m in container.memory_engine.get_all_active()
                    ]
                    typer.echo(f"\n[Active Persistent Memories ({len(active_memories)})]:")
                    for mem in active_memories:
                        m_id = mem["id"]
                        m_type = mem["memory_type"]
                        m_key = mem["key"]
                        m_val = mem["value"]
                        typer.echo(f"  - [{m_id}] ({m_type}) {m_key} = {m_val}")
                    typer.echo("")
                    continue
                if cmd == "/knowledge":
                    triples = container.knowledge_engine.traverse("Astronomy", max_depth=2)
                    typer.echo(f"\n[Knowledge Graph Relationships ({len(triples)})]:")
                    for t in triples:
                        typer.echo(f"  - {t.subject} --[{t.predicate}]--> {t.object}")
                    typer.echo("")
                    continue
                if cmd == "/explain":
                    if container.cognitive_engine.last_state:
                        explanation = container.explain_engine.explain_state(
                            container.cognitive_engine.last_state
                        )
                        typer.echo(f"\n{explanation}\n")
                    else:
                        typer.echo("\nNo cognitive cycle executed yet.\n")
                    continue
                if cmd == "/trace":
                    rec = container.cognitive_engine.last_record
                    if rec and rec.trace_id:
                        trace = container.trace_recorder.load_trace(rec.trace_id)
                        if trace:
                            typer.echo(f"\n[Cognitive Trace {trace.trace_id}]:")
                            typer.echo(f"  Duration: {trace.total_duration_ms:.2f}ms")
                            for span in trace.spans:
                                typer.echo(f"  - [{span.stage_name}] {span.duration_ms:.2f}ms")
                            typer.echo("")
                            continue
                    typer.echo("\nNo trace available for the last turn.\n")
                    continue
                if cmd == "/replay":
                    if len(parts) < 2:
                        rec = container.cognitive_engine.last_record
                        t_id = str(rec.trace_id) if rec and rec.trace_id else None
                    else:
                        t_id = parts[1]

                    if t_id:
                        result = container.replay_engine.replay(t_id)
                        is_match = result.matched
                        status = "✅ MATCH (0-variance match)" if is_match else "❌ MISMATCH"
                        typer.echo(f"\n[Trace Replay Result for {t_id}]:")
                        typer.echo(f"  Status:   {status}")
                        typer.echo(f"  Original: '{result.original_output}'")
                        typer.echo(f"  Replayed: '{result.replayed_output}'\n")
                    else:
                        typer.echo("\nUsage: /replay <trace_id>\n")
                    continue

            # Run 12-stage cognitive cycle
            state = container.cognitive_engine.process(user_input)
            last_rec = container.cognitive_engine.last_record
            trace_str = str(last_rec.trace_id) if last_rec and last_rec.trace_id else "N/A"

            typer.echo(f"\nTheo > {state.response_text}")
            typer.echo(
                f"       (Cognitive Depth: {state.cognitive_depth} stages | "
                f"Memory: {state.memory_classification} | "
                f"Trace ID: {trace_str})\n"
            )

        except (KeyboardInterrupt, EOFError):
            typer.echo("\nShutting down THEO cognitive runtime. Goodbye!")
            sys.exit(0)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
