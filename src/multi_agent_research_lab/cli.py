"""Command-line entrypoint for the lab starter."""

import os
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def run_baseline_internal(query: str) -> ResearchState:
    """Execute a single-call research flow."""
    search_client = SearchClient()
    llm_client = LLMClient()
    state = ResearchState(request=ResearchQuery(query=query))

    # 1. Search for sources
    sources = search_client.search(query, max_results=state.request.max_sources)
    state.sources = sources

    # 2. Format search results for LLM
    sources_text = ""
    for idx, src in enumerate(sources):
        sources_text += (
            f"[{idx + 1}] Title: {src.title}\nURL: {src.url or 'N/A'}\nSnippet: {src.snippet}\n\n"
        )

    system_prompt = (
        "You are an AI research assistant. Read the provided search sources and write a comprehensive answer "
        "to the query. Cite sources using numbers like [1], [2], etc., corresponding to the sources. "
        "End with a list of References showing titles and URLs."
    )
    user_prompt = f"Query: {query}\n\nSearch Sources:\n{sources_text}"

    llm_res = llm_client.complete(system_prompt, user_prompt)
    state.final_answer = llm_res.content
    state.agent_results.append(AgentResult(agent=AgentName.WRITER, content=llm_res.content))
    return state


def run_multi_agent_internal(query: str) -> ResearchState:
    """Execute the multi-agent graph workflow."""
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline search and synthesis."""
    _init()
    console.print("[bold blue]Starting Single-Agent Baseline...[/bold blue]")
    state = run_baseline_internal(query)
    console.print(
        Panel.fit(
            state.final_answer or "No answer generated.", title="Single-Agent Baseline Result"
        )
    )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""
    _init()
    console.print("[bold green]Starting Multi-Agent Workflow...[/bold green]")
    state = run_multi_agent_internal(query)
    console.print(
        Panel.fit(state.final_answer or "No answer generated.", title="Multi-Agent Workflow Result")
    )


@app.command("benchmark")
def benchmark() -> None:
    """Run comparison benchmarks between single-agent baseline and multi-agent workflow."""
    _init()
    console.print("[bold magenta]Starting Benchmark Runs...[/bold magenta]")

    queries = [
        "Research GraphRAG state-of-the-art and write a 500-word summary",
        "Compare single-agent and multi-agent workflows for customer support",
        "Summarize production guardrails for LLM agents",
    ]

    all_metrics = []

    for q in queries:
        console.print(f"\n[cyan]Benchmarking Query:[/cyan] '{q}'")

        # 1. Run Baseline
        console.print("  Running Baseline...")
        _, metric_base = run_benchmark(f"Baseline - {q[:20]}...", q, run_baseline_internal)
        all_metrics.append(metric_base)

        # 2. Run Multi-Agent
        console.print("  Running Multi-Agent...")
        _, metric_multi = run_benchmark(f"Multi-Agent - {q[:20]}...", q, run_multi_agent_internal)
        all_metrics.append(metric_multi)

    # Render report to string
    report_md = render_markdown_report(all_metrics)

    # Save to reports/
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    console.print(
        f"\n[green]Benchmark completed successfully. Report saved to {report_path}[/green]"
    )

    # Print comparison table to console
    table = Table(title="Benchmark Summary")
    table.add_column("Run Name", style="cyan")
    table.add_column("Latency (s)", justify="right", style="magenta")
    table.add_column("Cost (USD)", justify="right", style="green")
    table.add_column("Quality Score", justify="right", style="yellow")
    table.add_column("Notes", style="white")

    for item in all_metrics:
        cost = f"${item.estimated_cost_usd:.5f}" if item.estimated_cost_usd is not None else "N/A"
        quality = f"{item.quality_score:.1f}/10.0" if item.quality_score is not None else "N/A"
        table.add_row(item.run_name, f"{item.latency_seconds:.2f}s", cost, quality, item.notes)

    console.print(table)


if __name__ == "__main__":
    app()
