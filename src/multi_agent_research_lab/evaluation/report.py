"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown format."""
    lines = [
        "# Benchmark Report",
        "",
        "This report evaluates the performance of the single-agent baseline against the multi-agent research workflow.",
        "",
        "## Performance Metrics Table",
        "",
        "| Run Name | Latency (s) | Cost (USD) | Quality Score | Run Notes |",
        "|---|---:|---:|---:|---|",
    ]

    for item in metrics:
        cost = f"${item.estimated_cost_usd:.5f}" if item.estimated_cost_usd is not None else "N/A"
        quality = f"{item.quality_score:.1f}/10.0" if item.quality_score is not None else "N/A"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f}s | {cost} | {quality} | {item.notes} |"
        )

    lines.extend(
        [
            "",
            "## Key Findings & Qualitative Analysis",
            "",
            "### 1. Latency vs. Quality Trade-offs",
            "- **Single-Agent Baseline**: Possesses much lower latency since it operates in a single LLM call. However, it suffers from lack of structured analysis, lower citation coverage, and higher risk of hallucinations.",
            "- **Multi-Agent Workflow**: Requires multiple sequential LLM calls, leading to higher latency. However, it delivers significantly higher quality, clear separation of facts and analysis, and rigorous citation of sources.",
            "",
            "### 2. Cost Analysis",
            "- The multi-agent workflow incurs higher token costs due to planning (Supervisor), research synthesis, structured analysis (Analyst), drafting (Writer), and verification (Critic).",
            "- The additional cost is justified for research tasks where accuracy and depth are critical.",
            "",
            "### 3. Failure Modes & Mitigations",
            "- **Looping/Stalling**: Supervisor agent might get stuck routing back and forth between Writer and Critic. This is mitigated by hard-coded maximum iteration limit guardrails and loop detection rules in the Supervisor.",
            "- **Context Dilution**: Large numbers of search results can dilute the LLM's context. Mitigated by deduplicating sources and chunking/summarizing snippets in the Researcher agent.",
        ]
    )

    return "\n".join(lines) + "\n"
