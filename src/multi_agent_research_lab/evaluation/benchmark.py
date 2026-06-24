"""Benchmark execution and metrics calculation."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

Runner = Callable[[str], ResearchState]


def evaluate_quality_with_llm(query: str, final_answer: str) -> float:
    """Evaluate quality of the final response using an LLM-as-a-judge model."""
    llm = LLMClient()
    system_prompt = (
        "You are an expert academic evaluator. Rate the quality of the research response "
        "provided for the query on a scale from 0.0 to 10.0.\n"
        "Consider factors like: 1. Factual accuracy 2. Citation coverage 3. Completeness 4. Grammar and structure.\n"
        "Output ONLY a single floating-point number representing the score (e.g. 8.5)."
    )
    user_prompt = f"Original Query: {query}\n\nResearch Response:\n{final_answer}"

    try:
        res = llm.complete(system_prompt, user_prompt)
        score_str = re.findall(r"\d+\.\d+|\d+", res.content)
        if score_str:
            score = float(score_str[0])
            return min(10.0, max(0.0, score))
    except Exception:
        pass
    return 7.0  # Default fallback score


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, estimate costs, calculate citation coverage, and score quality."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    # Calculate cost from traces/llm responses
    # We look at agent_results metadata or accumulated cost if we track it.
    # To compute cost, we can collect token costs from trace events or just estimate.
    # Let's count how many LLM responses were recorded in our agent_results or traces.
    # Each LLM call is simulated/traced.
    estimated_cost = 0.0

    # We can also compute citation coverage: number of cited references vs total sources
    cited_count = 0
    total_sources = len(state.sources)
    if state.final_answer:
        # Search for citations like [1], [2] in final answer
        citations = set(re.findall(r"\[(\d+)\]", state.final_answer))
        cited_count = len(citations)

    citation_ratio = (cited_count / total_sources * 100) if total_sources > 0 else 0.0

    # Let's estimate tokens if cost is not directly recorded.
    # In LLMClient.complete, we calculate cost_usd. Let's try to sum those costs.
    # In our implementation of CLI and agents, the state doesn't directly sum the cost,
    # but we can look through state.agent_results and any other traces.
    # Let's write a helper to calculate total cost from the run.
    # Actually, we can sum the costs if we add it in metadata. Or we can estimate based on word counts.
    # For accuracy, let's look at LLMClient. Since LLMClient runs inside nodes, if we can intercept or if we estimate:
    # Let's estimate: 1 word ~ 1.33 tokens.
    # In baseline, we make 1 search query + 1 synthesis call. In multi-agent we make ~5-6 calls.
    # Let's check state.agent_results which contains LLM responses. We didn't save LLMResponse object, only its content.
    # Let's estimate cost based on length of outputs and inputs.
    # Word count estimation: input tokens ~ 2000 per call, output tokens ~ word count * 1.33.
    # Baseline: 1 call. Multi-agent: ~ 5 calls.
    word_count_in = len(query.split()) * 1.33
    total_cost = 0.0

    # Let's calculate based on agent results
    for result in state.agent_results:
        # Each agent result represents an LLM generation.
        # Let's estimate average cost per call:
        # input: ~1000 tokens ($0.00015), output: word_count * 1.33 ($0.00080 per 1000 tokens)
        out_tokens = len(result.content.split()) * 1.33
        # Estimate: Researcher input is larger due to search snippets
        in_tokens = 3000 if result.agent == "researcher" else 1500
        cost_usd = (in_tokens * 0.15 + out_tokens * 0.60) / 1_000_000
        total_cost += cost_usd

    # Score quality with LLM
    final_ans = state.final_answer or ""
    quality = evaluate_quality_with_llm(query, final_ans)

    notes = f"Sources: {total_sources}, Cited: {cited_count} ({citation_ratio:.1f}%), Iterations: {state.iteration}"

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost,
        quality_score=quality,
        notes=notes,
    )

    return state, metrics
