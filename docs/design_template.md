# Design Template

## Problem

The system must handle complex, long-form academic and professional research queries (e.g., investigating GraphRAG state-of-the-art or evaluating multi-agent system performance). This requires retrieving verified external facts, analyzing conflicting viewpoints, evaluating evidence quality, and synthesizing a structured report tailored to a specific audience, complete with formal inline source citations and a bibliography.

## Why multi-agent?

A single-agent baseline usually performs search and writing in a single prompt context. While fast, this approach has key limitations:
- **Lack of Critical Separation**: The agent writes the final response without a dedicated step to analyze claim strength or identify knowledge gaps, resulting in superficial summaries.
- **Hallucinations & Misattributions**: The agent frequently mixes facts across sources or invents claims without exact source mapping.
- **No Quality Control**: There is no self-correction loop to review if the response adheres to structural instructions and citation coverage rules.

Decomposing the task into a multi-agent workflow isolates concerns (Researching vs. Analyzing vs. Writing vs. Criticizing), resulting in more thorough verification, structured feedback loops, and highly rigorous reports.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode | Mitigation |
|---|---|---|---|---|---|
| **Supervisor** | Orchestrates the flow, coordinates handoffs between specialized agents, and stops when criteria are met. | `ResearchState` | Next agent selection (`next_agent`) | Infinite loop routing between nodes. | Enforces a maximum iteration limit (`max_iterations=6`) and fallback logic. |
| **Researcher** | Generates search queries, calls SearchClient, deduplicates resources, and drafts research notes with source citations. | Query from `request` | `sources`, `research_notes` | Retrieval of low-quality or irrelevant snippets. | Uses LLM to break down the main query into 2 targeted search queries. |
| **Analyst** | Critically evaluates research notes, extracts key claims, compares perspectives, and identifies gaps/contradictions. | `research_notes` | `analysis_notes` | Superficial review or repeating research notes. | Enforces strict structured output sections ('Key Claims', 'Viewpoint Analysis', etc.). |
| **Writer** | Synthesizes research and analysis notes into a cohesive, audience-targeted brief with inline citations and references. | `research_notes`, `analysis_notes`, `audience` | `final_answer` | Hallucinating claims or dropping original source URLs. | Contextually binds references and demands mapping of original URL list. |
| **Critic** | Fact-checks final answer draft against raw research notes, verifying citations and quality standards. | `final_answer`, `research_notes` | Status (`APPROVED` / `NEEDS_REVISION`) + feedback | Lazy approvals or infinite rejections. | Implements clear rules for approval criteria and limits maximum revision runs. |

## Shared state

We use `ResearchState` as the single source of truth:
- `request`: Contains query parameters (query string, max sources, target audience).
- `iteration`: Tracks count to prevent infinite runs.
- `route_history`: Logs routing sequences to help analyze flow and debug decisions.
- `sources`: Collected source list containing title, url, snippet.
- `research_notes`: Synthesized facts mapped to index sources.
- `analysis_notes`: Critical review of facts, viewpoints, and gaps.
- `final_answer`: Final synthesized document.
- `agent_results`: Individual agent outputs for transparency and debugging.
- `trace`: Structured trace events for observability.

## Routing policy

The architecture follows a **Hub-and-Spoke (Supervisor-centric)** routing model using LangGraph:

```text
       START
         |
         v
  +--------------+
  |  Supervisor  |<------------------------+
  +--------------+                         |
         | (Conditional Edge)              |
         +-------------------+             |
         |                   |             |
         v                   v             |
  [researcher]          [analyst]          |
         |                   |             |
         v                   v             |
         +-------------------+-------------+
         |                   |             |
         v                   v             |
    [writer]             [critic]          |
         |                   |             |
         v                   v             |
         +-------------------+-------------+
                             |
                             v
                            END
```

- Every worker agent completes its task and transitions control directly back to the **Supervisor**.
- The Supervisor inspects the shared state, makes an LLM routing call, and points to the next logical worker node or exits to `END`.

## Guardrails

- **Max iterations**: Caps at `6` runs (configurable via `MAX_ITERATIONS` settings). If reached, Supervisor automatically routes to `writer` or `done`.
- **Timeout**: Enforces a `60` seconds execution limit per API call.
- **Retry**: Wrap all LLM completions with `tenacity` exponential backoffs (3 attempts) on API failures.
- **Fallback**: Implements fallback state-routing rules if JSON parsing fails on Supervisor decisions, and falls back to a high-quality Local Mock Search if Tavily API fails or has no key.
- **Validation**: Enforces strict Pydantic schemas for the shared state and agent handoffs.

## Benchmark plan

- **Evaluation Queries**:
  1. "Research GraphRAG state-of-the-art and write a 500-word summary"
  2. "Compare single-agent and multi-agent workflows for customer support"
  3. "Summarize production guardrails for LLM agents"
- **Metrics**:
  - *Latency*: Measures total execution wall-clock time in seconds.
  - *Cost*: Estimates token costs (USD) based on input and output lengths.
  - *Quality*: Evaluates content using LLM-as-a-judge on a scale from 0.0 to 10.0.
  - *Citation Coverage*: Measures the percentage of search sources correctly cited in the final document.
- **Expected Outcome**: The Multi-Agent system will have higher latency and slightly higher costs but should deliver significantly better structured notes, zero hallucinations, and much higher quality scores with accurate citations compared to the Single-Agent baseline.
