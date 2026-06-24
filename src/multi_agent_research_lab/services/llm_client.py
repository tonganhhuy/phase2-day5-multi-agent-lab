"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client implementation supporting OpenAI and Gemini (OpenAI-compatible) APIs."""

    def __init__(self) -> None:
        self.settings = get_settings()

        # Determine provider configuration based on keys
        if self.settings.gemini_api_key:
            self._client = openai.OpenAI(
                api_key=self.settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self._model = self.settings.gemini_model
        else:
            self._client = openai.OpenAI(api_key=self.settings.openai_api_key or "mock_key")
            self._model = self.settings.openai_model

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Connect OpenAI, Azure OpenAI, or Gemini.
        Keep retry, timeout, and token logging here rather than inside agents.
        """
        # If no real keys are set, fallback to Mock completion for local validation
        if not self.settings.openai_api_key and not self.settings.gemini_api_key:
            return self._mock_complete(system_prompt, user_prompt)

        return self._openai_complete(system_prompt, user_prompt)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True
    )
    def _openai_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=self.settings.timeout_seconds,
        )
        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        # Calculate cost based on model type
        model_lower = self._model.lower()
        if "gemini" in model_lower:
            # Gemini 2.5 Flash Lite pricing (approx $0.075 / 1M input, $0.30 / 1M output)
            cost_usd = (input_tokens * 0.075 + output_tokens * 0.30) / 1_000_000
        elif "gpt-4o-mini" in model_lower:
            cost_usd = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
        elif "gpt-4o" in model_lower:
            cost_usd = (input_tokens * 2.50 + output_tokens * 10.00) / 1_000_000
        else:
            cost_usd = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

    def _mock_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Fallback mock complete function for running offline/keyless tests."""
        sys_lower = system_prompt.lower()
        usr_lower = user_prompt.lower()

        content = ""

        # 1. Supervisor Routing Decision
        if "supervisor" in sys_lower or "router" in sys_lower:
            if "research notes present: false" in usr_lower:
                content = '{"next_agent": "researcher", "reason": "Gather facts and research notes first"}'
            elif "analysis notes present: false" in usr_lower:
                content = (
                    '{"next_agent": "analyst", "reason": "Analyze the gathered research notes"}'
                )
            elif "final answer draft present: false" in usr_lower:
                content = '{"next_agent": "writer", "reason": "Draft the final response report"}'
            elif "final answer draft present: true" in usr_lower:
                if "last critic feedback" not in usr_lower:
                    content = '{"next_agent": "critic", "reason": "Check quality of final draft response"}'
                else:
                    content = '{"next_agent": "done", "reason": "Review completed and approved"}'
            else:
                content = '{"next_agent": "done", "reason": "All steps completed"}'

        # 2. Critic Feedback
        elif "critic" in sys_lower or "fact-checker" in sys_lower:
            content = (
                "STATUS: APPROVED\n\n"
                "The final draft covers all claims in the research notes, maintains factual accuracy, "
                "and correctly cites all source documents."
            )

        # 3. Researcher note planning
        elif "research query planner" in sys_lower:
            content = "GraphRAG state-of-the-art, Microsoft GraphRAG indexing cost"

        # 4. Researcher notes synthesis
        elif "research specialist" in sys_lower:
            content = (
                "# Research Notes: Advanced RAG and Agentic Workflows\n\n"
                "## Overview\n"
                "GraphRAG combines Knowledge Graph construction with Retrieval-Augmented Generation [1]. "
                "It builds hierarchical communities of entities, enabling global query answers [2].\n\n"
                "## Key Findings\n"
                "1. Standard RAG struggles with global questions (e.g., 'What are the main themes in the dataset?') [1].\n"
                "2. GraphRAG structures raw text data into an entity-relationship graph using LLMs [2].\n"
                "3. LangGraph concepts allow constructing multi-agent architectures using state-based routing [3].\n"
            )

        # 5. Analyst claims extraction
        elif "senior research analyst" in sys_lower:
            content = (
                "# Structured Analysis Notes\n\n"
                "## Key Claims\n"
                "- Claim 1: GraphRAG outperforms baseline RAG on comprehensive global queries [1].\n"
                "- Claim 2: Decomposing tasks to multi-agent reduces context window load [2].\n\n"
                "## Viewpoint Analysis\n"
                "- Microsoft advocates for graph indexing prior to query, while others prefer raw vector search with re-ranking.\n\n"
                "## Evidence Evaluation\n"
                "- Microsoft's GraphRAG paper provides robust empirical benchmarks on narrative datasets [1].\n\n"
                "## Gaps & Open Problems\n"
                "- Graph indexing cost is extremely high due to multiple entity extraction calls. Cost reduction remains an open challenge."
            )

        # 6. Writer synthesis
        elif "technical writer" in sys_lower:
            content = (
                "# Research Brief: GraphRAG and Agentic Workflows\n\n"
                "## Executive Summary\n"
                "This brief evaluates Microsoft's GraphRAG and agentic workflows, focusing on capabilities and trade-offs. "
                "GraphRAG integrates knowledge graph extraction with retrieval-augmented generation to address global datasets query limits [1].\n\n"
                "## Deep Dive Analysis\n"
                "Traditional vector search baseline RAG models struggle to answer global thematic questions [2]. "
                "GraphRAG resolves this by structuring information into entity relation maps. "
                "Additionally, multi-agent frameworks using LangGraph allow dividing complex research flows into specialized roles, "
                "mitigating error rates and contextual overload [3].\n\n"
                "## References\n"
                "[1] GraphRAG: Unlocking LLM Knowledge on Complex Data (https://arxiv.org/abs/2404.16130)\n"
                "[2] Introducing GraphRAG from Microsoft Research (https://www.microsoft.com/en-us/research/blog/graphrag/)\n"
                "[3] LangGraph Concepts: Multi-Agent Systems (https://langchain-ai.github.io/langgraph/concepts/multi_agent/)"
            )

        # 7. Evaluator score
        elif "academic evaluator" in sys_lower or "quality evaluator" in sys_lower:
            content = "9.0"

        else:
            content = (
                f"Mock research output for query: '{usr_lower[:100]}'. "
                "The research indicates that advanced retrieval methods and multi-agent cooperation "
                "provide robust framework solutions for automated information gathering [1]."
            )

        # Calculate simulated tokens
        in_tokens = len(system_prompt.split()) + len(user_prompt.split())
        out_tokens = len(content.split())
        cost_usd = (in_tokens * 0.075 + out_tokens * 0.30) / 1_000_000

        return LLMResponse(
            content=content, input_tokens=in_tokens, output_tokens=out_tokens, cost_usd=cost_usd
        )
