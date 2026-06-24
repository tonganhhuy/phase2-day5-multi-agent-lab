"""Search client abstraction for ResearcherAgent."""

import requests

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client implementing Tavily with Mock fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Uses Tavily API if TAVILY_API_KEY is configured, otherwise falls back to intelligent mock data.
        """
        api_key = self.settings.tavily_api_key
        if not api_key:
            return self._mock_search(query, max_results)

        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": query, "max_results": max_results},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            results = []
            for item in data.get("results", []):
                results.append(
                    SourceDocument(
                        title=item.get("title", "No Title"),
                        url=item.get("url"),
                        snippet=item.get("content", ""),
                    )
                )
            return results
        except Exception:
            # Fallback to local mock data on network errors or bad responses
            return self._mock_search(query, max_results)

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Return high-quality static search responses based on typical queries."""
        q = query.lower()
        results = []

        if "graphrag" in q:
            results.append(
                SourceDocument(
                    title="GraphRAG: Unlocking LLM Knowledge on Complex Data",
                    url="https://arxiv.org/abs/2404.16130",
                    snippet="Microsoft's GraphRAG uses knowledge graphs to structure information prior to generating LLM completions. It maps entities and relationships, clustering them into hierarchical communities, which allows the LLM to answer global query questions.",
                )
            )
            results.append(
                SourceDocument(
                    title="Introducing GraphRAG from Microsoft Research",
                    url="https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/",
                    snippet="GraphRAG combines Knowledge Graph construction with Retrieval-Augmented Generation. It demonstrates significant improvements over baseline RAG in comprehensive answers, entity connection, and answering high-level questions.",
                )
            )

        if "multi-agent" in q or "single-agent" in q or "agent" in q or "workflow" in q:
            results.append(
                SourceDocument(
                    title="Building Effective Agents - Anthropic Research",
                    url="https://www.anthropic.com/engineering/building-effective-agents",
                    snippet="Anthropic discusses patterns of agentic workflows: workflow patterns like chain, router, parallelization, orchestrator-workers, and evaluator-optimizer. They emphasize keeping designs simple and starting with single-agent architectures first.",
                )
            )
            results.append(
                SourceDocument(
                    title="OpenAI Agents SDK Orchestration and Handoffs",
                    url="https://developers.openai.com/api/docs/guides/agents/orchestration",
                    snippet="OpenAI outlines patterns for multi-agent handoffs, where a primary router delegates sub-tasks to specialized agents based on user intent. This reduces prompt size, splits context, and improves performance.",
                )
            )
            results.append(
                SourceDocument(
                    title="LangGraph Concepts: Multi-Agent Systems",
                    url="https://langchain-ai.github.io/langgraph/concepts/multi_agent/",
                    snippet="LangGraph supports constructing multi-agent systems using graphs where nodes represent agents and edges represent communication or control flow. Shared state is updated incrementally as agents yield new values.",
                )
            )

        if not results:
            results.append(
                SourceDocument(
                    title=f"General Research on: {query}",
                    url="https://example.com/research",
                    snippet=f"Mock search result content for '{query}'. AI research systems utilize multi-agent routing policies to distribute sub-tasks to expert researchers and analysts, producing synthesized reports.",
                )
            )

        return results[:max_results]
