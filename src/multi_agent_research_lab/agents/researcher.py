"""Researcher agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self, llm_client: LLMClient | None = None, search_client: SearchClient | None = None
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query

        # Ask LLM to generate 2 search queries
        system_prompt = (
            "You are a research query planner. Break down the user's research request into "
            "exactly 2 distinct, high-quality search queries optimized for a search engine. "
            "Output them as a comma-separated list on a single line. Do not include numbering or quotes."
        )
        user_prompt = f"Original Query: {query}"

        try:
            llm_res = self.llm_client.complete(system_prompt, user_prompt)
            search_queries = [q.strip() for q in llm_res.content.split(",") if q.strip()]
        except Exception:
            search_queries = [query]

        if not search_queries:
            search_queries = [query]

        # Perform searches and collect documents
        new_sources = []
        for s_query in search_queries:
            try:
                docs = self.search_client.search(s_query, max_results=state.request.max_sources)
                new_sources.extend(docs)
            except Exception:
                pass

        # Deduplicate sources by URL (or title if URL is missing)
        seen = set(s.url for s in state.sources if s.url)
        for doc in new_sources:
            identifier = doc.url or doc.title
            if not doc.url or doc.url not in seen:
                state.sources.append(doc)
                if doc.url:
                    seen.add(doc.url)

        # Synthesize research notes
        sources_text = ""
        for idx, src in enumerate(state.sources):
            sources_text += f"[{idx + 1}] Title: {src.title}\nURL: {src.url or 'N/A'}\nSnippet: {src.snippet}\n\n"

        notes_system_prompt = (
            "You are an expert Research Specialist. Synthesize the provided search results into a cohesive "
            "and objective set of research notes. Organize the notes with clear sections (e.g. Overview, Key Facts, Insights). "
            "You MUST cite facts using source numbers like [1], [2], etc., corresponding to the sources provided. "
            "Be factually rigorous. Do not invent details not present in the sources."
        )
        notes_user_prompt = f"Query: {query}\n\nSearch Results:\n{sources_text}"

        llm_res_notes = self.llm_client.complete(notes_system_prompt, notes_user_prompt)
        state.research_notes = llm_res_notes.content

        # Save to agent results
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=llm_res_notes.content,
                metadata={"queries_used": search_queries, "sources_fetched": len(new_sources)},
            )
        )

        # Add trace event
        state.add_trace_event(
            "researcher_run", {"queries": search_queries, "sources_count": len(state.sources)}
        )

        return state
