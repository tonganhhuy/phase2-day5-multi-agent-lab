"""Writer agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        research_notes = state.research_notes or "No research notes available."
        analysis_notes = state.analysis_notes or "No analysis notes available."
        audience = state.request.audience
        query = state.request.query

        # Format sources list
        sources_list = ""
        for idx, src in enumerate(state.sources):
            sources_list += f"[{idx + 1}] {src.title} - {src.url or 'No URL'}\n"

        system_prompt = (
            f"You are a professional Technical Writer. Synthesize the provided research notes and analysis notes "
            f"into a clear, comprehensive research brief customized for the audience: '{audience}'.\n\n"
            "Guidelines:\n"
            "1. Address the original research query fully.\n"
            "2. Keep a structured layout using markdown headings.\n"
            "3. Embed source citations (e.g. [1], [2]) seamlessly in the text.\n"
            "4. End with a compiled 'References' section that lists all sources and their URLs.\n"
            "5. Ensure the tone is objective and scholarly."
        )

        user_prompt = (
            f"Original Query: {query}\n\n"
            f"Research Notes:\n{research_notes}\n\n"
            f"Analysis Notes:\n{analysis_notes}\n\n"
            f"Source Documents:\n{sources_list}"
        )

        llm_res = self.llm_client.complete(system_prompt, user_prompt)
        state.final_answer = llm_res.content

        # Save to agent results
        state.agent_results.append(AgentResult(agent=AgentName.WRITER, content=llm_res.content))

        # Add trace event
        state.add_trace_event("writer_run", {})

        return state
