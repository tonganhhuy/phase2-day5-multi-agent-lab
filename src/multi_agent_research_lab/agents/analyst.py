"""Analyst agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        research_notes = state.research_notes or "No research notes available."

        system_prompt = (
            "You are a Senior Research Analyst. Your job is to critically evaluate and analyze the provided research notes. "
            "Specifically, you must:\n"
            "1. Extract key claims or findings.\n"
            "2. Compare different viewpoints, methodologies, or paradigms if present.\n"
            "3. Identify the strength of the evidence (e.g. peer-reviewed vs. blog post claims).\n"
            "4. Highlight any obvious knowledge gaps, uncertainties, or potential contradictions.\n\n"
            "Output your analysis in a structured format with headings: 'Key Claims', 'Viewpoint Analysis', "
            "'Evidence Evaluation', and 'Gaps & Open Problems'."
        )

        user_prompt = f"Research Notes:\n{research_notes}"

        llm_res = self.llm_client.complete(system_prompt, user_prompt)
        state.analysis_notes = llm_res.content

        # Save to agent results
        state.agent_results.append(AgentResult(agent=AgentName.ANALYST, content=llm_res.content))

        # Add trace event
        state.add_trace_event("analyst_run", {})

        return state
