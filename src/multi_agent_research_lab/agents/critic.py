"""Critic agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class CriticAgent(BaseAgent):
    """Fact-checks and reviews the quality of the final draft."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""
        final_answer = state.final_answer or ""
        research_notes = state.research_notes or ""

        system_prompt = (
            "You are a Quality Critic and Fact-Checker. Compare the final answer draft against the research notes "
            "to ensure absolute factual accuracy and robust citation coverage.\n\n"
            "Identify:\n"
            "- Any hallucinations (claims not supported by the notes).\n"
            "- Missing citations for key claims.\n"
            "- Structure/style issues.\n\n"
            "Your output MUST begin with a status line, choosing exactly one of the options below:\n"
            "STATUS: APPROVED\n"
            "STATUS: NEEDS_REVISION\n\n"
            "Follow the status line with detailed feedback explaining your decision."
        )

        user_prompt = f"Research Notes:\n{research_notes}\n\nFinal Draft Answer:\n{final_answer}"

        llm_res = self.llm_client.complete(system_prompt, user_prompt)
        content = llm_res.content

        # Save to agent results
        state.agent_results.append(AgentResult(agent=AgentName.CRITIC, content=content))

        # Determine approval
        approved = "STATUS: APPROVED" in content

        # Add trace event
        state.add_trace_event(
            "critic_run",
            {"approved": approved, "status_line": "APPROVED" if approved else "NEEDS_REVISION"},
        )

        return state
