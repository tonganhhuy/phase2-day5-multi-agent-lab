"""Supervisor / router implementation."""

import json

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()
        self.settings = get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        # 1. Enforce max iterations guardrail
        if state.iteration >= self.settings.max_iterations:
            if state.final_answer:
                state.record_route("done")
            else:
                state.record_route("writer")
            state.add_trace_event(
                "supervisor_max_iterations_reached",
                {"iteration": state.iteration, "max_iterations": self.settings.max_iterations},
            )
            return state

        # Determine if we have previous critic feedback
        critic_feedback = ""
        critic_results = [r for r in state.agent_results if r.agent == AgentName.CRITIC]
        if critic_results:
            critic_feedback = f"Last Critic Feedback:\n{critic_results[-1].content}\n"

        system_prompt = (
            "You are the Supervisor/Router of a collaborative AI research team.\n"
            "Your job is to coordinate tasks between the following specialists:\n"
            "- researcher: Finds sources, retrieves information, and writes research notes.\n"
            "- analyst: Reads research notes to analyze claims, evaluate evidence, and structure viewpoints.\n"
            "- writer: Synthesizes research and analysis notes into a structured research brief.\n"
            "- critic: Fact-checks the writer's final draft and points out gaps or errors.\n"
            "- done: Finish the workflow and output the final response.\n\n"
            "Routing Rules:\n"
            "1. If sources or research notes are missing, route to 'researcher'.\n"
            "2. If research notes exist but analysis notes are missing, route to 'analyst'.\n"
            "3. If analysis notes exist but the final answer is missing, route to 'writer'.\n"
            "4. If the final answer exists, route to 'critic' to review. If the critic APPROVED the report, "
            "route to 'done'. If the critic flagged issues (NEEDS_REVISION), route to 'writer' to fix them (or "
            "'researcher' if more data is required).\n"
            "5. To avoid infinite loops: If you have already revised once or twice, route to 'done' to wrap up.\n\n"
            "You must return ONLY a JSON object containing the fields:\n"
            "{\n"
            '  "next_agent": "researcher" | "analyst" | "writer" | "critic" | "done",\n'
            '  "reason": "Clear explanation of the decision"\n'
            "}"
        )

        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Current Iteration: {state.iteration} / Max: {self.settings.max_iterations}\n"
            f"Route History: {state.route_history}\n"
            f"Research Notes Present: {bool(state.research_notes)}\n"
            f"Analysis Notes Present: {bool(state.analysis_notes)}\n"
            f"Final Answer Draft Present: {bool(state.final_answer)}\n"
            f"{critic_feedback}"
        )

        try:
            llm_res = self.llm_client.complete(system_prompt, user_prompt)
            content = llm_res.content.strip()

            # Clean LLM output if wrapped in code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)
            next_agent = data.get("next_agent", "done")
        except Exception:
            # Fallback static routing if LLM fail
            if not state.research_notes:
                next_agent = "researcher"
            elif not state.analysis_notes:
                next_agent = "analyst"
            elif not state.final_answer:
                next_agent = "writer"
            else:
                next_agent = "done"

        # Safe guard to ensure agent value is valid
        valid_agents = ["researcher", "analyst", "writer", "critic", "done"]
        if next_agent not in valid_agents:
            next_agent = "done"

        # Safe guard to prevent immediate loops
        # If the supervisor attempts to route to researcher or analyst but we already ran them
        # and there is a critic feedback that got approved or we are hitting high iterations, force wrap up.
        if next_agent == "critic" and state.route_history and state.route_history[-1] == "critic":
            next_agent = "done"

        state.record_route(next_agent)
        state.add_trace_event(
            "supervisor_decision", {"next_agent": next_agent, "iteration": state.iteration}
        )

        return state
