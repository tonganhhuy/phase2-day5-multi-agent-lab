"""LangGraph workflow implementation."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self) -> None:
        self.llm_client = LLMClient()
        self.search_client = SearchClient()

        # Instantiate agents
        self.supervisor = SupervisorAgent(llm_client=self.llm_client)
        self.researcher = ResearcherAgent(
            llm_client=self.llm_client, search_client=self.search_client
        )
        self.analyst = AnalystAgent(llm_client=self.llm_client)
        self.writer = WriterAgent(llm_client=self.llm_client)
        self.critic = CriticAgent(llm_client=self.llm_client)

    def build(self) -> Any:
        """Create a LangGraph graph."""
        # 1. Initialize StateGraph using Pydantic schema or dict.
        # Since ResearchState is a Pydantic model, LangGraph supports Pydantic models directly.
        # To make it super robust with dict conversions, we can define the schema.
        workflow = StateGraph(ResearchState)

        # 2. Add nodes
        workflow.add_node("supervisor", lambda state: self.supervisor.run(state))
        workflow.add_node("researcher", lambda state: self.researcher.run(state))
        workflow.add_node("analyst", lambda state: self.analyst.run(state))
        workflow.add_node("writer", lambda state: self.writer.run(state))
        workflow.add_node("critic", lambda state: self.critic.run(state))

        # 3. Add entry point
        workflow.add_edge(START, "supervisor")

        # 4. Define routing decision helper
        def route_next(state: ResearchState) -> str:
            if not state.route_history:
                # Default safety fallback
                return "done"

            last_agent = state.route_history[-1]
            if last_agent == "researcher":
                return "researcher"
            elif last_agent == "analyst":
                return "analyst"
            elif last_agent == "writer":
                return "writer"
            elif last_agent == "critic":
                return "critic"
            else:
                return END

        # 5. Add conditional edges from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            route_next,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                END: END,
            },
        )

        # 6. Add standard edges back to supervisor
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("writer", "supervisor")
        workflow.add_edge("critic", "supervisor")

        return workflow.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        app = self.build()

        # Invoke the compiled LangGraph application.
        # LangGraph invoke outputs the final state dictionary or Pydantic model.
        output = app.invoke(state)

        # Convert output back to ResearchState if it returned a dict or object
        if isinstance(output, ResearchState):
            return output
        elif isinstance(output, dict):
            return ResearchState(**output)
        else:
            # Fallback parsing
            return ResearchState.model_validate(output)
