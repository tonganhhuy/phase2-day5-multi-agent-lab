import pytest
from unittest.mock import MagicMock
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMResponse


def test_supervisor_routes_using_mock_llm() -> None:
    # Arrange
    mock_llm = MagicMock()
    # Mock LLM response return JSON format
    mock_llm.complete.return_value = LLMResponse(
        content='{"next_agent": "researcher", "reason": "need facts"}'
    )
    
    supervisor = SupervisorAgent(llm_client=mock_llm)
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    
    # Act
    res_state = supervisor.run(state)
    
    # Assert
    assert res_state.route_history[-1] == "researcher"
    assert res_state.iteration == 1


def test_analyst_agent_runs_with_mock_llm() -> None:
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLMResponse(content="Detailed claims analysis")
    
    analyst = AnalystAgent(llm_client=mock_llm)
    state = ResearchState(request=ResearchQuery(query="Query"))
    state.research_notes = "Fact: LangGraph is a library."
    
    res_state = analyst.run(state)
    
    assert res_state.analysis_notes == "Detailed claims analysis"
    assert len(res_state.agent_results) == 1
    assert res_state.agent_results[0].agent == AgentName.ANALYST
