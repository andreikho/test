"""LangGraph implementation of the multi-agent system."""
from typing import Literal, Dict, Any, List, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from models.schemas import AgentType
from agents.router_agent import RouterAgent
from agents.agent_1 import Agent1
from agents.agent_2 import Agent2
from agents.agent_3 import Agent3
from agents.pdf_agent import PDFAgent
from agents.summary_agent import SummaryAgent
from langchain_openai import ChatOpenAI


class GraphState(TypedDict):
    """State for the LangGraph - using TypedDict for compatibility."""
    messages: List[Dict[str, Any]]
    current_agent: Optional[str]
    collected_data: Dict[str, Any]
    user_input: str
    context: Dict[str, Any]


def create_multi_agent_graph(llm: ChatOpenAI = None):
    """
    Create the multi-agent LangGraph with dependency injection.
    
    Args:
        llm: Language model instance (injected dependency)
        
    Returns:
        Compiled LangGraph
    """
    # Initialize LLM if not provided
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Initialize agents with dependency injection
    router = RouterAgent(llm=llm)
    agent1 = Agent1(llm=llm)
    agent2 = Agent2(llm=llm)
    agent3 = Agent3(llm=llm)
    pdf_agent = PDFAgent(llm=llm)
    summary_agent = SummaryAgent(llm=llm)
    
    # Wrapper functions to handle dict state
    async def router_node(state: GraphState) -> GraphState:
        from models.schemas import ConversationState
        conv_state = ConversationState(**state)
        result = await router.process(conv_state)
        return {
            "messages": result.messages,
            "current_agent": result.current_agent.value if result.current_agent else None,
            "collected_data": {k.value: v.model_dump() for k, v in result.collected_data.items()},
            "user_input": result.user_input,
            "context": result.context,
        }
    
    async def agent1_node(state: GraphState) -> GraphState:
        from models.schemas import ConversationState
        conv_state = ConversationState(**state)
        result = await agent1.process(conv_state)
        return {
            "messages": result.messages,
            "current_agent": result.current_agent.value if result.current_agent else None,
            "collected_data": {k.value: v.model_dump() for k, v in result.collected_data.items()},
            "user_input": result.user_input,
            "context": result.context,
        }
    
    async def agent2_node(state: GraphState) -> GraphState:
        from models.schemas import ConversationState
        conv_state = ConversationState(**state)
        result = await agent2.process(conv_state)
        return {
            "messages": result.messages,
            "current_agent": result.current_agent.value if result.current_agent else None,
            "collected_data": {k.value: v.model_dump() for k, v in result.collected_data.items()},
            "user_input": result.user_input,
            "context": result.context,
        }
    
    async def agent3_node(state: GraphState) -> GraphState:
        from models.schemas import ConversationState
        conv_state = ConversationState(**state)
        result = await agent3.process(conv_state)
        return {
            "messages": result.messages,
            "current_agent": result.current_agent.value if result.current_agent else None,
            "collected_data": {k.value: v.model_dump() for k, v in result.collected_data.items()},
            "user_input": result.user_input,
            "context": result.context,
        }
    
    async def pdf_agent_node(state: GraphState) -> GraphState:
        from models.schemas import ConversationState
        conv_state = ConversationState(**state)
        result = await pdf_agent.process(conv_state)
        return {
            "messages": result.messages,
            "current_agent": result.current_agent.value if result.current_agent else None,
            "collected_data": {k.value: v.model_dump() for k, v in result.collected_data.items()},
            "user_input": result.user_input,
            "context": result.context,
        }
    
    async def summary_agent_node(state: GraphState) -> GraphState:
        from models.schemas import ConversationState
        conv_state = ConversationState(**state)
        result = await summary_agent.process(conv_state)
        return {
            "messages": result.messages,
            "current_agent": result.current_agent.value if result.current_agent else None,
            "collected_data": {k.value: v.model_dump() for k, v in result.collected_data.items()},
            "user_input": result.user_input,
            "context": result.context,
        }
    
    # Create the graph with TypedDict state
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("agent_1", agent1_node)
    workflow.add_node("agent_2", agent2_node)
    workflow.add_node("agent_3", agent3_node)
    workflow.add_node("pdf_agent", pdf_agent_node)
    workflow.add_node("summary_agent", summary_agent_node)
    
    # Define routing function
    def route_after_router(state: GraphState) -> Literal["agent_1", "agent_2", "agent_3", "pdf_agent", "summary_agent", "__end__"]:
        """Route to the appropriate agent based on router decision."""
        current_agent = state.get("current_agent")
        if current_agent is None:
            return END
        
        agent_map = {
            "agent_1": "agent_1",
            "agent_2": "agent_2",
            "agent_3": "agent_3",
            "pdf_agent": "pdf_agent",
            "summary_agent": "summary_agent",
        }
        
        return agent_map.get(current_agent, END)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add edges
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "agent_1": "agent_1",
            "agent_2": "agent_2",
            "agent_3": "agent_3",
            "pdf_agent": "pdf_agent",
            "summary_agent": "summary_agent",
            END: END,
        }
    )
    
    # All agents return to END (can be modified to loop back to router)
    workflow.add_edge("agent_1", END)
    workflow.add_edge("agent_2", END)
    workflow.add_edge("agent_3", END)
    workflow.add_edge("pdf_agent", END)
    workflow.add_edge("summary_agent", END)
    
    # Compile the graph
    return workflow.compile()
