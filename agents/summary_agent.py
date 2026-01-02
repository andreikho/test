"""Summary Agent: Provides final summary of collected data."""
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import ConversationState, AgentType
from agents.base_agent import BaseAgent


class SummaryAgent(BaseAgent):
    """Agent that provides a final summary of all collected data."""
    
    def __init__(self, llm=None, **kwargs):
        super().__init__(llm=llm, agent_type=AgentType.SUMMARY_AGENT, **kwargs)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a summary agent that provides comprehensive summaries of all data collected by the specialized agents.

Review all collected data from Agent 1, Agent 2, Agent 3, and PDF Agent, and provide a clear, structured summary."""),
            ("human", """Collected data from all agents:
{collected_data}

Conversation history:
{history}

Provide a comprehensive summary of all collected information."""),
        ])
    
    async def process(self, state: ConversationState) -> ConversationState:
        """Generate summary of all collected data."""
        # Format collected data
        collected_data_str = "\n\n".join([
            f"{result.agent_type.value}:\n{result.data}"
            for result in state.collected_data.values()
            if result.success
        ])
        
        if not collected_data_str:
            collected_data_str = "No data has been collected yet."
        
        history = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in state.messages[-20:]
        ])
        
        chain = self.prompt | self.llm
        response = await chain.ainvoke({
            "collected_data": collected_data_str,
            "history": history or "No previous conversation",
        })
        
        # Add summary to messages
        summary_content = response.content if hasattr(response, 'content') else str(response)
        state.messages.append({
            "role": "assistant",
            "content": f"Summary:\n{summary_content}",
        })
        
        return state

