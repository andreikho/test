"""Agent 1: Collects data fields a, b, c through conversation."""
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import ConversationState, Agent1Data, AgentType, DataCollectionResult
from agents.base_agent import BaseAgent


class Agent1(BaseAgent):
    """Agent 1 collects fields a, b, c through conversational questioning."""
    
    def __init__(self, llm=None, **kwargs):
        super().__init__(llm=llm, agent_type=AgentType.AGENT_1, **kwargs)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Agent 1, responsible for collecting three related pieces of information: field_a, field_b, and field_c.

The user may not know that these fields belong together. Your job is to:
1. Acknowledge what they want to provide
2. Ask for ALL THREE fields (a, b, c) one by one in a friendly, conversational way
3. Keep track of what has already been collected

IMPORTANT: 
- If this is a new conversation, welcome them and ask for field_a first
- If some fields are already collected, ask for the missing ones
- When you have all three, confirm the collected data

Already collected data: {collected_data}

Respond conversationally and ask for the next missing field."""),
            ("human", """Conversation history:
{history}

User just said: {user_input}

Respond helpfully and ask for the next field if needed."""),
        ])
    
    async def process(self, state: ConversationState) -> ConversationState:
        """Process data collection for Agent 1 through conversation."""
        history = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in state.messages[-10:]
        ])
        
        # Get existing collected data for this agent
        existing_data = {}
        if self.agent_type in state.collected_data:
            existing_data = state.collected_data[self.agent_type].data
        
        # Check context for partial collection
        agent_context = state.context.get("agent_1_data", {})
        existing_data.update(agent_context)
        
        chain = self.prompt | self.llm
        response = await chain.ainvoke({
            "history": history or "No previous conversation",
            "user_input": state.user_input,
            "collected_data": existing_data if existing_data else "Nothing collected yet",
        })
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Try to extract data from user input
        user_input_lower = state.user_input.lower()
        
        # Simple extraction logic - in real app, use LLM for extraction
        if "field_a" not in existing_data and any(keyword in user_input_lower for keyword in ["a:", "a is", "a ="]):
            # Extract value after the keyword
            pass  # Would need more sophisticated extraction
        
        # Store partial data in context
        state.context["agent_1_data"] = existing_data
        
        # Check if all fields collected
        if all(k in existing_data for k in ["field_a", "field_b", "field_c"]):
            result = DataCollectionResult(
                agent_type=self.agent_type,
                data=existing_data,
                success=True,
            )
            state.collected_data[self.agent_type] = result
        
        # Add agent response to history
        state.messages.append({
            "role": "assistant",
            "content": response_text,
        })
        
        return state
    
    def extract_data(self, state: ConversationState) -> DataCollectionResult:
        """Extract structured data."""
        return state.collected_data.get(self.agent_type, DataCollectionResult(
            agent_type=self.agent_type,
            data={},
            success=False,
        ))
