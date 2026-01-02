"""Router agent for intent-based routing."""
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from models.schemas import ConversationState, AgentIntent, AgentType
from agents.base_agent import BaseAgent


class RouterAgent(BaseAgent):
    """Router agent that analyzes intent and routes to appropriate agents."""
    
    def __init__(self, llm=None, **kwargs):
        super().__init__(llm=llm, agent_type=None, **kwargs)
        self.parser = PydanticOutputParser(pydantic_object=AgentIntent)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a router agent that analyzes user intent and routes to the appropriate specialized agent.

ROUTING RULES - Route based on which fields the user mentions or wants to provide:

- If user mentions fields "a", "b", or "c" (or wants to provide info related to these) → route to agent_1
- If user mentions fields "d" or "e" (or wants to provide info related to these) → route to agent_2  
- If user mentions fields "f", "g", or "h" (or wants to provide info related to these) → route to agent_3
- If user wants to upload or process a PDF document → route to pdf_agent
- If user asks for a summary of collected data → route to summary_agent

Examples:
- "I want to input a" → agent_1
- "I have data for field b" → agent_1
- "let me give you c" → agent_1
- "I want to provide d" → agent_2
- "here's my e information" → agent_2
- "I need to enter f, g" → agent_3
- "upload my document.pdf" → pdf_agent
- "give me a summary" → summary_agent

Analyze the user's input carefully to determine which field(s) they want to provide.

{format_instructions}"""),
            ("human", """User input: {user_input}

Conversation history:
{history}

Determine which agent to route to based on the fields the user wants to provide."""),
        ])
    
    async def process(self, state: ConversationState) -> ConversationState:
        """Process routing decision."""
        # Format conversation history
        history = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in state.messages[-5:]  # Last 5 messages for context
        ])
        
        # Get routing decision
        chain = self.prompt | self.llm | self.parser
        intent: AgentIntent = await chain.ainvoke({
            "user_input": state.user_input,
            "history": history or "No previous conversation",
            "format_instructions": self.parser.get_format_instructions(),
        })
        
        # Update state with routing decision
        state.current_agent = intent.intent
        state.context["routing_decision"] = {
            "intent": intent.intent.value,
            "confidence": intent.confidence,
            "reasoning": intent.reasoning,
        }
        
        # Add router message to history
        state.messages.append({
            "role": "router",
            "content": f"Routing to {intent.intent.value} (confidence: {intent.confidence:.2f}). {intent.reasoning}",
        })
        
        return state
