"""Base agent class with dependency injection support."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from models.schemas import ConversationState, AgentType, DataCollectionResult


class BaseAgent(ABC):
    """Base class for all agents with dependency injection."""
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        agent_type: Optional[AgentType] = None,
        **kwargs
    ):
        """
        Initialize base agent with dependency injection.
        
        Args:
            llm: Language model instance (injected dependency)
            agent_type: Type of this agent
            **kwargs: Additional configuration
        """
        self.llm = llm or ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
        )
        self.agent_type = agent_type
        self.config = kwargs
    
    @abstractmethod
    async def process(self, state: ConversationState) -> ConversationState:
        """
        Process the conversation state and return updated state.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated conversation state
        """
        pass
    
    def extract_data(self, state: ConversationState) -> DataCollectionResult:
        """
        Extract structured data from the conversation.
        
        Args:
            state: Current conversation state
            
        Returns:
            Data collection result
        """
        return DataCollectionResult(
            agent_type=self.agent_type,
            data={},
            success=False,
        )

