"""Structured output models for the multi-agent system."""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class AgentType(str, Enum):
    """Types of agents in the system."""
    AGENT_1 = "agent_1"  # Collects data: a, b, c
    AGENT_2 = "agent_2"  # Collects data: d, e
    AGENT_3 = "agent_3"  # Collects data: f, g, h
    PDF_AGENT = "pdf_agent"  # Loads and processes PDFs
    SUMMARY_AGENT = "summary_agent"  # Final summary


class AgentIntent(BaseModel):
    """Intent classification result from router agent."""
    intent: AgentType = Field(description="The type of agent to route to")
    confidence: float = Field(description="Confidence score (0-1)", ge=0, le=1)
    reasoning: str = Field(description="Explanation for the routing decision")


class Agent1Data(BaseModel):
    """Structured data collected by Agent 1."""
    field_a: str = Field(description="Data field a")
    field_b: str = Field(description="Data field b")
    field_c: str = Field(description="Data field c")


class Agent2Data(BaseModel):
    """Structured data collected by Agent 2."""
    field_d: str = Field(description="Data field d")
    field_e: str = Field(description="Data field e")


class Agent3Data(BaseModel):
    """Structured data collected by Agent 3."""
    field_f: str = Field(description="Data field f")
    field_g: str = Field(description="Data field g")
    field_h: str = Field(description="Data field h")


class PDFData(BaseModel):
    """Structured data extracted from PDF."""
    filename: str = Field(description="Name of the PDF file")
    content: str = Field(description="Extracted text content from PDF")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="PDF metadata")
    page_count: int = Field(description="Number of pages in the PDF")


class DataCollectionResult(BaseModel):
    """Result from a data collection agent."""
    agent_type: Union[AgentType, str] = Field(description="Type of agent")
    data: Dict[str, Any] = Field(default_factory=dict, description="Collected structured data")
    success: bool = Field(default=False, description="Whether collection was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    
    @field_validator("agent_type", mode="before")
    @classmethod
    def convert_agent_type(cls, v):
        if isinstance(v, str):
            return AgentType(v)
        return v


class ConversationState(BaseModel):
    """State maintained across conversation turns."""
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="Conversation history")
    current_agent: Optional[Union[AgentType, str]] = Field(default=None, description="Currently active agent")
    collected_data: Dict[Union[AgentType, str], Union[DataCollectionResult, Dict[str, Any]]] = Field(
        default_factory=dict, description="Data collected by each agent"
    )
    user_input: str = Field(default="", description="Current user input")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    
    @field_validator("current_agent", mode="before")
    @classmethod
    def convert_current_agent(cls, v):
        if isinstance(v, str):
            try:
                return AgentType(v)
            except ValueError:
                return None
        return v
    
    @model_validator(mode="before")
    @classmethod
    def convert_collected_data(cls, data):
        if isinstance(data, dict) and "collected_data" in data:
            collected = data.get("collected_data", {})
            if collected:
                new_collected = {}
                for key, value in collected.items():
                    # Convert string key to AgentType
                    if isinstance(key, str):
                        try:
                            agent_type = AgentType(key)
                        except ValueError:
                            agent_type = key
                    else:
                        agent_type = key
                    
                    # Convert dict value to DataCollectionResult
                    if isinstance(value, dict):
                        new_collected[agent_type] = DataCollectionResult(**value)
                    else:
                        new_collected[agent_type] = value
                
                data["collected_data"] = new_collected
        return data
