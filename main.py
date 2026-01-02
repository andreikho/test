"""Main application entry point."""
import asyncio
import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from models.schemas import ConversationState, AgentType, DataCollectionResult
from graph.multi_agent_graph import create_multi_agent_graph

# Load environment variables
load_dotenv()


def dict_to_state(data: Dict[str, Any]) -> ConversationState:
    """Convert dictionary from LangGraph back to ConversationState."""
    # Handle collected_data conversion
    collected_data = {}
    if "collected_data" in data and data["collected_data"]:
        for key, value in data["collected_data"].items():
            if isinstance(key, str):
                agent_type = AgentType(key)
            else:
                agent_type = key
            
            if isinstance(value, dict):
                collected_data[agent_type] = DataCollectionResult(**value)
            else:
                collected_data[agent_type] = value
    
    # Handle current_agent conversion
    current_agent = data.get("current_agent")
    if isinstance(current_agent, str):
        current_agent = AgentType(current_agent)
    
    return ConversationState(
        messages=data.get("messages", []),
        current_agent=current_agent,
        collected_data=collected_data,
        user_input=data.get("user_input", ""),
        context=data.get("context", {}),
    )


def state_to_dict(state: ConversationState) -> Dict[str, Any]:
    """Convert ConversationState to dictionary for LangGraph."""
    collected_data = {}
    for agent_type, result in state.collected_data.items():
        key = agent_type.value if isinstance(agent_type, AgentType) else agent_type
        if isinstance(result, DataCollectionResult):
            collected_data[key] = result.model_dump()
        else:
            collected_data[key] = result
    
    return {
        "messages": state.messages,
        "current_agent": state.current_agent.value if state.current_agent else None,
        "collected_data": collected_data,
        "user_input": state.user_input,
        "context": state.context,
    }


async def main():
    """Main application loop."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        return
    
    # Initialize LLM with dependency injection
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=api_key,
    )
    
    # Create the multi-agent graph
    app = create_multi_agent_graph(llm=llm)
    
    # Initialize conversation state as dictionary
    state_dict = {
        "messages": [],
        "current_agent": None,
        "collected_data": {},
        "user_input": "",
        "context": {},
    }
    
    print("🤖 Multi-Agent Data Collection System")
    print("=" * 50)
    print("Type 'exit' to quit, 'summary' to get a summary of collected data")
    print("=" * 50)
    print()
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Update state with user input
        state_dict["user_input"] = user_input
        state_dict["messages"].append({
            "role": "user",
            "content": user_input,
        })
        
        # Run the graph
        try:
            result = await app.ainvoke(state_dict)
            state_dict = result  # LangGraph returns a dict
            
            # Display the last assistant message
            messages = state_dict.get("messages", [])
            if messages:
                last_message = messages[-1]
                if last_message.get("role") == "assistant":
                    print(f"\nAssistant: {last_message.get('content', '')}\n")
            
            # Show collected data status
            collected_data = state_dict.get("collected_data", {})
            if collected_data:
                print("Collected data:")
                for agent_type, result_data in collected_data.items():
                    if isinstance(result_data, dict):
                        success = result_data.get("success", False)
                        data = result_data.get("data", {})
                    else:
                        success = result_data.success
                        data = result_data.data
                    status = "✓" if success else "✗"
                    print(f"  {status} {agent_type}: {len(data)} fields")
                print()
            
        except Exception as e:
            print(f"Error: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
