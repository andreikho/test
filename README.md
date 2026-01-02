# 🤖 Multi-Agent Data Collection System

A LangGraph-based multi-agent system for intelligent data collection with intent-based routing.

## Overview

This system implements a router agent that intelligently directs users to specialized data collection agents:

- **Router Agent**: Analyzes user intent and routes to appropriate agents
- **Agent 1**: Collects general data (fields a, b, c)
- **Agent 2**: Collects specific data (fields d, e)
- **Agent 3**: Collects detailed data (fields f, g, h)
- **PDF Agent**: Loads and processes PDF documents
- **Summary Agent**: Provides final summary of collected data

## Features

✅ **AI Agents**: Multiple specialized agents for different data collection tasks  
✅ **Structured Output**: Pydantic models for type-safe data extraction  
✅ **Intent-Based Routing**: Router agent analyzes user intent and routes accordingly  
✅ **PDF Processing**: One agent can load and extract content from PDF files  
✅ **Conversation History**: Maintains context across multiple conversation turns  
✅ **Dependency Injection**: Agents receive dependencies (LLM) via constructor injection  
✅ **LangGraph**: Built with LangGraph for state management and agent orchestration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INPUT                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     ROUTER AGENT                             │
│  Analyzes intent and routes to appropriate agent             │
└─────────────────────────────────────────────────────────────┘
          │              │              │              │
          ▼              ▼              ▼              ▼
     ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
     │ Agent 1 │   │ Agent 2 │   │ Agent 3 │   │  PDF    │
     │ a, b, c │   │  d, e   │   │ f, g, h │   │  Agent  │
     └─────────┘   └─────────┘   └─────────┘   └─────────┘
                              │
                              ▼
                    ┌───────────────┐
                    │ Summary Agent │
                    └───────────────┘
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Create PDF Directory (Optional)

If you want to use the PDF agent, create a directory for PDF files:

```bash
mkdir pdfs
```

## Usage

Run the main application:

```bash
python main.py
```

### Example Interactions

1. **General Data Collection**:
   ```
   You: I need to provide information about my company
   ```
   The router will likely route to Agent 1 to collect fields a, b, c.

2. **PDF Processing**:
   ```
   You: Please process the document.pdf file
   ```
   The router will route to the PDF agent to load and extract content.

3. **Get Summary**:
   ```
   You: summary
   ```
   The router will route to the summary agent to provide a comprehensive summary.

## Project Structure

```
.
├── agents/              # Agent implementations
│   ├── base_agent.py   # Base agent class with DI
│   ├── router_agent.py # Router agent
│   ├── agent_1.py      # Agent 1 (fields a, b, c)
│   ├── agent_2.py      # Agent 2 (fields d, e)
│   ├── agent_3.py      # Agent 3 (fields f, g, h)
│   ├── pdf_agent.py    # PDF processing agent
│   └── summary_agent.py # Summary agent
├── models/             # Data models
│   └── schemas.py      # Pydantic schemas
├── graph/              # LangGraph implementation
│   └── multi_agent_graph.py
├── main.py             # Application entry point
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Key Components

### Structured Output Models

All agents use Pydantic models for structured data extraction:
- `Agent1Data`: Fields a, b, c
- `Agent2Data`: Fields d, e
- `Agent3Data`: Fields f, g, h
- `PDFData`: PDF content and metadata
- `ConversationState`: Maintains conversation history and collected data

### Dependency Injection

Agents receive their dependencies (LLM instances) via constructor injection, making them testable and flexible:

```python
llm = ChatOpenAI(model="gpt-4o-mini")
agent = Agent1(llm=llm)  # Dependency injection
```

### Conversation History

The `ConversationState` maintains:
- Message history across turns
- Currently active agent
- Collected data from all agents
- Additional context

## Development

### Adding New Agents

1. Create a new agent class inheriting from `BaseAgent`
2. Implement the `process()` method
3. Add the agent type to `AgentType` enum
4. Register the agent in `multi_agent_graph.py`

### Modifying Routing Logic

Edit the `RouterAgent` class in `agents/router_agent.py` to change how intent is analyzed and routing decisions are made.

## License

MIT

