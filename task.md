

Skip to content
Using Gmail with screen readers
Conversations
4% of 2,048 GB used
Cancel subscription
Terms · Privacy · Program Policies
Last account activity: 0 minutes ago
Open in 1 other location · Details
# 🤖 Multi-Agent System Programming Exercise

## Overview

In this exercise, you'll build a **multi-agent data collection system**. The system uses a router agent to intelligently direct users to specialized data collection agents.

---

## 🎯 Goal

1. Create AI agents
2. Define structured output types with ideally models
3. Implement a router agent for intent-based routing / one of the agent should load in pdf
4. Manage conversation history across multiple turns
5. Extract structured data from conversations
6. Use dependency injection with agents

---

## 📋 System Architecture

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
     │ Agent 1 │   │ Agent 2 │   │ Agent 3 │   │  Final  │
     │ a, b, c │   │  d, e   │   │ f, g, h │   │ Summary │
     └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

---

### Environment Setup

Set your OpenAI API key:

Good luck! 🚀
TASK.md
Displaying TASK.md.