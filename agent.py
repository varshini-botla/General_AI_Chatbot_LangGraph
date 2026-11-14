"""
General-Purpose AI Chatbot - LangGraph Implementation
Architecture:
  START -> chatbot -> END
Uses LangChain Groq (Llama / OSS 120B) with LangSmith tracing and LangGraph state management.
"""

import os
from typing import Annotated, Sequence
from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()

# System Prompt defining the General-Purpose AI Assistant persona and boundaries
SYSTEM_PROMPT = """You are a highly capable, thoughtful, and precise General-Purpose AI Assistant, similar to ChatGPT.

Your expertise spans:
1. General Knowledge & History, Geography, Culture, Everyday Questions
2. Science & Technology (Physics, Chemistry, Biology, Astronomy, Engineering)
3. Mathematics, Quantitative Analysis, and Logical Reasoning
4. Software Engineering & Programming (Python, Java, JavaScript, TypeScript, Web Development, SQL, Git)
5. Artificial Intelligence, Machine Learning, Deep Learning, and Agentic AI (LangChain, LangGraph)
6. Education, Academic Subjects, and Concept Explanation (ELI5 to advanced technical levels)
7. Writing, Editing, Brainstorming, and Content Creation
8. Career Guidance, Interview Preparation, and Problem Solving

Core Behavioral Rules:
- Communicate clearly, concisely, and helpfully with structured formatting (bullet points, markdown headers, and code blocks where helpful).
- When writing code, provide clean, idiomatic, fully explained, and bug-free code examples.
- Remember previous context within the conversation and maintain seamless multi-turn dialogue.
- Honestly state your boundaries and limitations. If a question cannot be answered with certainty, admit it rather than inventing facts (zero hallucination).
- Do not claim to browse the live internet unless explicitly provided with a search tool.
"""

class ChatState(TypedDict):
    """
    Message state for the Chatbot graph.
    Uses LangGraph's add_messages reducer to properly append conversation turns.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


def get_llm():
    """
    Initializes and returns a configured Groq Chat LLM instance from environment variables.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment.")
    
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    temperature = float(os.getenv("GROQ_TEMPERATURE", "0.7"))
    
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=temperature,
    )


def chatbot_node(state: ChatState) -> dict:
    """
    Chatbot execution node.
    Extracts conversation messages, ensures system instructions are present,
    invokes Groq LLM, and returns the AI response.
    """
    messages = list(state.get("messages", []))
    
    # Prepend system prompt if not already present at the head of conversation
    conversation = []
    if not messages or not isinstance(messages[0], SystemMessage):
        conversation.append(SystemMessage(content=SYSTEM_PROMPT))
    conversation.extend(messages)
    
    llm = get_llm()
    try:
        response = llm.invoke(conversation)
        return {"messages": [response]}
    except Exception as e:
        error_type = type(e).__name__
        err_str = str(e)
        if "AuthenticationError" in error_type or "401" in err_str:
            fallback = "I encountered an authentication error connecting to Groq. Please verify that GROQ_API_KEY in your configuration is valid."
        elif "RateLimitError" in error_type or "429" in err_str:
            fallback = "The Groq API rate limit has been reached. Please wait a moment and try again."
        elif "NotFoundError" in error_type or "model_not_found" in err_str:
            fallback = f"The configured Groq model was not found. Please verify GROQ_MODEL in your configuration."
        else:
            fallback = f"An unexpected error occurred while generating a response: {err_str}"
        return {"messages": [AIMessage(content=fallback)]}


def create_graph(checkpointer=None):
    """
    Constructs and compiles the StateGraph workflow.
    """
    workflow = StateGraph(ChatState)
    
    # Add Nodes
    workflow.add_node("chatbot", chatbot_node)
    
    # Define Edges: START -> chatbot -> END
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)
    
    # Compile graph with optional checkpointer (used for local standalone execution)
    return workflow.compile(checkpointer=checkpointer)


# Exported compiled graph for LangGraph CLI / LangGraph Server / Studio
graph = create_graph()

if __name__ == "__main__":
    import uuid
    from langgraph.checkpoint.sqlite import SqliteSaver

    print("\n--- Testing General AI Chatbot Graph Standalone ---")
    db_path = os.path.join(os.path.dirname(__file__), "data", "local_checkpoints.sqlite")
    
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        standalone_graph = create_graph(checkpointer=checkpointer)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        test_msg = "Hello! In two sentences, what are the core branches of modern artificial intelligence?"
        print(f"User: {test_msg}\n")
        
        result = standalone_graph.invoke(
            {"messages": [HumanMessage(content=test_msg)]},
            config=config,
        )
        ai_resp = result["messages"][-1].content
        print(f"Assistant: {ai_resp}\n")
        print(f"[Verified] Standalone execution and SQLite checkpointing to: {db_path}")
