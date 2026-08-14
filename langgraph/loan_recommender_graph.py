"""
LangGraph Loan Recommender (React Agent)
=========================================
A conversational loan-type recommender built with create_react_agent.

This graph uses ChatOpenAI as its internal LLM and exposes the standard
LangChain message interface ({"messages": [...]}) so it works directly
with livekit-plugins-langchain's LLMAdapter.

The system prompt instructs the LLM to follow a decision-tree flow:
  1. Ask whether the user wants to purchase or renovate
  2. If purchase, ask for down payment percentage
"""
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig

class RecommenderState(TypedDict):
    """State for the Recommender LangGraph"""
    messages: Annotated[list, add_messages]
    loan_purpose: Optional[Literal["purchase", "renovate"]]
    down_payment_percent: Optional[float]
    recommendation_given: bool
    wants_to_exit: bool

class Classification(BaseModel):
    loan_purpose: Optional[Literal["purchase", "renovate"]] = Field(None, description="Whether the user wants to purchase or renovate. Only extract if explicitly stated.")
    down_payment_percent: Optional[float] = Field(None, description="The down payment percentage if mentioned. For example, if they say '20%', put 20.0.")
    wants_to_exit: bool = Field(False, description="True if the user says they have no more questions, or want to exit, stop, or go back.")

# Global model instances for the nodes
_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
_structured_model = _model.with_structured_output(Classification)

# ==========================================
# GRAPH NODES
# ==========================================

# Node 1: Classify Input
# Extracts values from user input and updates the state.
# NOTE: We pass `config: RunnableConfig` into the node and down into the LLM 
# (`_structured_model.ainvoke(..., config)`). This config contains the streaming "pipes".
# It wires the LLM's internal token generator directly to LangGraph's event stream, 
# allowing LiveKit to stream the audio instantly instead of waiting for the full sentence.
async def classify_input(state: RecommenderState, config: RunnableConfig):
    """Silently analyze the user's latest message and extract context variables."""
    messages = state.get("messages", [])
    print(f"DEBUG MESSAGES: {messages}", flush=True)
    if not messages:
        return {}
        
    # Process the entire message history statelessly
    system_prompt = SystemMessage(content="Analyze the conversation history. Extract the user's loan intent (purchase or renovate) and down payment percentage. 'Buy a home' = purchase. IF THE USER DOES NOT EXPLICITLY STATE THEIR INTENT OR DOWN PAYMENT, YOU MUST RETURN NULL/NONE FOR THOSE FIELDS. DO NOT GUESS OR ASSUME.")
    history = [system_prompt] + messages
    classification = await _structured_model.ainvoke(history, config={"callbacks": []})
    return {
        "loan_purpose": classification.loan_purpose if classification.loan_purpose else state.get("loan_purpose"),
        "down_payment_percent": classification.down_payment_percent if classification.down_payment_percent is not None else state.get("down_payment_percent"),
        "wants_to_exit": classification.wants_to_exit
    }

# Node 2: Ask Purpose
# Prompts the user if we don't know whether they want to purchase or renovate.
async def ask_purpose(state: RecommenderState, config: RunnableConfig):
    """Ask the user if they are looking to purchase or renovate."""
    prompt = "Politely ask the user if they are looking to purchase a new home, or renovate an existing one. Keep it to 1 sentence."
    response = await _model.ainvoke([SystemMessage(content=prompt)] + state.get("messages", []), config)
    return {"messages": [response]}

# Node 3: Ask Down Payment
# Prompts the user for their down payment percentage (if purchasing).
async def ask_down_payment(state: RecommenderState, config: RunnableConfig):
    """Ask the user for their down payment percentage."""
    prompt = "Ask the user: 'What percentage down payment are you planning to put down?' Keep it to 1 sentence."
    response = await _model.ainvoke([SystemMessage(content=prompt)] + state.get("messages", []), config)
    return {"messages": [response]}

# Node 4: Recommend Loan / Answer Questions
# Evaluates the gathered state and generates the final loan recommendation, or answers follow-up questions.
async def recommend_loan(state: RecommenderState, config: RunnableConfig):
    """Evaluate gathered state and generate the final loan recommendation or answer questions."""
    purpose = state.get("loan_purpose")
    down_payment = state.get("down_payment_percent")
    
    prompt = "You are a loan recommender. Give a recommendation based on the user's data, OR answer their follow-up questions if a recommendation was already given.\n"
    if purpose == "renovate":
        prompt += "Recommend a HELOC. Explain that a HELOC lets them borrow against their home's existing equity with flexible draw periods, making it ideal for renovation projects where costs come in stages.\n"
    else:
        if down_payment is not None and down_payment < 20:
            prompt += "Recommend an FHA Loan. Explain that FHA loans allow down payments as low as 3.5% and are backed by the Federal Housing Administration, but they require mortgage insurance premiums.\n"
        else:
            prompt += "Recommend a Conventional Mortgage. Explain that with 20%+ down they avoid private mortgage insurance and typically get the best rates.\n"
    
    prompt += "Keep responses concise (2-3 sentences max). Do NOT use complex formatting, bullet points, asterisks, or emojis. At the end, ask if they have any other questions."
    
    response = await _model.ainvoke([SystemMessage(content=prompt)] + state.get("messages", []), config)
    return {"messages": [response], "recommendation_given": True}

# ==========================================
# GRAPH ROUTING
# ==========================================

# Routing Edge: Determine Next Step
def route_next_step(state: RecommenderState):
    """Conditional routing edge to determine the next graph step based on state."""
    if state.get("wants_to_exit"):
        # The LiveKit AgentTask monitors the LangGraph stream for this state update
        # and safely shuts itself down when it detects wants_to_exit = True.
        return END
        
    purpose = state.get("loan_purpose")
    down_payment = state.get("down_payment_percent")
    print(f"DEBUG GRAPH STATE: purpose={purpose}, down_payment={down_payment}", flush=True)

    if not purpose:
        return "ask_purpose"
    
    if purpose == "purchase" and down_payment is None:
        return "ask_down_payment"
        
    # Recommendation phase (handles both the initial recommendation and follow-up questions)
    return "recommend_loan"


workflow = StateGraph(RecommenderState)

# 1. Add nodes
workflow.add_node("classify_input", classify_input)
workflow.add_node("ask_purpose", ask_purpose)
workflow.add_node("ask_down_payment", ask_down_payment)
workflow.add_node("recommend_loan", recommend_loan)

# 2. Add edges
workflow.add_edge(START, "classify_input")
workflow.add_conditional_edges("classify_input", route_next_step)

workflow.add_edge("ask_purpose", END)
workflow.add_edge("ask_down_payment", END)
workflow.add_edge("recommend_loan", END)

# Export the globally compiled graph
graph = workflow.compile()
