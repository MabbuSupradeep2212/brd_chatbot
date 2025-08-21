import os
from typing import Dict, List, Annotated, TypedDict, Union
from typing_extensions import TypedDict
import ollama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_core.tools import tool
import logging
import json
import re
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")

# Agent state type definitions
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    current_brd: str
    brd_id: str
    memories: List[Dict]
    next_step: str
    artifacts: List[Dict]
    errors: List[str]

# Tool definitions
@tool("extract_requirements")
def extract_requirements(brd_text: str) -> List[str]:
    """Extract requirements from BRD text."""
    try:
        system_prompt = (
            "You are a BRD analyst specialized in requirements extraction. "
            "Extract key requirements from the BRD text, focusing on functional "
            "and non-functional requirements. Return a JSON array of requirement strings."
        )
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"BRD text:\n\n{brd_text}\n\nExtract requirements as JSON array."}
            ],
            options={"temperature": 0.2}
        )
        content = resp.get("message", {}).get("content", "[]").strip()
        if "{" in content and "}" in content:
            start = content.find("[")
            end = content.rfind("]") + 1
            content = content[start:end]
        reqs = json.loads(content)
        return [str(r) for r in reqs if r]
    except Exception as e:
        logger.error(f"Requirements extraction failed: {str(e)}")
        return []

@tool("extract_stakeholders")
def extract_stakeholders(brd_text: str) -> List[Dict]:
    """Extract stakeholders and their roles from BRD text."""
    try:
        system_prompt = (
            "You are a BRD analyst focused on stakeholder analysis. "
            "Extract stakeholders and their roles from the BRD text. "
            "Return a JSON array of objects with 'name' and 'role' fields."
        )
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"BRD text:\n\n{brd_text}\n\nExtract stakeholders as JSON array."}
            ],
            options={"temperature": 0.2}
        )
        content = resp.get("message", {}).get("content", "[]").strip()
        if "{" in content and "}" in content:
            start = content.find("[")
            end = content.rfind("]") + 1
            content = content[start:end]
        stakeholders = json.loads(content)
        return [s for s in stakeholders if isinstance(s, dict) and s.get("name")]
    except Exception as e:
        logger.error(f"Stakeholder extraction failed: {str(e)}")
        return []

@tool("generate_summary")
def generate_summary(brd_text: str) -> str:
    """Generate a concise summary of the BRD."""
    try:
        system_prompt = (
            "You are a BRD analyst specialized in creating executive summaries. "
            "Create a concise summary of the BRD focusing on key objectives, scope, "
            "and major requirements. Be clear and professional."
        )
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"BRD text:\n\n{brd_text}\n\nCreate a concise executive summary."}
            ],
            options={"temperature": 0.3}
        )
        return resp.get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.error(f"Summary generation failed: {str(e)}")
        return ""

@tool("analyze_risks")
def analyze_risks(brd_text: str) -> List[Dict]:
    """Analyze and extract risks from BRD text."""
    try:
        system_prompt = (
            "You are a risk analysis expert for BRDs. Extract and analyze risks "
            "from the BRD text. For each risk, identify its description, impact, "
            "and suggested mitigation. Return as JSON array."
        )
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"BRD text:\n\n{brd_text}\n\nAnalyze risks as JSON array."}
            ],
            options={"temperature": 0.2}
        )
        content = resp.get("message", {}).get("content", "[]").strip()
        if "{" in content and "}" in content:
            start = content.find("[")
            end = content.rfind("]") + 1
            content = content[start:end]
        risks = json.loads(content)
        return [r for r in risks if isinstance(r, dict) and r.get("description")]
    except Exception as e:
        logger.error(f"Risk analysis failed: {str(e)}")
        return []

# Agent nodes
def route_by_intent(state: AgentState) -> str:
    """Route to appropriate node based on user intent."""
    try:
        last_msg = state["messages"][-1].content.lower()
        if any(kw in last_msg for kw in ["requirement", "feature"]):
            return "extract_requirements"
        if any(kw in last_msg for kw in ["stakeholder", "role", "user"]):
            return "extract_stakeholders"
        if any(kw in last_msg for kw in ["summary", "overview"]):
            return "generate_summary"
        if any(kw in last_msg for kw in ["risk", "issue", "concern"]):
            return "analyze_risks"
        return "default_response"
    except Exception:
        return "default_response"

def extract_requirements_node(state: AgentState) -> AgentState:
    """Node to extract and format requirements."""
    try:
        if not state["current_brd"]:
            state["errors"].append("No BRD text available")
            return state
        
        reqs = extract_requirements(state["current_brd"])
        if reqs:
            response = "Here are the key requirements I've identified:\n\n"
            for i, req in enumerate(reqs, 1):
                response += f"{i}. {req}\n"
        else:
            response = "I couldn't identify any clear requirements in the BRD text."
        
        state["messages"].append(AIMessage(content=response))
        state["artifacts"].append({
            "type": "requirements",
            "data": reqs,
            "timestamp": datetime.utcnow().isoformat()
        })
        return state
    except Exception as e:
        state["errors"].append(str(e))
        return state

def extract_stakeholders_node(state: AgentState) -> AgentState:
    """Node to extract and format stakeholder information."""
    try:
        if not state["current_brd"]:
            state["errors"].append("No BRD text available")
            return state
        
        stakeholders = extract_stakeholders(state["current_brd"])
        if stakeholders:
            response = "I've identified the following stakeholders:\n\n"
            for s in stakeholders:
                response += f"• {s['name']}: {s['role']}\n"
        else:
            response = "I couldn't identify any clear stakeholders in the BRD text."
        
        state["messages"].append(AIMessage(content=response))
        state["artifacts"].append({
            "type": "stakeholders",
            "data": stakeholders,
            "timestamp": datetime.utcnow().isoformat()
        })
        return state
    except Exception as e:
        state["errors"].append(str(e))
        return state

def generate_summary_node(state: AgentState) -> AgentState:
    """Node to generate and format BRD summary."""
    try:
        if not state["current_brd"]:
            state["errors"].append("No BRD text available")
            return state
        
        summary = generate_summary(state["current_brd"])
        if summary:
            response = f"Here's a summary of the BRD:\n\n{summary}"
        else:
            response = "I couldn't generate a meaningful summary from the BRD text."
        
        state["messages"].append(AIMessage(content=response))
        state["artifacts"].append({
            "type": "summary",
            "data": summary,
            "timestamp": datetime.utcnow().isoformat()
        })
        return state
    except Exception as e:
        state["errors"].append(str(e))
        return state

def analyze_risks_node(state: AgentState) -> AgentState:
    """Node to analyze and format risks."""
    try:
        if not state["current_brd"]:
            state["errors"].append("No BRD text available")
            return state
        
        risks = analyze_risks(state["current_brd"])
        if risks:
            response = "Here are the key risks I've identified:\n\n"
            for i, risk in enumerate(risks, 1):
                response += f"{i}. Risk: {risk['description']}\n"
                response += f"   Impact: {risk['impact']}\n"
                response += f"   Mitigation: {risk['mitigation']}\n\n"
        else:
            response = "I couldn't identify any clear risks in the BRD text."
        
        state["messages"].append(AIMessage(content=response))
        state["artifacts"].append({
            "type": "risks",
            "data": risks,
            "timestamp": datetime.utcnow().isoformat()
        })
        return state
    except Exception as e:
        state["errors"].append(str(e))
        return state

def default_response_node(state: AgentState) -> AgentState:
    """Node for general BRD analysis and response."""
    try:
        if not state["current_brd"]:
            state["errors"].append("No BRD text available")
            return state
        
        last_msg = state["messages"][-1].content
        system_prompt = (
            "You are an expert BRD analyst. Analyze the BRD text and answer "
            "the user's question professionally and thoroughly."
        )
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"BRD text:\n\n{state['current_brd']}\n\nQuestion: {last_msg}"}
            ],
            options={"temperature": 0.3}
        )
        response = resp.get("message", {}).get("content", "").strip()
        if not response:
            response = "I apologize, but I couldn't generate a meaningful response to your question."
        
        state["messages"].append(AIMessage(content=response))
        return state
    except Exception as e:
        state["errors"].append(str(e))
        return state

# Create the agent graph
def create_brd_agent() -> Graph:
    """Create and configure the BRD analysis agent graph."""
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("route", route_by_intent)
    workflow.add_node("extract_requirements", extract_requirements_node)
    workflow.add_node("extract_stakeholders", extract_stakeholders_node)
    workflow.add_node("generate_summary", generate_summary_node)
    workflow.add_node("analyze_risks", analyze_risks_node)
    workflow.add_node("default_response", default_response_node)
    
    # Add edges
    workflow.set_entry_point("route")
    workflow.add_edge("route", "extract_requirements")
    workflow.add_edge("route", "extract_stakeholders")
    workflow.add_edge("route", "generate_summary")
    workflow.add_edge("route", "analyze_risks")
    workflow.add_edge("route", "default_response")
    
    # Add conditional edges back to route
    workflow.add_conditional_edges(
        "extract_requirements",
        lambda x: "route" if x["messages"][-1].content else "end"
    )
    workflow.add_conditional_edges(
        "extract_stakeholders",
        lambda x: "route" if x["messages"][-1].content else "end"
    )
    workflow.add_conditional_edges(
        "generate_summary",
        lambda x: "route" if x["messages"][-1].content else "end"
    )
    workflow.add_conditional_edges(
        "analyze_risks",
        lambda x: "route" if x["messages"][-1].content else "end"
    )
    workflow.add_conditional_edges(
        "default_response",
        lambda x: "route" if x["messages"][-1].content else "end"
    )
    
    # Compile
    return workflow.compile()

# Initialize agent
brd_agent = create_brd_agent()

def process_brd_query(content: str, brd_text: str = None) -> str:
    """Process a BRD query using the agent graph."""
    try:
        # Initialize state
        state = AgentState(
            messages=[HumanMessage(content=content)],
            current_brd=brd_text or "",
            brd_id=str(uuid.uuid4()) if brd_text else None,
            memories=[],
            next_step="route",
            artifacts=[],
            errors=[]
        )
        
        # Run agent
        result = brd_agent.invoke(state)
        
        # Extract response
        if result["messages"] and len(result["messages"]) > 1:
            return result["messages"][-1].content
        return "I apologize, but I couldn't process your request successfully."
        
    except Exception as e:
        logger.error(f"Agent processing failed: {str(e)}")
        return f"Error processing query: {str(e)}"
