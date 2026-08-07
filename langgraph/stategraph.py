"""
LangGraph Loan Underwriting StateGraph
Pure deterministic computation graph for loan qualification and risk scoring.
No LLM, no tools — takes structured inputs and returns a structured underwriting decision.
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END


# Typed state schema defining inputs, intermediate calculations, and final underwriting decisions
class UnderwritingState(TypedDict):
    # Customer financial profile inputs
    monthly_income: float
    monthly_debt: float
    credit_score: int
    loan_amount: float
    property_value: float

    # Financial ratio calculations
    dti_ratio: Optional[float]
    ltv_ratio: Optional[float]

    # Risk assessment and pricing tiers
    credit_tier: Optional[str]
    base_interest_rate: Optional[float]

    # Final underwriting decision outputs
    status: Optional[str]
    approved_interest_rate: Optional[float]
    estimated_monthly_payment: Optional[float]
    decision_reason: Optional[str]
    summary: Optional[str]


# Computes debt-to-income (DTI) and loan-to-value (LTV) ratios from customer inputs
def calculate_ratios_node(state: UnderwritingState) -> dict:
    income = state["monthly_income"]
    debt = state["monthly_debt"]
    loan = state["loan_amount"]
    prop_val = state["property_value"]

    dti = round((debt / income) * 100, 2)
    ltv = round((loan / prop_val) * 100, 2)

    return {
        "dti_ratio": dti,
        "ltv_ratio": ltv,
    }


# Maps credit score to bank risk tier and baseline interest rate
def evaluate_credit_risk_node(state: UnderwritingState) -> dict:
    score = state["credit_score"]

    if score >= 740:
        tier = "Tier 1 (Prime / Excellent)"
        base_rate = 6.25
    elif score >= 680:
        tier = "Tier 2 (Good / Standard)"
        base_rate = 6.75
    elif score >= 620:
        tier = "Tier 3 (Fair / Near-Prime)"
        base_rate = 7.50
    else:
        tier = "Tier 4 (Subprime / High Risk)"
        base_rate = 9.25

    return {
        "credit_tier": tier,
        "base_interest_rate": base_rate,
    }


# Evaluates bank underwriting policies, risk limits, and computes monthly payments
def underwrite_decision_node(state: UnderwritingState) -> dict:
    dti = state["dti_ratio"]
    ltv = state["ltv_ratio"]
    score = state["credit_score"]
    base_rate = state["base_interest_rate"]
    loan = state["loan_amount"]

    # Decline if credit score is below minimum cutoff or DTI exceeds regulatory limits
    if score < 620:
        status = "DECLINED"
        reason = f"Credit score ({score}) is below the bank's minimum qualification cutoff of 620."
        final_rate = None
        emi = None
    elif dti > 45.0:
        status = "DECLINED"
        reason = f"Debt-to-Income ratio ({dti}%) exceeds the maximum regulatory threshold of 45%."
        final_rate = None
        emi = None

    # Conditional approval if LTV requires insurance or DTI is in elevated range
    elif ltv > 80.0 or dti > 38.0:
        status = "CONDITIONAL_APPROVAL"
        final_rate = base_rate + 0.25
        reason = (
            f"Conditional approval granted. LTV is {ltv}% (requires Private Mortgage Insurance) "
            f"and DTI is {dti}%."
        )
        # Standard 30-year fixed loan amortization formula
        r = (final_rate / 100) / 12
        n = 30 * 12
        emi = round((loan * r * ((1 + r) ** n)) / (((1 + r) ** n) - 1), 2)

    # Full approval for prime credit and healthy financial ratios
    else:
        status = "APPROVED"
        final_rate = base_rate
        reason = f"Fully qualified with strong credit ({score}), healthy DTI ({dti}%), and solid LTV ({ltv}%)."
        # Standard 30-year fixed loan amortization formula
        r = (final_rate / 100) / 12
        n = 30 * 12
        emi = round((loan * r * ((1 + r) ** n)) / (((1 + r) ** n) - 1), 2)

    # Format human-readable summary for voice synthesizer and logging
    summary_text = (
        f"Loan Underwriting Result: {status}\n"
        f"- Credit Tier: {state['credit_tier']}\n"
        f"- Debt-to-Income (DTI): {dti}%\n"
        f"- Loan-to-Value (LTV): {ltv}%\n"
        f"- Approved Rate: {f'{final_rate}%' if final_rate else 'N/A'}\n"
        f"- Monthly Payment (30-yr): {f'${emi:,.2f}' if emi else 'N/A'}\n"
        f"- Underwriting Notes: {reason}"
    )

    return {
        "status": status,
        "approved_interest_rate": final_rate,
        "estimated_monthly_payment": emi,
        "decision_reason": reason,
        "summary": summary_text,
    }


# Assembles and compiles the StateGraph workflow
def build_underwriting_graph():
    workflow = StateGraph(UnderwritingState)

    # Add workflow nodes
    workflow.add_node("calculate_ratios", calculate_ratios_node)
    workflow.add_node("evaluate_credit_risk", evaluate_credit_risk_node)
    workflow.add_node("underwrite_decision", underwrite_decision_node)

    # Define sequential graph execution flow
    workflow.add_edge(START, "calculate_ratios")
    workflow.add_edge("calculate_ratios", "evaluate_credit_risk")
    workflow.add_edge("evaluate_credit_risk", "underwrite_decision")
    workflow.add_edge("underwrite_decision", END)

    return workflow.compile()


# Global compiled graph instance — import this in livekit_agent.py
underwriting_graph = build_underwriting_graph()


if __name__ == "__main__":
    # Standalone test to verify graph execution without LiveKit or LangChain
    result = underwriting_graph.invoke({
        "monthly_income": 9000,
        "monthly_debt": 1200,
        "credit_score": 745,
        "loan_amount": 300000,
        "property_value": 400000,
        "dti_ratio": None,
        "ltv_ratio": None,
        "credit_tier": None,
        "base_interest_rate": None,
        "status": None,
        "approved_interest_rate": None,
        "estimated_monthly_payment": None,
        "decision_reason": None,
        "summary": None,
    })
    print(result["summary"])
