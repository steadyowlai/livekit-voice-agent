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
    ltv = state["ltv_ratio"]
    score = state["credit_score"]
    base_rate = state["base_interest_rate"]
    loan = state["loan_amount"]
    income = state["monthly_income"]
    debt = state["monthly_debt"]

    # 1. Compute Loan-to-Income (LTI)
    annual_income = income * 12
    lti = round(loan / annual_income, 2) if annual_income > 0 else float('inf')

    # 2. Determine pricing and calculate projected monthly payment
    # Surcharge if LTV > 80% (PMI requirement)
    final_rate = base_rate + 0.25 if ltv > 80.0 else base_rate
    
    r = (final_rate / 100) / 12
    n = 30 * 12
    emi = round((loan * r * ((1 + r) ** n)) / (((1 + r) ** n) - 1), 2)
    
    # 3. Calculate true back-end DTI (including the new mortgage payment)
    total_dti = round(((debt + emi) / income) * 100, 2) if income > 0 else float('inf')

    # 4. Evaluate hard limits and rules
    if score < 620:
        status = "DECLINED"
        reason = f"Credit score ({score}) is below the bank's minimum qualification cutoff of 620."
        final_rate = None
        emi = None
    elif ltv > 95.0:
        status = "DECLINED"
        reason = f"Loan-to-Value ratio ({ltv}%) exceeds the absolute maximum of 95%."
        final_rate = None
        emi = None
    elif lti > 5.0:
        status = "DECLINED"
        reason = f"Loan amount exceeds 5x annual income limit (LTI is {lti}x)."
        final_rate = None
        emi = None
    elif total_dti > 45.0:
        status = "DECLINED"
        reason = f"Total Debt-to-Income ratio ({total_dti}%) including the new mortgage exceeds the 45% maximum."
        final_rate = None
        emi = None
    elif ltv > 80.0 or total_dti > 38.0:
        status = "CONDITIONAL_APPROVAL"
        reason = (
            f"Conditional approval granted. LTV is {ltv}% (requires Private Mortgage Insurance) "
            f"and total DTI is {total_dti}%."
        )
    else:
        status = "APPROVED"
        reason = f"Fully qualified with strong credit ({score}), healthy total DTI ({total_dti}%), and solid LTV ({ltv}%)."

    # Format human-readable summary for voice synthesizer and logging
    summary_text = (
        f"Loan Underwriting Result: {status}\n"
        f"- Credit Tier: {state['credit_tier']}\n"
        f"- Total DTI (Back-End): {total_dti}%\n"
        f"- Loan-to-Value (LTV): {ltv}%\n"
        f"- Loan-to-Income (LTI): {lti}x\n"
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
