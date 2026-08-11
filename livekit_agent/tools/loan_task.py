"""
LiveKit TaskGroup Workflow & Underwriting Integration Tool
=========================================================
Follows the official LiveKit TaskGroup pattern:
  - Each task collects specific data slots and completes with a typed Result dataclass.
  - Tasks are isolated from MCP servers (mcp_servers=[]).
  - Main tool runs the sequential TaskGroup, invokes LangGraph StateGraph, and returns
    the underwriting evaluation result to the Assistant for a clean single announcement.
"""

from dataclasses import dataclass
from typing import Optional, cast
from livekit.agents import AgentTask, RunContext, llm
from livekit.agents.beta.workflows import TaskGroup

from langgraph.stategraph import underwriting_graph, UnderwritingState
from livekit_agent.session_state import SessionState


# =====================================================================
# Result Dataclasses for TaskGroup Stages
# =====================================================================

@dataclass
class LoanRequestResult:
    """Result payload from Stage 1 (Loan & Collateral Intake)."""
    loan_amount: float
    property_value: float


@dataclass
class FinancialProfileResult:
    """Result payload from Stage 2 (Financial Capacity Assessment)."""
    monthly_income: float
    monthly_debt: float
    credit_score: int


# =====================================================================
# Stage 1 Task: Loan Details & Collateral Intake
# =====================================================================

class LoanRequestTask(AgentTask[LoanRequestResult]):
    """
    Stage 1 AgentTask in the TaskGroup.
    Collects the borrower's target loan amount and estimated property value.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Collect the caller's requested loan amount and estimated property value. "
                "When the caller provides both numbers, call record_loan_request immediately without asking follow-up questions."
            ),
        )

    async def on_enter(self) -> None:
        """Proactively ask for loan details upon entering Stage 1."""
        await self.session.generate_reply(
            instructions="Ask the caller how much they would like to borrow and the estimated property value."
        )

    @llm.function_tool
    async def record_loan_request(
        self, loan_amount: float, property_value: float
    ) -> None:
        """Record the requested loan amount and estimated property value in dollars."""
        self.complete(
            LoanRequestResult(
                loan_amount=loan_amount, property_value=property_value
            )
        )


# =====================================================================
# Stage 2 Task: Financial Assessment & Borrower Capacity
# =====================================================================

class FinancialProfileTask(AgentTask[FinancialProfileResult]):
    """
    Stage 2 AgentTask in the TaskGroup.
    Collects gross monthly household income, monthly debt payments, and credit score.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Collect the caller's gross monthly household income, total existing monthly debt payments (excluding rent), and credit score. "
                "When the caller provides these numbers, call record_financial_profile immediately without asking follow-up questions."
            ),
        )

    async def on_enter(self) -> None:
        """Proactively ask for financial details upon entering Stage 2."""
        await self.session.generate_reply(
            instructions="Ask the caller for their gross monthly household income, existing monthly debt payments (excluding rent), and estimated credit score."
        )

    @llm.function_tool
    async def record_financial_profile(
        self, monthly_income: float, monthly_debt: float, credit_score: int
    ) -> None:
        """Record the caller's gross monthly income, monthly debt payments, and estimated credit score."""
        self.complete(
            FinancialProfileResult(
                monthly_income=monthly_income,
                monthly_debt=monthly_debt,
                credit_score=credit_score,
            )
        )


# =====================================================================
# Main Tool: Orchestrates TaskGroup & Invokes LangGraph StateGraph
# =====================================================================

@llm.function_tool
async def evaluate_loan_underwriting(context: RunContext["SessionState"]) -> str:
    """Start the structured 2-step loan pre-qualification interview to assess eligibility and compute an underwriting decision."""
    # Build the 2-stage TaskGroup workflow
    group = (
        TaskGroup()
        .add(
            lambda: LoanRequestTask(),
            id="loan",
            description="Collect requested loan amount and property value",
        )
        .add(
            lambda: FinancialProfileTask(),
            id="financials",
            description="Collect monthly income, monthly debt, and credit score",
        )
    )

    # Await sequential TaskGroup execution
    results = await group

    # Extract typed results from each task
    loan_res: LoanRequestResult = results.task_results["loan"]
    fin_res: FinancialProfileResult = results.task_results["financials"]

    # Construct the initial state for the pure LangGraph StateGraph engine
    initial_state: UnderwritingState = {
        "monthly_income": fin_res.monthly_income,
        "monthly_debt": fin_res.monthly_debt,
        "credit_score": fin_res.credit_score,
        "loan_amount": loan_res.loan_amount,
        "property_value": loan_res.property_value,
        "dti_ratio": None,
        "ltv_ratio": None,
        "credit_tier": None,
        "base_interest_rate": None,
        "status": None,
        "approved_interest_rate": None,
        "estimated_monthly_payment": None,
        "decision_reason": None,
        "summary": None,
    }

    # Cache the inputs into the per-session userdata so revise_loan_underwriting can patch them later
    context.userdata.loan.last_underwriting_state = initial_state.copy()

    # Invoke the pure LangGraph StateGraph asynchronously
    result = await underwriting_graph.ainvoke(initial_state)

    # Safely format fields to handle DENIED or unapproved loan outcomes where values may be None
    rate_str = (
        f"{result['approved_interest_rate']}%"
        if result.get("approved_interest_rate") is not None
        else "N/A"
    )
    payment_str = (
        f"${result['estimated_monthly_payment']:,.2f}"
        if result.get("estimated_monthly_payment") is not None
        else "N/A"
    )

    # Return structured decision result directly to the Assistant
    return (
        f"Loan Underwriting Result:\n"
        f"- Status: {result.get('status')}\n"
        f"- Credit Tier: {result.get('credit_tier')}\n"
        f"- Approved Interest Rate: {rate_str}\n"
        f"- Requested Loan Amount: ${result.get('loan_amount', 0):,.2f}\n"
        f"- Estimated Monthly Payment: {payment_str}\n"
        f"- Decision Reason: {result.get('decision_reason')}"
    )


# =====================================================================
# Revision Tool: Re-runs the graph with a patched field, no re-interview
# =====================================================================

@llm.function_tool
async def revise_loan_underwriting(
    context: RunContext["SessionState"],
    loan_amount: Optional[float] = None,
    property_value: Optional[float] = None,
    monthly_income: Optional[float] = None,
    monthly_debt: Optional[float] = None,
    credit_score: Optional[int] = None,
) -> str:
    """
    Revise one or more fields from the previous underwriting assessment and re-run the decision.
    Only call this after evaluate_loan_underwriting has already been completed in this session.
    Pass only the fields the caller wants to correct; all other values are carried over from the previous run.
    """
    if context.userdata.loan.last_underwriting_state is None:
        return "No previous underwriting assessment found. Please run a full evaluation first."

    # We start by carrying over everything from the previous assessment. 
    # This acts as our baseline so the user doesn't have to repeat information 
    # for the fields they aren't changing.
    revised_state: UnderwritingState = {
        **context.userdata.loan.last_underwriting_state,
    }

    # Because arguments default to None, only the values explicitly passed by the user will be not None.
    # We only update the revised_state for those fields. The rest will remain as their old values.
    #
    # Example:
    #   Old state: loan_amount=300_000, credit_score=720
    #   User passes: revise_loan_underwriting(credit_score=750)
    #   
    #   - credit_score is not None (750), so we update revised_state["credit_score"]
    #   - loan_amount is not passed, so it's None, thus revised_state["loan_amount"] is not updated and remains as old value.
    if loan_amount is not None:
        revised_state["loan_amount"] = loan_amount
    if property_value is not None:
        revised_state["property_value"] = property_value
    if monthly_income is not None:
        revised_state["monthly_income"] = monthly_income
    if monthly_debt is not None:
        revised_state["monthly_debt"] = monthly_debt
    if credit_score is not None:
        revised_state["credit_score"] = credit_score

    # Update the per-session cache so further revisions in this call still work
    context.userdata.loan.last_underwriting_state = revised_state.copy()

    result = await underwriting_graph.ainvoke(revised_state)

    rate_str = (
        f"{result['approved_interest_rate']}%"
        if result.get("approved_interest_rate") is not None
        else "N/A"
    )
    payment_str = (
        f"${result['estimated_monthly_payment']:,.2f}"
        if result.get("estimated_monthly_payment") is not None
        else "N/A"
    )

    return (
        f"Revised Loan Underwriting Result:\n"
        f"- Status: {result.get('status')}\n"
        f"- Credit Tier: {result.get('credit_tier')}\n"
        f"- Approved Interest Rate: {rate_str}\n"
        f"- Requested Loan Amount: ${result.get('loan_amount', 0):,.2f}\n"
        f"- Estimated Monthly Payment: {payment_str}\n"
        f"- Decision Reason: {result.get('decision_reason')}"
    )
