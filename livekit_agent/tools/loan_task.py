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
from livekit.agents import AgentTask, RunContext, llm
from livekit.agents.beta.workflows import TaskGroup

from langgraph.stategraph import underwriting_graph, UnderwritingState


# =====================================================================
# 1. Result Dataclasses for TaskGroup Stages
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
# 2. Stage 1 Task: Loan Details & Collateral Intake
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
# 3. Stage 2 Task: Financial Assessment & Borrower Capacity
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
# 4. Main Tool: Orchestrates TaskGroup & Invokes LangGraph StateGraph
# =====================================================================

@llm.function_tool
async def evaluate_loan_underwriting() -> str:
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
