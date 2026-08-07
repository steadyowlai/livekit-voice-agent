from livekit.agents import llm


@llm.function_tool
async def calculate_loan_emi(
    principal: float, interest_rate: float, tenure_years: int
) -> str:
    """
    Calculates the exact monthly loan repayment (EMI), total interest, and total cost of the loan.

    Args:
        principal: Total loan amount borrowed in dollars (e.g. 250000).
        interest_rate: Annual interest rate percentage (e.g. 6.5 for 6.5%).
        tenure_years: Loan duration in years (e.g. 15, 20, 30).
    """
    if principal <= 0 or interest_rate <= 0 or tenure_years <= 0:
        return "Please provide positive numbers for principal, interest rate, and tenure."

    monthly_rate = (interest_rate / 100) / 12
    total_months = tenure_years * 12
    emi = (principal * monthly_rate * ((1 + monthly_rate) ** total_months)) / (
        ((1 + monthly_rate) ** total_months) - 1
    )
    total_payment = emi * total_months
    total_interest = total_payment - principal

    return (
        f"Loan Calculation Summary:\n"
        f"- Principal: ${principal:,.2f}\n"
        f"- Interest Rate: {interest_rate}%\n"
        f"- Tenure: {tenure_years} years ({total_months} months)\n"
        f"- Estimated Monthly Payment: ${emi:,.2f}\n"
        f"- Total Interest Paid: ${total_interest:,.2f}\n"
        f"- Total Loan Cost: ${total_payment:,.2f}"
    )
