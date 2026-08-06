"""
MCP Banking Server
Provides tools and resources for official bank policies, customer profiles, and compliance guidelines via Model Context Protocol (MCP).
"""

import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("bank-policy-server")

# Paths to data files
DATA_DIR = Path(__file__).parent
DB_PATH = DATA_DIR / "db.json"
GUIDELINES_PATH = DATA_DIR / "bank_guidelines.txt"


def _load_db() -> dict:
    """Loads current data from db.json."""
    if not DB_PATH.exists():
        return {"policies": {}, "customers": {}}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# MCP RESOURCES
# ==============================================================================

@mcp.resource("bank://guidelines")
def get_lending_guidelines_resource() -> str:
    """Exposes official lending guidelines, compliance disclosures, and required borrower documentation as an MCP resource."""
    if not GUIDELINES_PATH.exists():
        return "Guidelines document not found."
    with open(GUIDELINES_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ==============================================================================
# MCP TOOLS
# ==============================================================================

@mcp.tool()
def read_lending_guidelines() -> str:
    """
    Reads the official bank lending compliance guidelines, required documentation, and disclosure rules.
    Use this when the customer asks about requirements, documentation, closing costs, or compliance rules.
    """
    return get_lending_guidelines_resource()


@mcp.tool()
def fetch_bank_policy(customer_type: str = "retail", loan_type: str = "home") -> str:
    """
    Fetches the official bank policy, interest rates, and loan terms for a given customer and loan type.
    
    Args:
        customer_type: Type of customer, e.g. 'retail' or 'corporate'.
        loan_type: Type of loan, e.g. 'home', 'auto', 'personal', 'commercial_real_estate', 'equipment'.
    """
    db = _load_db()
    policies = db.get("policies", {})

    cust_type = customer_type.strip().lower()
    l_type = loan_type.strip().lower()

    if cust_type not in policies:
        return f"Unknown customer type '{customer_type}'. Available customer types: {list(policies.keys())}."

    customer_policies = policies[cust_type]
    if l_type not in customer_policies:
        return (
            f"Loan type '{loan_type}' not found for {customer_type}. "
            f"Available loan types for {customer_type}: {list(customer_policies.keys())}."
        )

    policy = customer_policies[l_type]
    return (
        f"Bank Policy for {customer_type.capitalize()} {loan_type.capitalize()} Loan:\n"
        f"- Standard Interest Rate: {policy['interest_rate']}%\n"
        f"- Maximum Loan Amount: ${policy['max_loan_amount']:,}\n"
        f"- Minimum Down Payment: {policy['min_down_payment_pct']}%\n"
        f"- Maximum Tenure: {policy['max_tenure_years']} years\n"
        f"- Policy Details: {policy['description']}"
    )


@mcp.tool()
def get_customer_profile(account_id: str) -> str:
    """
    Retrieves the customer's account profile, credit tier, and pre-approved limits.
    
    Args:
        account_id: Customer's account ID (e.g. 'ACC-101', 'ACC-202').
    """
    db = _load_db()
    customers = db.get("customers", {})

    acc_id = account_id.strip().upper()
    if acc_id not in customers:
        return f"Customer account '{account_id}' not found. Available test accounts: {list(customers.keys())}."

    cust = customers[acc_id]
    return (
        f"Customer Account Details for {acc_id}:\n"
        f"- Customer Name: {cust['name']}\n"
        f"- Account Type: {cust['customer_type'].capitalize()}\n"
        f"- Credit Score: {cust['credit_score']}\n"
        f"- Pre-Approved Limit: ${cust['pre_approved_limit']:,}\n"
        f"- Status: {cust['account_status']}"
    )


@mcp.tool()
def list_available_loan_products(customer_type: str = "all") -> str:
    """
    Lists all available loan types and products offered by the bank for retail, corporate, or both.
    Use this when the customer asks what loan options, products, or types of financing the bank offers.

    Args:
        customer_type: Filter by 'retail', 'corporate', or 'all'.
    """
    db = _load_db()
    policies = db.get("policies", {})
    c_type = customer_type.strip().lower()

    if c_type != "all" and c_type in policies:
        target_groups = {c_type: policies[c_type]}
    else:
        target_groups = policies

    output_lines = ["Available Bank Loan Products:"]
    for grp_name, loans in target_groups.items():
        output_lines.append(f"\n[{grp_name.upper()} LOANS]")
        for loan_key, details in loans.items():
            output_lines.append(
                f"- {loan_key.replace('_', ' ').capitalize()} Loan: "
                f"Rate {details['interest_rate']}%, up to ${details['max_loan_amount']:,} "
                f"({details['description']})"
            )

    return "\n".join(output_lines)

if __name__ == "__main__":
    mcp.run(transport="sse")
