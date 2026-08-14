"""
Session State
=============
Single source of truth for all per-caller session data.

This dataclass is passed as `userdata` to AgentSession in main.py and
is injected into every @llm.function_tool via RunContext.userdata.

Each field is a domain-specific state slice. Tools read/write only
their own slice via context.userdata.<domain>.
"""

from dataclasses import dataclass, field
from typing import Optional

from langgraph.stategraph import UnderwritingState

#we could add more state classes here if needed
@dataclass
class LoanState:
    """Per-caller underwriting cache so revisions skip the interview."""
    last_underwriting_state: Optional[UnderwritingState] = None


@dataclass
class SessionState:
    """Holds per-caller state across all tool calls within a single session."""

    # default_factory creates a new LoanState for each new session
    # unlike loan = LoanState() which would create only one object for all sessions
    loan: LoanState = field(default_factory=LoanState)
