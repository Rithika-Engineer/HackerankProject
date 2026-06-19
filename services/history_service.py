"""
User history service.
"""

import pandas as pd

class UserHistoryService:
    def __init__(self, history_df: pd.DataFrame):
        self.history_df = history_df
        # Index by user_id for fast lookup
        self.history_df.set_index("user_id", inplace=True)
        
    def get_user_history(self, user_id: str) -> dict:
        """Return user risk profile dict."""
        if user_id in self.history_df.index:
            return self.history_df.loc[user_id].to_dict()
        return {
            "past_claim_count": 0,
            "accept_claim": 0,
            "manual_review_claim": 0,
            "rejected_claim": 0,
            "last_90_days_claim_count": 0,
            "history_flags": "none",
            "history_summary": "New user with no prior claim history"
        }
        
    def get_risk_flags(self, user_id: str) -> list[str]:
        """Return history-based risk flags."""
        history = self.get_user_history(user_id)
        flags_str = history.get("history_flags", "none")
        if pd.isna(flags_str) or flags_str == "none":
            return []
        return [f.strip() for f in flags_str.split(";") if f.strip()]
        
    def is_high_risk(self, user_id: str) -> bool:
        """Check if user has user_history_risk flag."""
        return "user_history_risk" in self.get_risk_flags(user_id)
