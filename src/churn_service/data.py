from __future__ import annotations

import numpy as np
import pandas as pd


def make_dataset(rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic demo dataset with a realistic churn signal."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, rows)
    monthly_charges = rng.normal(75, 25, rows).clip(20, 180)
    support_tickets = rng.poisson(2, rows)
    contract_months = rng.choice([1, 12, 24], rows, p=[0.55, 0.3, 0.15])
    logit = (
        1.6
        - 0.035 * tenure
        + 0.018 * monthly_charges
        + 0.22 * support_tickets
        - 0.055 * contract_months
    )
    churn = (rng.random(rows) < (1 / (1 + np.exp(-logit)))).astype(int)
    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly_charges.round(2),
            "support_tickets": support_tickets,
            "contract_months": contract_months,
            "churn": churn,
        }
    )
