"""Config data"""
from datetime import datetime

import numpy as np

# Taken from https://www.gov.uk/stamp-duty-land-tax/residential-property-rates
stamp_duty_rates = [
    {
        "date_range": (datetime(2020, 7, 8), datetime(2021, 6, 30)),
        "thresholds": [500_000, 925_000, 1_500_000, np.inf],
        "normal_rate": [0.0, 0.05, 0.1, 0.12],
        "first_time_rate": [0.0, 0.05, 0.1, 0.12],
        "higher_rate": [0.03, 0.08, 0.13, 0.15],
    },
    {
        "date_range": (datetime(2021, 7, 1), datetime(2021, 9, 30)),
        "thresholds": [250_000, 500_000, 925_000, 1_500_000, np.inf],
        "normal_rate": [0.0, 0.05, 0.05, 0.1, 0.12],
        "first_time_rate": [0.0, 0.0, 0.05, 0.1, 0.12],
        "higher_rate": [0.03, 0.08, 0.08, 0.13, 0.15],
    },
    {
        "date_range": (datetime(2021, 10, 1), datetime(3000, 1, 1)),
        "thresholds": [125_000, 250_000, 500_000, 925_000, 1_500_000, np.inf],
        "normal_rate": [0.0, 0.02, 0.05, 0.05, 0.1, 0.12],
        "first_time_rate": [0.0, 0.0, 0.0, 0.05, 0.1, 0.12],
        "higher_rate": [0.03, 0.05, 0.08, 0.08, 0.13, 0.15],
    },
]
