import pandas as pd

PERIOD_ECO = 18  # years
PERIOD_SIM = pd.Timedelta(days=365)
START_SIM = pd.to_datetime('2023-01-01 00:00')
FREQ_SIM = pd.Timedelta(hours=1)