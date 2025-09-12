import pandas as pd


DTI = pd.date_range(start='2023-01-01', end='2024-01-01 00:00', freq='h', tz='Europe/Berlin', inclusive='left')
FREQ_HOURS = pd.Timedelta(DTI.freq).total_seconds() / 3600
TIME_PRJ_YRS = 18
