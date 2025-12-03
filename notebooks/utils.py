from pathlib import Path
import pandas as pd

def load_caribbean_weather():
    path = Path("../data/processed")
    
    return pd.read_parquet(path / "caribbean_temp_precip_1980_2025.parquet")