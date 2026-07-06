"""Uploaded diagnostic data validation and parsing."""
from io import BytesIO
import pandas as pd

def parse_dcrm_csv(content : bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(content))