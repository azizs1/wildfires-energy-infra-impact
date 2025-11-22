# utils.py
# This file implements utility functions for use throughout
# 11/21/2025

import sqlite3
import pandas as pd
from pathlib import Path
from config import WILDFIRE_SQLITE_PATH

def get_fires_df():
    sl_connection = sqlite3.connect(WILDFIRE_SQLITE_PATH)
    fires_df = pd.read_sql("SELECT * FROM Fires;", sl_connection)
    return fires_df