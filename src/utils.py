# utils.py
# This file implements utility functions for use throughout
# 11/21/2025

import sqlite3
import pandas as pd
import os
from pathlib import Path
from config import WILDFIRE_SQLITE_PATH, WILDFIRE_CSV_PATH

def convert_fires_sql_to_csv():
    if os.path.exists(WILDFIRE_CSV_PATH):
        print("Wildfires CSV file already exists!")
        return
    
    if not os.path.exists(WILDFIRE_SQLITE_PATH):
        print("Wildfires SQLite does not exist!")
        return

    sl_connection = sqlite3.connect(WILDFIRE_SQLITE_PATH)
    fires_df = pd.read_sql("SELECT * FROM Fires;", sl_connection)
    sl_connection.close()

    fires_df.to_csv(WILDFIRE_CSV_PATH, index=False)
    print(f"Saved CSV to {WILDFIRE_CSV_PATH}")
    return