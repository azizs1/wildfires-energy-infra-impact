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

def plot_wildfires(ax, wf_gdf, col="size"):
    if not wf_gdf.empty:
        wf_gdf.plot(ax=ax, column=col, cmap="OrRd", alpha=0.5, 
                    edgecolor="red", linewidth=0.1, legend=True)
    return

def plot_infra(ax, infra_graph, infra_gdf):
    if not infra_gdf.empty:
        substations = infra_gdf[infra_gdf["node_type"] == "substation"]
        plants = infra_gdf[infra_gdf["node_type"] == "plant"]

        ax.scatter(substations.geometry.x, substations.geometry.y,
                color="cyan", marker="^", s=2, label="Substations")
        ax.scatter(plants.geometry.x, plants.geometry.y,
                color="yellow", marker="o", s=2, label="Power Plants")

        # Drawing the edges here
        for start, end in infra_graph.edges():
            start_geom = infra_gdf.loc[infra_gdf["id"] == start, "geometry"].values[0]
            end_geom = infra_gdf.loc[infra_gdf["id"] == end, "geometry"].values[0]

            ax.plot([start_geom.x, end_geom.x], [start_geom.y, end_geom.y], 
                    color="white", linewidth=0.2, alpha=0.5)
    return