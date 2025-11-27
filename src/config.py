# config.py
# This file contains config options, i.e. paths to files.
# 11/22/2025

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

WILDFIRE_SQLITE_PATH  = DATA_DIR / "FPA_FOD_20221014.sqlite"
WILDFIRE_CSV_PATH     = DATA_DIR / "FPA_FOD_20221014.csv"
POWER_LINES_CSV_PATH  = DATA_DIR / "Electric__Power_Transmission_Lines.csv"
POWER_PLANTS_CSV_PATH = DATA_DIR / "Power_Plants.csv"
SUBSTATIONS_CSV_PATH  = DATA_DIR / "Substations.csv"

WILDFIRE_PKL_PATH = DATA_DIR / "FPA_FOD_20221014.pkl"
WILDFIRE_SHP_PATH = DATA_DIR / "MTBS_Points" / "mtbs_FODpoints_DD.shp"
WILDFIRE_PERIMS_SHP_PATH = DATA_DIR / "MTBS_Perimeters" / "mtbs_perims_DD.shp"