from utils import convert_fires_sql_to_csv
from wildfire_graph import WildfireGraph
from infra_graph import build_infra_graph
# from graph_integration import analyze_overlay
# from clustering import run_clustering
from config import WILDFIRE_CSV_PATH, POWER_LINES_CSV_PATH, POWER_PLANTS_CSV_PATH, SUBSTATIONS_CSV_PATH

def main():
    convert_fires_sql_to_csv()

    wildfire_graph = WildfireGraph("data/fires.csv", 5, 24)
    wf_graph, metrics = wildfire_graph.build_graph()

    infra_graph = build_infra_graph(POWER_LINES_CSV_PATH, POWER_PLANTS_CSV_PATH, SUBSTATIONS_CSV_PATH)



if __name__ == "__main__":
    main()
