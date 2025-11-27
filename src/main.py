from utils import convert_fires_sql_to_csv
from wildfire_graph import WildfireGraph
from infra_graph import InfraGraph
# from graph_integration import analyze_overlay
# from clustering import run_clustering
from config import POWER_LINES_CSV_PATH, POWER_PLANTS_CSV_PATH, SUBSTATIONS_CSV_PATH

def main():
    wildfire_graph = WildfireGraph()
    wf_graph, wf_metrics = wildfire_graph.build_graph()

    infra_graph = InfraGraph(POWER_LINES_CSV_PATH, POWER_PLANTS_CSV_PATH, SUBSTATIONS_CSV_PATH)
    i_graph, i_metrics = infra_graph.build_graph()

if __name__ == "__main__":
    main()
