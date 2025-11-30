import os
import json
import pickle
import argparse
from config import CACHE_DIR

from utils import convert_fires_sql_to_csv
from wildfire_graph import WildfireGraph
from infra_graph import InfraGraph
from graph_integration import analyze_overlay
# from clustering import run_clustering

def load_build(graph_name, build_graph_func, no_cache):
    graph_cache_pkl = f"{CACHE_DIR}/{graph_name}_graph_cache.pkl"
    metrics_cache_json = f"{CACHE_DIR}/{graph_name}_metrics_cache.json"
    
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    if not no_cache and os.path.exists(graph_cache_pkl) and os.path.exists(metrics_cache_json):
        with open(graph_cache_pkl, "rb") as f:
            graph = pickle.load(f)

        with open(metrics_cache_json, "r") as f:
            metrics = json.load(f)
    else:
        graph, metrics = build_graph_func()
        with open(graph_cache_pkl, "wb") as f:
            pickle.dump(graph, f)
        
        with open(metrics_cache_json, "w") as f:
            json.dump(metrics, f, indent=2)
    
    return graph, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true",
                        help="Force rebuild of graphs instead of using cache")
    args = parser.parse_args()

    wildfire_graph = WildfireGraph()
    wf_graph, wf_metrics = load_build("wildfires", wildfire_graph.build_graph, args.no_cache)

    infra_graph = InfraGraph()
    i_graph, i_metrics = load_build("infra", infra_graph.build_graph, args.no_cache)

    analyze_overlay(i_graph, wf_graph)

if __name__ == "__main__":
    main()
