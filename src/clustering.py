import networkx as nx
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from utils import plot_infra, plot_wildfires
from config import IMG_DIR

def run_clustering(infra_gdf, wf_gdf, infra_graph, wf_graph):
    print(wf_gdf.head())
    pairs = gpd.sjoin(wf_gdf, wf_gdf)
    pairs = pairs[pairs["id_left"] != pairs["id_right"]]

    cluster_map = {id: i for i, id in enumerate(wf_gdf["id"], start=1)}
    cluster_id = 0
    visited = set()

    for _, row in pairs.iterrows():
        left, right = row["id_left"], row["id_right"]
        if left not in visited and right not in visited:
            cluster_id += 1
        cluster_map[left] = cluster_id
        cluster_map[right] = cluster_id
        visited.add(left)
        visited.add(right)

    wf_gdf = wf_gdf.copy()
    wf_gdf["cluster"] = wf_gdf["id"].map(cluster_map)
    clustered_gdf = wf_gdf.dissolve(by="cluster")

    fig, ax = plt.subplots(figsize=(12, 12))
    plot_wildfires(ax, clustered_gdf)
    plot_infra(ax, infra_graph, infra_gdf)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.DarkMatter)
    plt.savefig(IMG_DIR / "clustered_fires.png", dpi=300)

    pass