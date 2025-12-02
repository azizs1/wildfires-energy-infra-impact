import networkx as nx
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from shapely.geometry import Point
from utils import plot_infra, plot_wildfires
from config import IMG_DIR

def analyze_overlay(infra_graph, wf_graph):
    print(infra_graph.nodes[list(infra_graph.nodes())[0]])
    print(wf_graph.nodes[list(wf_graph.nodes())[0]])

    # The idea for plotting is basically load all the nodes into a GDF and plot them after
    infra_nodes = []
    for n, data in infra_graph.nodes(data=True):
        if "latitude" in data and "longitude" in data:
            # It HAS to be called geometry to make gdf happy
            infra_nodes.append({
                "id": n, "geometry": Point(data["longitude"], data["latitude"]),
                "node_type": data.get("node_type", "infra"), "state": data.get("state")
            })
    infra_gdf = gpd.GeoDataFrame(infra_nodes, crs="EPSG:4326").to_crs(epsg=3857)

    wf_nodes = []
    for n, data in wf_graph.nodes(data=True):
        if "geometry" in data:
            wf_nodes.append({"id": n, "geometry": data["geometry"], 
                             "x": data["geometry"].centroid.x, "y": data["geometry"].centroid.y,
                             "size": data["size"], "severity": data["severity"]})
    wf_gdf = gpd.GeoDataFrame(wf_nodes, geometry="geometry", crs=f"EPSG:{3857}").to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(12, 12))
    plot_infra(ax, infra_graph, infra_gdf)
    plot_wildfires(ax, wf_gdf)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.DarkMatter)
    plt.savefig(IMG_DIR / "overlay.png", dpi=300)

    return infra_gdf, wf_gdf