import networkx as nx
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from shapely.geometry import Point

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
            end_geom   = infra_gdf.loc[infra_gdf["id"] == end, "geometry"].values[0]

            ax.plot([start_geom.x, end_geom.x], [start_geom.y, end_geom.y], 
                    color="white", linewidth=0.2, alpha=0.5)

    if not wf_gdf.empty:
        wf_gdf.plot(ax=ax, column="size", cmap="OrRd", alpha=0.5, 
                    edgecolor="red", linewidth=0.1, legend=True)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.DarkMatter)
    plt.savefig("overlay.png", dpi=300)