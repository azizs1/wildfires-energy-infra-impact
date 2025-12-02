import pandas as pd
import networkx as nx
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from shapely.geometry import Point
from utils import plot_infra
from config import POWER_LINES_CSV_PATH, POWER_PLANTS_CSV_PATH, SUBSTATIONS_CSV_PATH, IMG_DIR

class InfraGraph:
    def __init__(self):
        """Initialize dem variables"""
        self.graph = nx.Graph()
        self.metrics = {}
        self.lines_df = None
        self.plants_df = None
        self.substations_df = None

    def build_graph(self):
        """Uses substation/power plants as nodes and transmission lines as edges"""
        self._preprocessing()

        G = nx.Graph()

        # Substation nodes
        subs = self.substations_df
        if subs is not None:
            for _, row in subs.iterrows():
                if "name" not in row or pd.isna(row["name"]):
                    continue
                node_id = row["name"]

                G.add_node(node_id, node_type="substation", state=row.get("state"), latitude=row.get("latitude"), longitude=row.get("longitude"), sub_id=row.get("id") or row.get("objectid"))

        # Power plant nodes
        plants = self.plants_df
        if plants is not None:
            for _, row in plants.iterrows():

                plant_code = row.get("plant_code")

                # Skip missing plant_code
                if plant_code is None or pd.isna(plant_code):
                    continue

                plant_id = f"PLANT_{str(plant_code).strip()}"

                G.add_node(plant_id, node_type="plant", name=row.get("name"), state=row.get("state"), latitude=row.get("latitude"), longitude=row.get("longitude"))

        # Transmission line edges
        lines = self.lines_df
        if lines is not None and {"sub_1", "sub_2"}.issubset(lines.columns):
            for _, row in lines.iterrows():
                s1 = row["sub_1"]
                s2 = row["sub_2"]

                # Skip empty nodes
                if pd.isna(s1) or pd.isna(s2):
                    continue
                if s1 == s2:
                    continue  
                if not (G.has_node(s1) and G.has_node(s2)):
                    continue

                edge_attrs = {
                    "line_id": row.get("id") or row.get("objectid"),
                    "voltage": row.get("voltage"),
                    "status": row.get("status"),
                    "length": row.get("shape__length") or row.get("shape_length"),
                }

                G.add_edge(s1, s2, **edge_attrs)

        print(f"Created Infra Graph... {G.number_of_edges()} edges & {G.number_of_nodes()} nodes")

        self.graph = G
        self._compute_metrics()
        self._plot_geo()

        return self.graph, self.metrics

    def _preprocessing(self):
        """Clean data, convert N/As, drop required observations."""
        lines = pd.read_csv(POWER_LINES_CSV_PATH)
        plants = pd.read_csv(POWER_PLANTS_CSV_PATH)
        subs = pd.read_csv(SUBSTATIONS_CSV_PATH, low_memory=False)

        NULL_LIKE = ["NOT AVAILABLE", "N/A", "NA", ""]

        for df in (lines, plants, subs):
            df.columns = df.columns.str.strip().str.lower()
            df.replace(NULL_LIKE, pd.NA, inplace=True)

        def clean_name(x):
            if isinstance(x, str):
                return x.strip().upper()
            return x

        # lines endpoints (SUB_1 / SUB_2)
        if "sub_1" in lines.columns:
            lines["sub_1"] = lines["sub_1"].map(clean_name)
        if "sub_2" in lines.columns:
            lines["sub_2"] = lines["sub_2"].map(clean_name)

        if "name" in subs.columns:
            subs["name"] = subs["name"].map(clean_name)

        if {"sub_1", "sub_2"} <= set(lines.columns):
            lines = lines.dropna(subset=["sub_1", "sub_2"], how="all")

        # filter only CA
        if "state" in subs.columns:
            subs = subs[subs["state"] == "CA"]

        if "state" in plants.columns:
            plants = plants[plants["state"] == "CA"]

        self.lines_df = lines
        self.plants_df = plants
        self.substations_df = subs

        return

    def _compute_metrics(self):
        G = self.graph
        self.metrics['degree_centrality'] = nx.degree_centrality(G)
        self.metrics['clustering'] = nx.clustering(G)
        self.metrics['num_clusters'] = nx.number_connected_components(G)
        self.metrics['clusters'] = [list(c) for c in nx.connected_components(G)]
        self.metrics['betweenness'] = nx.betweenness_centrality(G)
        self.metrics['cluster_sizes'] = [len(c) for c in self.metrics['clusters']]
        self.metrics['num_nodes'] = G.number_of_nodes()
        self.metrics['num_edges'] = G.number_of_edges()
        return self.metrics

    def _plot_geo(self, state=None, show_plants=True):
        """Plot the infrastructure network"""
        if self.graph is None or self.graph.number_of_nodes() == 0:
            self.build_graph()

        # The idea for plotting is basically load all the nodes into a GDF and plot them after
        infra_nodes = []
        for n, data in self.graph.nodes(data=True):
            if "latitude" in data and "longitude" in data:
                # It HAS to be called geometry to make gdf happy
                infra_nodes.append({
                    "id": n, "geometry": Point(data["longitude"], data["latitude"]),
                    "node_type": data.get("node_type", "infra"), "state": data.get("state")
                })
        infra_gdf = gpd.GeoDataFrame(infra_nodes, crs="EPSG:4326").to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(12, 12))
        plot_infra(ax, self.graph, infra_gdf)
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.DarkMatter)
        plt.savefig(IMG_DIR / "infra.png", dpi=300)
