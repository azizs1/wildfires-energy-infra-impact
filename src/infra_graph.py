import pandas as pd
import networkx as nx
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import Point

from utils import plot_infra
from config import POWER_LINES_CSV_PATH, SUBSTATIONS_CSV_PATH, IMG_DIR

class InfraGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.metrics = {}
        self.lines_df = None
        self.substations_df = None

    def build_graph(self):
        """Construct the infrastructure graph."""
        self._preprocessing()

        G = nx.Graph()

        # Add substation nodes
        subs = self.substations_df
        if subs is not None:
            for _, row in subs.iterrows():
                if "name" not in row or pd.isna(row["name"]):
                    continue
                    
                node_id = row["name"]
                G.add_node(
                    node_id,
                    node_type="substation",
                    state=row.get("state"),
                    latitude=row.get("latitude"),
                    longitude=row.get("longitude"),
                    sub_id=row.get("id") or row.get("objectid")
                )

        # Add transmission line edges
        lines = self.lines_df
        if lines is not None and {"sub_1", "sub_2"}.issubset(lines.columns):
            for _, row in lines.iterrows():
                s1, s2 = row["sub_1"], row["sub_2"]
                
                # Skip invalid edges
                if pd.isna(s1) or pd.isna(s2):
                    continue
                if s1 == s2:
                    continue
                if not (G.has_node(s1) and G.has_node(s2)):
                    continue

                voltage = row.get("voltage")
                length = row.get("shape__length") or row.get("shape_length")
                
                G.add_edge(
                    s1, s2,
                    line_id=row.get("id") or row.get("objectid"),
                    voltage=voltage,
                    status=row.get("status"),
                    length=length,
                    importance=1.0 / voltage if voltage and voltage > 0 else 1.0
                )

        print(f"Created Infra Graph: {G.number_of_nodes()} substations, {G.number_of_edges()} transmission lines")

        self.graph = G
        self._compute_metrics()
        self._plot_geo()

        return self.graph, self.metrics

    def _preprocessing(self):
        """Load and clean substation/transmission data."""
        lines = pd.read_csv(POWER_LINES_CSV_PATH)
        subs = pd.read_csv(SUBSTATIONS_CSV_PATH)

        NULL_LIKE = ["NOT AVAILABLE", "N/A", "NA", ""]

        for df in (lines, subs):
            df.columns = df.columns.str.strip().str.lower()
            df.replace(NULL_LIKE, pd.NA, inplace=True)

        # Normalize name columns for consistent matching
        for col, df in [("sub_1", lines), ("sub_2", lines), ("name", subs)]:
            if col in df.columns:
                df[col] = df[col].str.strip().str.upper()

        if {"sub_1", "sub_2"} <= set(lines.columns):
            lines = lines.dropna(subset=["sub_1", "sub_2"], how="all")

        if "state" in subs.columns:
            subs = subs[subs["state"] == "CA"]

        self.lines_df = lines
        self.substations_df = subs

    def _compute_metrics(self):
        """Compute basic network metrics."""
        G = self.graph
        
        self.metrics['degree_centrality'] = nx.degree_centrality(G)
        self.metrics['clustering'] = nx.clustering(G)
        self.metrics['num_clusters'] = nx.number_connected_components(G)
        self.metrics['clusters'] = [list(c) for c in nx.connected_components(G)]
        self.metrics['cluster_sizes'] = [len(c) for c in self.metrics['clusters']]
        self.metrics['num_nodes'] = G.number_of_nodes()
        self.metrics['num_edges'] = G.number_of_edges()
        self.metrics['betweenness'] = nx.betweenness_centrality(G, weight='importance')

        return self.metrics

    def _plot_geo(self):
        """Plot the infrastructure network with basemap."""
        infra_nodes = []
        for n, data in self.graph.nodes(data=True):
            if "latitude" in data and "longitude" in data:
                infra_nodes.append({
                    "id": n,
                    "geometry": Point(data["longitude"], data["latitude"]),
                    "node_type": data.get("node_type"),
                    "state": data.get("state")
                })
                
        infra_gdf = gpd.GeoDataFrame(infra_nodes, crs="EPSG:4326").to_crs(epsg=3310)

        fig, ax = plt.subplots(figsize=(12, 12))
        infra_gdf_display = infra_gdf.to_crs(epsg=3857)
        plot_infra(ax, self.graph, infra_gdf_display)
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.DarkMatter)
        plt.savefig(IMG_DIR / "infra.png", dpi=300)
        print(f"Saved infrastructure plot to {IMG_DIR / 'infra.png'}")

    def get_high_risk_substations(self, top_n=20):
        """Get substations ranked by betweenness centrality."""
        betweenness = self.metrics.get('betweenness', {})
        sorted_subs = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        return sorted_subs[:top_n]

    def get_network_summary(self):
        num_substations = self.metrics.get("num_nodes", 0)
        num_transmission_lines = self.metrics.get("num_edges", 0)
        num_connected_components = self.metrics.get("num_clusters", 0)
        
        cluster_sizes = self.metrics.get("cluster_sizes", [0])
        largest_component_size = max(cluster_sizes)

        avg_degree = 0
        if self.graph:
            degrees = []
            for node, degree in self.graph.degree():
                degrees.append(degree)
            avg_degree = np.mean(degrees)
        
        summary = {
            "num_substations": num_substations, 
            "num_transmission_lines": num_transmission_lines, 
            "num_connected_components": num_connected_components, 
            "largest_component_size": largest_component_size,
            "avg_degree": avg_degree
        }
        
        return summary