import pandas as pd
import networkx as nx
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import numpy as np

from collections import Counter
from shapely.geometry import Point
from utils import plot_infra
from config import POWER_LINES_CSV_PATH, POWER_PLANTS_CSV_PATH, SUBSTATIONS_CSV_PATH, IMG_DIR
from sklearn.cluster import DBSCAN

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
        """Perform calculations for all metrics."""
        G = self.graph
        self.metrics['degree_centrality'] = nx.degree_centrality(G)
        self.metrics['clustering'] = nx.clustering(G)
        self.metrics['num_clusters'] = nx.number_connected_components(G)
        self.metrics['clusters'] = [list(c) for c in nx.connected_components(G)]
        self.metrics['betweenness'] = nx.betweenness_centrality(G)
        self.metrics['cluster_sizes'] = [len(c) for c in self.metrics['clusters']]
        self.metrics['num_nodes'] = G.number_of_nodes()
        self.metrics['num_edges'] = G.number_of_edges()
        self.dbscan_clustering(node_type="substation", eps_deg=0.05, min_samples=3)
        self.dbscan_clustering(node_type="plant", eps_deg=0.05, min_samples=3)

        return self.metrics

    def dbscan_clustering(self, node_type="substation", eps_deg=0.05, 
                            min_samples=3, top_n=5):
        """Run DBSCAN clustering and plot results with basemap."""
        # Extract nodes with coordinates
        nodes = [
            (nid, d["longitude"], d["latitude"]) 
            for nid, d in self.graph.nodes(data=True)
            if d.get("node_type") == node_type 
            and d.get("longitude") 
            and d.get("latitude")
        ]

        if not nodes:
            print(f"No {node_type} nodes to cluster.")
            return
        
        # Run DBSCAN clustering
        df = pd.DataFrame(nodes, columns=["node_id", "lon", "lat"])
        df["node_type"] = node_type
        coords = df[["lon", "lat"]].values
        df["cluster_label"] = DBSCAN(eps=eps_deg, min_samples=min_samples).fit_predict(coords)
        
        # Convert to Web Mercator for basemap
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326").to_crs(epsg=3857)
        
        # Identify top clusters
        labels = gdf["cluster_label"]
        n_total = len(labels)
        n_noise = (labels == -1).sum()
        cluster_counts = Counter(labels[labels != -1])
        top_clusters = [c for c, _ in cluster_counts.most_common(top_n)]
        n_clusters = len(cluster_counts)
        print(f"\n--- THIS IS THE DBSCAN Analysis for {node_type} ---")
        print(f"Total points: {n_total}")
        print(f"Clusters (excl. noise): {n_clusters}")
        print(f"Noise points: {n_noise} ({n_noise / n_total:.1%})")
        print("Top cluster sizes:", cluster_counts.most_common(5))

        gdf_clustered = gdf[labels != -1]
        
        comp = (gdf_clustered.groupby(["cluster_label", "node_type"]).size().unstack(fill_value=0))

        print("\nCluster composition (node counts):")
        print(comp.head())

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 12))
        cmap = cm.get_cmap('Set1', top_n)
        
        # Plot noise points
        noise = gdf[labels == -1]
        if not noise.empty:
            noise.plot(ax=ax, color="lightgray", markersize=5, alpha=0.4, label="Noise")
        
        # Plot top clusters
        for i, cluster_id in enumerate(top_clusters):
            cluster = gdf[labels == cluster_id]
            cluster.plot(ax=ax, color=cmap(i), markersize=20, alpha=0.7, label=f"Cluster {cluster_id} (n={len(cluster)})")
        
        # Plot remaining clusters
        other = gdf[~labels.isin(top_clusters + [-1])]
        if not other.empty:
            other.plot(ax=ax, color="gray", markersize=8, alpha=0.3, label=f"Other ({n_clusters - top_n})")
        
        # Style and save
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.DarkMatter)
        ax.set_title(f"DBSCAN: {node_type.capitalize()}s (Top {top_n})\neps={eps_deg}, min_samples={min_samples}, {n_clusters} clusters",fontsize=14, color='white')
        ax.legend(loc="upper right", fontsize=10, facecolor='black', edgecolor='white', labelcolor='white')
        plt.tight_layout()
        plt.savefig(IMG_DIR / f"infra_clusters_{node_type}.png", dpi=300, bbox_inches="tight")
        print(f"Saved to {IMG_DIR / f'infra_clusters_{node_type}.png'}")


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
