import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go

from shapely.geometry import Point, LineString
from config import POWER_LINES_CSV_PATH, POWER_PLANTS_CSV_PATH, SUBSTATIONS_CSV_PATH

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
        self.plot_geo()

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
        self.metrics['clusters'] = list(nx.connected_components(G))
        self.metrics['betweenness'] = nx.betweenness_centrality(G)
        self.metrics['cluster_sizes'] = [len(c) for c in self.metrics['clusters']]
        self.metrics['num_nodes'] = G.number_of_nodes()
        self.metrics['num_edges'] = G.number_of_edges()
        return self.metrics

    def plot_geo(self, state=None, show_plants=True):
        """Plot the infrastructure network using Plotly (Scattergeo) with low node opacity and low edge opacity."""
        if self.graph is None or self.graph.number_of_nodes() == 0:
            self.build_graph()

        G = self.graph

        node_lons_sub = []
        node_lats_sub = []
        node_lons_plant = []
        node_lats_plant = []

        included_nodes = set()

        if state == "CA":
            lon_min, lon_max = -124.48, -114.13
            lat_min, lat_max = 32.53, 42.01

        for n, data in G.nodes(data=True):
            lon = data.get("longitude")
            lat = data.get("latitude")

            if pd.isna(lon) or pd.isna(lat):
                continue

            if state == "CA":
                if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                    continue

            included_nodes.add(n)

            if data.get("node_type") == "substation":
                node_lons_sub.append(lon)
                node_lats_sub.append(lat)
            elif data.get("node_type") == "plant":
                node_lons_plant.append(lon)
                node_lats_plant.append(lat)

        edge_lons = []
        edge_lats = []

        for u, v, edata in G.edges(data=True):
            if u not in included_nodes or v not in included_nodes:
                continue

            nu = G.nodes[u]
            nv = G.nodes[v]

            lon_u, lat_u = nu.get("longitude"), nu.get("latitude")
            lon_v, lat_v = nv.get("longitude"), nv.get("latitude")

            if pd.isna(lon_u) or pd.isna(lat_u) or pd.isna(lon_v) or pd.isna(lat_v):
                continue

            edge_lons += [lon_u, lon_v, None]
            edge_lats += [lat_u, lat_v, None]

        fig = go.Figure()

        # Transmission lines
        if edge_lons:
            fig.add_trace(
                go.Scattergeo(
                    lon=edge_lons,
                    lat=edge_lats,
                    mode="lines",
                    line=dict(width=0.4, color="rgba(150,150,150,0.10)"),
                    hoverinfo="none",
                    name="Transmission Lines",
                )
            )

        # Substations
        if node_lons_sub:
            fig.add_trace(
                go.Scattergeo(
                    lon=node_lons_sub,
                    lat=node_lats_sub,
                    mode="markers",
                    marker=dict(
                        size=1.5,
                        color="blue",
                        opacity=0.25, 
                    ),
                    name="Substations",
                )
            )

        # Plants
        if show_plants and node_lons_plant:
            fig.add_trace(
                go.Scattergeo(
                    lon=node_lons_plant,
                    lat=node_lats_plant,
                    mode="markers",
                    marker=dict(
                        size=4,
                        symbol="triangle-up",
                        color="red",
                        opacity=0.35,
                    ),
                    name="Plants",
                )
            )

        # Layout
        fig.update_layout(
            title="California Electric Infrastructure",
            showlegend=True,
            geo=dict(
                projection_type="mercator",
                showland=True,
                landcolor="rgb(240,240,240)",
                lataxis=dict(range=[32.53, 42.01]), 
                lonaxis=dict(range=[-124.48, -114.13]), 
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        fig.show()
        return fig
