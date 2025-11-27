import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go

from shapely.geometry import Point, LineString

class InfraGraph:
    def __init__(self, lines_csv, plants_csv, substations_csv):
        """Initialize dem variables"""
        self.lines_csv_path = lines_csv
        self.plants_csv_path = plants_csv
        self.substations_csv_path = substations_csv
        self.graph = nx.Graph()
        self.metrics = {}
        self.lines_df = None
        self.plants_df = None
        self.substations_df = None

    def build_graph(self):
        """Uses substation/power plants as nodes and transmission lines as edges"""
        self._preprocessing()
        self.california_records()

        G = nx.Graph()

        # Substation nodes
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
                    sub_id=row.get("id") or row.get("objectid"),
                )

        # Power plant nodes
        plants = self.plants_df
        if plants is not None:
            for _, row in plants.iterrows():
                plant_id = None
                if "plant_code" in plants.columns:
                    plant_id = f"PLANT_{row['plant_code']}"
                elif "name" in plants.columns:
                    plant_id = f"PLANT_{row['name']}"

                if plant_id is None or pd.isna(plant_id):
                    continue

                G.add_node(
                    plant_id,
                    node_type="plant",
                    name=row.get("name"),
                    state=row.get("state"),
                    latitude=row.get("latitude"),
                    longitude=row.get("longitude"),
                )

        # Transmission line edges
        lines = self.lines_df
        if lines is not None and {"sub_1", "sub_2"}.issubset(lines.columns):
            for _, row in lines.iterrows():
                s1 = row["sub_1"]
                s2 = row["sub_2"]

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

        return self.graph, self.metrics

    def _preprocessing(self):
        """Clean data, convert N/As, drop required observations."""
        lines = pd.read_csv(self.lines_csv_path)
        plants = pd.read_csv(self.plants_csv_path)
        subs = pd.read_csv(self.substations_csv_path)

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

        self.lines_df = lines
        self.plants_df = plants
        self.substations_df = subs

        return

    def _compute_metrics(self):
        G = self.graph

        if G is None or G.number_of_nodes() == 0:
            self.metrics["warning"] = "Graph is empty?!"
            return self.metrics

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        # Count node types
        num_substations = sum(
            1 for _, data in G.nodes(data=True)
            if data.get("node_type") == "substation"
        )
        num_plants = sum(
            1 for _, data in G.nodes(data=True)
            if data.get("node_type") == "plant"
        )

        # Connected components
        components = list(nx.connected_components(G))
        num_components = len(components)
        largest_component_size = max(len(c) for c in components) if components else 0

        # Average degree (2E / N for undirected graph)
        avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0.0

        self.metrics.update(
            {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "num_substations": num_substations,
                "num_plants": num_plants,
                "num_components": num_components,
                "largest_component_size": largest_component_size,
                "avg_degree": avg_degree,
            }
        )

        print("Basic infra graph metrics:", self.metrics)

        self.plot_geo()
        return self.metrics

    def california_records(self):
        lines = self.lines_df
        plants = self.plants_df
        subs = self.substations_df

        self.metrics['num_ca_plants'] = len(plants[plants['state'] == 'CA'])
        self.metrics['num_ca_substations'] = len(subs[subs['state'] == 'CA'])

        ca_sub_names = set(subs[subs['state'] == 'CA']['name'])
        ca_lines = lines[lines['sub_1'].isin(ca_sub_names) | lines['sub_2'].isin(ca_sub_names)]
        self.metrics['num_ca_lines'] = len(ca_lines)

        print(self.metrics)
        return self.metrics

    def plot_geo(self, state=None, show_plants=True):
        """Plot the infrastructure network using Plotly (Scattergeo)
        with low node opacity and low edge opacity."""
        import plotly.graph_objects as go
        import pandas as pd

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

        # Build plotly figure
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

        # Plants (triangles)
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

        # Plot layout
        title = "US Electric Infrastructure"
        if state == "CA":
            title = "California Electric Infrastructure"

        fig.update_layout(
            title=title,
            showlegend=True,
            geo=dict(
                scope="usa",
                projection_type="albers usa",
                showland=True,
                landcolor="rgb(240, 240, 240)",
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        fig.show()
        return fig
