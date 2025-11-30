import networkx as nx
import pandas as pd
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from tqdm import tqdm
import contextily as ctx
import geopandas as gpd
from config import WILDFIRE_SHP_PATH, WILDFIRE_PERIMS_SHP_PATH

class WildfireGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.metrics = {}

    def build_graph(self):
        mtbs_pts = gpd.read_file(WILDFIRE_SHP_PATH)

        mtbs_perims = gpd.read_file(WILDFIRE_PERIMS_SHP_PATH)
        mtbs_perims_ca = mtbs_perims[mtbs_perims["Event_ID"].str.startswith("CA")].copy()
        print(f"ca_size: {len(mtbs_perims_ca)}")

        # Reproject for basemap
        mtbs_perims_ca = mtbs_perims_ca.to_crs(epsg=3857)
        mtbs_perims_ca["centroid"] = mtbs_perims_ca.geometry.centroid

        # Save the plot that just has the perims with no edges
        ax = mtbs_perims_ca.plot(column="BurnBndAc", cmap="OrRd", alpha=0.5, figsize=(12, 12))
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.DarkMatter)
        plt.savefig("wildfire_graph_mtbs.png", dpi=300)

        # Find pairs of wildfires that overlap and remove self-joins
        pairs = gpd.sjoin(mtbs_perims_ca, mtbs_perims_ca)
        pairs = pairs[pairs["Event_ID_left"] != pairs["Event_ID_right"]]
        
        # Make the nodes just based on all of the fire regions
        for _, row in tqdm(mtbs_perims_ca.iterrows(), total=len(mtbs_perims_ca), desc="Adding nodes"):
            self.graph.add_node(row["Event_ID"], size=row["BurnBndAc"], severity=row["High_T"])
        
        # Make the edges based on spatial overlaps
        for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Adding edges"):
            self.graph.add_edge(row["Event_ID_left"], row["Event_ID_right"])

        # Plot the wildfires. Using BurnBndAc to color based on severity for now
        fig, ax = plt.subplots(figsize=(12, 12))
        mtbs_perims_ca.plot(column="BurnBndAc", cmap="OrRd", alpha=0.5, ax=ax)

        # Plot edges between centroids
        for u, v in self.graph.edges():
            c1 = mtbs_perims_ca.loc[mtbs_perims_ca["Event_ID"] == u, "centroid"].values[0]
            c2 = mtbs_perims_ca.loc[mtbs_perims_ca["Event_ID"] == v, "centroid"].values[0]
            xs = [c1.x, c2.x]
            ys = [c1.y, c2.y]
            ax.plot(xs, ys, color="blue", linewidth=0.5, alpha=0.5)

        # Add dark basemap for new plot
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.DarkMatter)
        plt.savefig("wildfire_graph_edges.png", dpi=300)

        return self.graph, self._compute_metrics()

    def _preprocessing(self, df):
        df['DISCOVERY_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'], errors='coerce')
        df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'DISCOVERY_DATE'])
        df = df[df['STATE'] == 'CA']
        return df

    def _compute_metrics(self):
        self.metrics['degree_centrality'] = nx.degree_centrality(self.graph)
        self.metrics['clustering'] = nx.clustering(self.graph)
        self.metrics['betweenness'] = nx.betweenness_centrality(self.graph)
        self.metrics['num_clusters'] = nx.number_connected_components(self.graph)
        self.metrics['clusters'] = [list(c) for c in nx.connected_components(self.graph)]

        # print("Degree centrality:", self.metrics['degree_centrality'])
        # print("Clustering:", self.metrics['clustering'])
        # print("Betweenness:", self.metrics['betweenness'])
        # print("Number of clusters:", self.metrics['num_clusters'])
        # print("Cluster sizes:", [len(c) for c in self.metrics['clusters']])
        return self.metrics