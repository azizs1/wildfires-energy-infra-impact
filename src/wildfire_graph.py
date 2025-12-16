import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import contextily as ctx
import geopandas as gpd
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.patches as mpatches
from sklearn.neighbors import KernelDensity
from config import WILDFIRE_SHP_PATH, WILDFIRE_PERIMS_SHP_PATH, IMG_DIR

class WildfireGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.metrics = {}

    def build_graph(self):
        mtbs_pts = gpd.read_file(WILDFIRE_SHP_PATH)

        mtbs_perims = gpd.read_file(WILDFIRE_PERIMS_SHP_PATH)
        mtbs_perims_ca = mtbs_perims[mtbs_perims["Event_ID"].str.startswith("CA")].copy()
        print(f"ca_size: {len(mtbs_perims_ca)}")

        # Reproject to 3310 for cali, then 3857 for plotting
        mtbs_perims_ca = mtbs_perims_ca.to_crs(epsg=3310)
        mtbs_perims_ca["centroid"] = mtbs_perims_ca.geometry.centroid

        # Save the plot that just has the perims with no edges
        mtbs_perims_ca_display = mtbs_perims_ca.to_crs(epsg=3857)
        mtbs_perims_ca_display["centroid"] = mtbs_perims_ca_display.geometry.centroid
        ax = mtbs_perims_ca_display.plot(column="BurnBndAc", cmap="OrRd", alpha=0.5, figsize=(12, 12))
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.DarkMatter)
        plt.savefig(IMG_DIR / "wildfire_graph_mtbs.png", dpi=300)

        # Find pairs of wildfires that overlap and remove self-joins
        pairs = gpd.sjoin(mtbs_perims_ca, mtbs_perims_ca)
        pairs = pairs[pairs["Event_ID_left"] != pairs["Event_ID_right"]]
        
        # Make the nodes just based on all of the fire regions
        for _, row in tqdm(mtbs_perims_ca.iterrows(), total=len(mtbs_perims_ca), desc="Adding nodes"):
            self.graph.add_node(row["Event_ID"], size=row["BurnBndAc"], severity=row["High_T"], 
                                geometry=row.geometry)
        
        # Make the edges based on spatial overlaps
        for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Adding edges"):
            self.graph.add_edge(row["Event_ID_left"], row["Event_ID_right"])

        # Plot the wildfires. Using BurnBndAc to color based on size for now
        fig, ax = plt.subplots(figsize=(12, 12))
        mtbs_perims_ca_display.plot(column="BurnBndAc", cmap="OrRd", alpha=0.5, ax=ax)

        # Plot edges between centroids
        for u, v in self.graph.edges():
            c1 = mtbs_perims_ca_display.loc[mtbs_perims_ca_display["Event_ID"] == u, "centroid"].values[0]
            c2 = mtbs_perims_ca_display.loc[mtbs_perims_ca_display["Event_ID"] == v, "centroid"].values[0]
            xs = [c1.x, c2.x]
            ys = [c1.y, c2.y]
            ax.plot(xs, ys, color="blue", linewidth=0.5, alpha=0.5)

        # Add dark basemap for new plot
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.DarkMatter)
        plt.savefig(IMG_DIR / "wildfire_graph_edges.png", dpi=300)

        self._dbscan_wildfires(mtbs_perims_ca)
        # self._wildfire_hotspots(mtbs_perims_ca)

        return self.graph, self._compute_metrics()
    
    def _dbscan_wildfires(self, mtbs_perims_ca, eps=20000, min_samples=17):
        # Reproject to 3310
        mtbs_perims_proj = mtbs_perims_ca.to_crs(epsg=3310)
        mtbs_perims_proj["centroid"] = mtbs_perims_proj.geometry.centroid
        
        centroids = np.array([[row["centroid"].x, row["centroid"].y] 
                            for _, row in mtbs_perims_proj.iterrows()])
        ids = mtbs_perims_proj["Event_ID"].tolist()

        # Run DBSCAN with Euclidean distance
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(centroids)

        # Assign cluster labels to nodes
        labels = clustering.labels_
        for node_id, label in zip(ids, labels):
            if node_id in self.graph.nodes:
                self.graph.nodes[node_id]["cluster"] = int(label)

        # Store cluster summary
        clusters = {}
        for node_id, label in zip(ids, labels):
            clusters.setdefault(str(int(label)), []).append(node_id)
        self.metrics["dbscan_clusters"] = clusters

        cluster_map = {}
        for n, d in self.graph.nodes(data=True):
            if "cluster" in d:
                cluster_map[n] = d["cluster"]
            else:
                cluster_map[n] = -1

        mtbs_perims_mercator = mtbs_perims_ca.to_crs(epsg=3857)  # for basemap display
        mtbs_perims_mercator["cluster"] = mtbs_perims_mercator["Event_ID"].map(cluster_map)

        unique_clusters = mtbs_perims_mercator["cluster"].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        color_dict = {c: colors[i] for i, c in enumerate(unique_clusters)}
        color_dict[-1] = "whitesmoke" # this is the noise so make it gray, we'll lower alpha later
        mtbs_perims_mercator["color"] = mtbs_perims_mercator["cluster"].map(color_dict)

        fig, ax = plt.subplots(figsize=(12, 12))
        for cluster_id, group in mtbs_perims_mercator.groupby("cluster"):
            if cluster_id == -1:
                group.plot(color=color_dict[cluster_id], alpha=0.1, ax=ax)
            else:
                group.plot(color=color_dict[cluster_id], alpha=0.5, ax=ax)

        ctx.add_basemap(ax, source=ctx.providers.CartoDB.DarkMatter)

        # Build legend handles manually
        handles = []
        for c in sorted(color_dict.keys()):
            patch = mpatches.Patch(color=color_dict[c], label=f"Cluster {c}")
            handles.append(patch)
        ax.legend(handles=handles, title="DBSCAN Clusters", loc="lower left", fontsize="small")

        plt.savefig(IMG_DIR / "wildfire_dbscan_clusters.png", dpi=300)
        plt.close(fig)

        return clusters

    def _wildfire_hotspots(self, mtbs_perims_ca, bandwidth=20000):
        gdf = mtbs_perims_ca.to_crs(epsg=3310)
        gdf["centroid"] = gdf.geometry.centroid
        coords = np.array([[pt.x, pt.y] for pt in gdf["centroid"]])

        # use KDE for the heatmaps
        kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
        kde.fit(coords)

        x_min, y_min, x_max, y_max = gdf.total_bounds
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid_coords = np.vstack([xx.ravel(), yy.ravel()]).T
        zz = np.exp(kde.score_samples(grid_coords)).reshape(xx.shape)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 12))
        gdf.plot(ax=ax, facecolor="none", edgecolor="gray", alpha=0.3)
        ax.imshow(zz, extent=(x_min, x_max, y_min, y_max),
                origin="lower", cmap="hot", alpha=0.6)
        plt.savefig(IMG_DIR / "wildfire_kde_hotspots.png", dpi=300)
        plt.close(fig)

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

    def get_perimeters_gdf(self):        
        mtbs_perims = gpd.read_file(WILDFIRE_PERIMS_SHP_PATH)
        mtbs_perims_ca = mtbs_perims[mtbs_perims["Event_ID"].str.startswith("CA")].copy()
        mtbs_perims_ca = mtbs_perims_ca.to_crs(epsg=3310)
        return mtbs_perims_ca
