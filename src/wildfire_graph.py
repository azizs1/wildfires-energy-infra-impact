import networkx as nx
import pandas as pd
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import KDTree
import pickle
import contextily as ctx
import geopandas as gpd
from config import WILDFIRE_PKL_PATH, WILDFIRE_SHP_PATH, WILDFIRE_PERIMS_SHP_PATH

class WildfireGraph:
    def __init__(self, fires_csv, spatial_dist, time_dist):
        self.fires_csv_path = fires_csv
        self.spatial_dist = spatial_dist
        self.time_dist = time_dist
        self.graph = nx.Graph()
        self.metrics = {}

    def build_graph(self):
        mtbs_pts = gpd.read_file(WILDFIRE_SHP_PATH)

        # print(mtbs.head())
        # print(mtbs.columns)
        # mtbs.plot(markersize=2, figsize=(10, 10), alpha=0.6)
        # plt.savefig("wildfire_graph_mtbs.png", dpi=300)

        mtbs_perims = gpd.read_file(WILDFIRE_PERIMS_SHP_PATH)
        print(f"all_size: {len(mtbs_perims)}")
        mtbs_perims_ca = mtbs_perims[mtbs_perims["Event_ID"].str.startswith("CA")]
        print(f"ca_size: {len(mtbs_perims_ca)}")
        # print(mtbs_perims_ca.head())


        # Reproject for basemap
        mtbs_perims_web = mtbs_perims_ca.to_crs(epsg=3857)

        # Plot polygons colored by severity
        ax = mtbs_perims_web.plot(column="BurnBndAc", cmap="OrRd", alpha=0.5, figsize=(12, 12))
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.DarkMatter)
        plt.savefig("wildfire_graph_mtbs.png", dpi=300)
        # cols_to_use = ["FOD_ID", "LATITUDE", "LONGITUDE", "DISCOVERY_DATE", "FIRE_SIZE",
        #     "FIRE_SIZE_CLASS", "NWCG_CAUSE_CLASSIFICATION", "STATE", "OWNER_DESCR"]
        # fires_df = self._preprocessing(pd.read_csv(self.fires_csv_path, usecols=cols_to_use))

        # print("Building wildfire graph...")
        
        # for _, row in tqdm(fires_df.iterrows(), total=len(fires_df), desc="Adding nodes"):
        #     self.graph.add_node(row['FOD_ID'], latitude=row['LATITUDE'], longitude=row['LONGITUDE'],
        #                         date=row['DISCOVERY_DATE'], fire_size=row['FIRE_SIZE'], 
        #                         fire_class=row['FIRE_SIZE_CLASS'], cause=row['NWCG_CAUSE_CLASSIFICATION'],
        #                         state=row['STATE'], owner=row['OWNER_DESCR'])

        # coords = fires_df[['LATITUDE', 'LONGITUDE']].to_numpy()
        # kd_tree = KDTree(coords)

        # for idx, row in tqdm(fires_df.iterrows(), total=len(fires_df), desc="Building edges"):
        #     coord = (row['LATITUDE'], row['LONGITUDE'])
        #     radius_deg = self.spatial_dist / 69.0
        #     neighbor_idxs = kd_tree.query_ball_point([row['LATITUDE'], row['LONGITUDE']], r=radius_deg)

        #     for n_idx in neighbor_idxs:
        #         if n_idx <= idx:  # avoid duplicates
        #             continue
        #         neighbor = fires_df.iloc[n_idx]
        #         time_diff = abs((row['DISCOVERY_DATE'] - neighbor['DISCOVERY_DATE']).total_seconds()) / 3600
        #         if time_diff <= self.time_dist:
        #             dist_miles = geodesic(coord, (neighbor['LATITUDE'], neighbor['LONGITUDE'])).miles
        #             self.graph.add_edge(row['FOD_ID'], neighbor['FOD_ID'],
        #                                 spatial_dist=dist_miles, time_diff=time_diff)
    
        # print("Wildfire Graph Info:")
        # print(f"Nodes: {self.graph.number_of_nodes()}")
        # print(f"Edges: {self.graph.number_of_edges()}")
        # print("Number of clusters:", nx.number_connected_components(self.graph))
        # # print("Cluster sizes:", [len(c) for c in list(nx.connected_components(self.graph))])

        # with open(WILDFIRE_PKL_PATH, "wb") as f:
        #     pickle.dump(self.graph, f)

        # pos = {node:(data['longitude'], data['latitude']) for node, data in self.graph.nodes(data=True)}

        # nx.draw(self.graph, pos, node_size=0.001, alpha=0.6, with_labels=False)
        # plt.xlabel("Longitude")
        # plt.ylabel("Latitude")
        # plt.show()
        # plt.savefig("wildfire_graph.png", dpi=300)

        return self.graph, self._compute_metrics

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
        self.metrics['clusters'] = list(nx.connected_components(self.graph))

        print("Degree centrality:", self.metrics['degree_centrality'])
        print("Clustering:", self.metrics['clustering'])
        print("Betweenness:", self.metrics['betweenness'])
        print("Number of clusters:", self.metrics['num_clusters'])
        print("Cluster sizes:", [len(c) for c in self.metrics['clusters']])

        return