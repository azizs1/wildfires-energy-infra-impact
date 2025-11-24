import networkx as nx

class WildfireGraph:
    def __init__(self, fires_csv, spatial_dist, time_dist):
        self.fires_csv_path = fires_csv
        self.spatial_dist = spatial_dist
        self.time_dist = time_dist
        self.graph = nx.Graph()
        self.metrics = {}

    def build_graph(self):        
        return self.graph, self.metrics

    def _preprocessing(self):
        return

    def _compute_metrics(self):
        return