import networkx as nx

class InfraGraph:
    def __init__(self, lines_csv, plants_csv, substations_csv):
        self.lines_csv_path = lines_csv
        self.plants_csv_path = plants_csv
        self.substations_csv_path = substations_csv
        self.graph = nx.Graph()
        self.metrics = {}

    def build_graph(self):        
        return self.graph, self.metrics

    def _preprocessing(self):
        return

    def _compute_metrics(self):
        return