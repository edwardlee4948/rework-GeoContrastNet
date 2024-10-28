import pickle
import dgl


# Load the pickled data
def convert(file_in, file_out):
    with open(file_in, "rb") as f:
        data = pickle.load(f)

    graphs = []

    for elem in data:
        # Assuming node features and edge features are available in the dataset
        node_data = elem["node_data"]
        edge_data = elem["edge_data"]

        # You need node connections (src and dst) to construct the graph
        # For simplicity, assuming you have them somewhere in the edge data

        # Create a DGL graph
        graph = dgl.graph((elem["src_ids"], elem["dst_ids"]))

        # Assign node features
        graph.ndata["geom"] = node_data["geom"]
        graph.ndata["feat"] = node_data["feat"]
        graph.ndata["norm"] = node_data["norm"]
        graph.ndata["label"] = node_data["label"]
        graph.ndata["Geometric"] = node_data["Geometric"]
        graph.ndata["area"] = node_data["area"]
        graph.ndata["regional_encoding"] = node_data["regional_encoding"]

        # Assign edge features
        graph.edata["feat"] = edge_data["feat"]
        graph.edata["distance_not_tresh"] = edge_data["distance_not_tresh"]
        graph.edata["weights"] = edge_data["weights"]
        graph.edata["label"] = edge_data["label"]
        graph.edata["angle"] = edge_data["angle"]
        graph.edata["distance"] = edge_data["distance"]
        graph.edata["discrete_bin_edges"] = edge_data["discrete_bin_edges"]

        # Append the created graph to the list of graphs
        graphs.append(graph)

    with open(file_out, "wb") as f:
        pickle.dump(graphs, f)


if __name__ == "__main__":

    convert(file_in="new_test_data.pkl", file_out="test_dgl_graphs.pkl")
    convert(file_in="new_train_data.pkl", file_out="train_dgl_graphs.pkl")
    convert(file_in="new_val_data.pkl", file_out="val_dgl_graphs.pkl")
