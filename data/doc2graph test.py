import dgl
import torch

def create_graph(data):
    """
    Function to create a graph from document data.
    
    Args:
    - data (dict): The parsed JSON-like data containing nodes and linking information.
    
    Returns:
    - DGLGraph: The created graph with nodes and edges.
    """
    # Step 1: Extract the form elements (nodes) from the data
    forms = data['form']
    num_nodes = len(forms)

    # Step 2: Create a graph with the number of nodes equal to the number of form entries
    graph = dgl.DGLGraph()
    graph.add_nodes(num_nodes)

    # Step 3: Add node features (bounding box, text, label)
    boxes = []
    texts = []
    labels = []
    for form in forms:
        boxes.append(torch.tensor(form['box'], dtype=torch.float32))
        texts.append(form['text'])
        labels.append(form['label'])
    
    # Add bounding boxes as node data (converted to a tensor)
    graph.ndata['box'] = torch.stack(boxes)
    
    # Store the text and label as node features (you can process texts into embeddings later)
    graph.ndata['text'] = texts
    graph.ndata['label'] = labels

    # Step 4: Add edges based on the linking field
    src_nodes = []
    dst_nodes = []
    for form in forms:
        for link in form['linking']:
            src_nodes.append(link[0])  # The node creating the link
            dst_nodes.append(link[1])  # The node being linked to
    
    # Add edges to the graph
    graph.add_edges(src_nodes, dst_nodes)

    return graph

# Example usage:
data = {
    "form": [
        {
            "box": [84, 109, 136, 119],
            "text": "COMPOUND",
            "label": "question",
            "linking": [[0, 37]]
        },
        {
            "box": [85, 141, 119, 152],
            "text": "SOURCE",
            "label": "question",
            "linking": [[1, 38]]
        },
        # ... more forms
    ]
}

graph = create_graph(data)

# Print the graph to verify
print(graph)
print("Number of nodes:", graph.number_of_nodes())
print("Number of edges:", graph.number_of_edges())
print("Node features (bounding boxes):", graph.ndata['box'])