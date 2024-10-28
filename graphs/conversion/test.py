import pickle
import torch
import dgl


with open(
    "graphs/conversion/train_dgl_graphs.pkl",
    "rb",
) as file:
    data = pickle.load(file)


def collate(batch):
    graphs, labels = map(list, zip(*batch))
    graphs = dgl.batch(graphs)
    print(graphs)
    print(torch.cat(labels, dim=0))
    return graphs, torch.cat(labels, dim=0)


train_loader = torch.utils.data.DataLoader(
    data,
    batch_size=4,
    collate_fn=collate,
    shuffle=True,
)
