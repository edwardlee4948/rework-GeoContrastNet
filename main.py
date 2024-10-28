from pathlib import Path
import torch
import argparse
import pprint
import numpy as np

# Ensure deterministic behavior
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# step 1. build graphs
#
# step 1.1. load funsd data
# train paths and test paths

#