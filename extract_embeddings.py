import os
import argparse
import torch
import torch.nn as nn
from data import BENCHMARK
from model import DGCNN_semseg
import numpy as np

def extractor(data):
    """
    Input
    - numpy array data (1, N,6) where default N (number of points) is 4096 but it can probably be any number
    Output
    - seg_pred_np: embedding shape of (1, 4096, 7)
    - pred_np: prediction from embedding (1, 4096)
    """

    parser = argparse.ArgumentParser(description='Pointcloud embedding extractor')
    args = parser.parse_args()
    ## model weights, can be hardcoded.
    args.model_root = 'outputs/benchmark_6d_1/models/'

    ## Model related paramters
    args.emb_dims = 1024
    args.k = 20
    args.dropout = 0.25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
    #Try to load models
    model = DGCNN_semseg(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_all.t7')))
    model = model.eval()


    data = torch.from_numpy(data)
    ## data needs to be torch.float32 
    data = data.float()
    data = data.permute(0, 2, 1)
    seg_pred = model(data)
    seg_pred = seg_pred.permute(0, 2, 1).contiguous()
    pred = seg_pred.max(dim=2)[1] 
    seg_pred_np = seg_pred.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    print(seg_pred_np.shape, pred_np.shape, seg_pred_np, pred_np)

    return seg_pred_np, pred_np

if __name__ == "__main__":
    N = 2000
    data = np.random.randn(1, N ,6).astype(float)
    extractor(data)