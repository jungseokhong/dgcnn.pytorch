import os
import argparse
import torch
import torch.nn as nn
from data import BENCHMARK
from model import DGCNN_semseg
import numpy as np
import h5py
import json
import cv2
from plyfile import PlyData, PlyElement

def load_color_benchmark_seg():
    colors = []
    labels = []
    f = open("prepare_data/meta/benchmark_seg_colors.txt")
    for line in json.load(f):
        colors.append(line['color'])
        labels.append(line['label'])
    semseg_colors = np.array(colors)
    semseg_colors = semseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1500
    img = np.zeros((500, img_size, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (img_size, 750), [255, 255, 255], thickness=-1)
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for _ in range(0, img_size):
        column_index = 32
        for _ in range(0, img_size):
            color = semseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(img, (column_index, row_index), (column_index + color_size, row_index + color_size),
                          color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
            img = cv2.putText(img, label, (column_index + int(color_size * 1.15), row_index + int(color_size / 2)),
                              font,
                              0.7, (0, 0, 0), 2)
            column_index = column_index + 200
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 7:
                cv2.imwrite("prepare_data/meta/benchmark_seg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return np.array(colors)
            elif (column_index >= 1280):
                break
        row_index = row_index + int(color_size * 1.3)
        if (row_index >= img_size):
            break  

def calculate_sem_IoU(pred_np, seg_np, visual=False):
    I_all = np.zeros(7)
    U_all = np.zeros(7)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(7):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    if visual:
        for sem in range(7):
            if U_all[sem] == 0:
                I_all[sem] = 1
                U_all[sem] = 1

    # print(f"U all {U_all}, I all {I_all}")
    zeros_idx = np.where(U_all == 0)[0]
    U_all = np.delete(U_all, zeros_idx)
    I_all = np.delete(I_all, zeros_idx)
    # print(f"after U all {U_all}, I all {I_all}")
    return I_all / U_all 


def visualization(data, seg, pred):
    ## data shape (1,N,6), seg (1,N), pred (1,N)
    data = data[0] # shape (N,6)
    semseg_colors = load_color_benchmark_seg()
    
    # print("color")
    # print(semseg_colors[pred][0])
    # print(semseg_colors[pred].shape)
    
    RGB = semseg_colors[pred][0]
    RGB_gt = semseg_colors[seg][0]

    print(sum(pred[0]==seg[0])/pred.shape[1])


    xyzRGB = np.concatenate((data[:,:3], RGB), axis=1)
    xyzRGB_gt = np.concatenate((data[:,:3], RGB_gt), axis=1)

    mIoU = np.mean(calculate_sem_IoU(np.array(pred), np.array(seg), visual=False))
    print(mIoU)
    mIoU = str(round(mIoU, 4))

    filepath = 'testdata'+'/'+'test2'+'_pred_'+mIoU+'_'+'.ply'
    filepath_gt = 'testdata'+'/'+'test2'+'_gt.ply'
    xyzRGB = [(xyzRGB[i, 0], xyzRGB[i, 1], xyzRGB[i, 2], xyzRGB[i, 3], xyzRGB[i, 4], xyzRGB[i, 5]) for i in range(xyzRGB.shape[0])]
    xyzRGB_gt = [(xyzRGB_gt[i, 0], xyzRGB_gt[i, 1], xyzRGB_gt[i, 2], xyzRGB_gt[i, 3], xyzRGB_gt[i, 4], xyzRGB_gt[i, 5]) for i in range(xyzRGB_gt.shape[0])]
    vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
    PlyData([vertex]).write(filepath)
    print('PLY visualization file saved in', filepath)
    vertex = PlyElement.describe(np.array(xyzRGB_gt, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
    PlyData([vertex]).write(filepath_gt)
    print('PLY visualization file saved in', filepath_gt)


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
    # args.model_root = 'outputs/benchmark_6d_1/models/'
    args.model_root = 'outputs/benchmark_6d_random/models/'


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
    f = h5py.File('test_2_embeddings.h5', 'w')
    f.create_dataset('embeddings', data = seg_pred_np)
    print(f"output from network {seg_pred_np}")
    # print("seg_pred_shape, ",seg_pred_np.shape, pred_np.shape) #, seg_pred_np, pred_np)
    return seg_pred_np, pred_np

if __name__ == "__main__":
    # N = 2000
    # data = np.random.randn(1, N ,6).astype(float)
    # extractor(data)

    ## test case
    f = h5py.File('testdata/data_002.h5', 'r')
    xyzRGB = f['data']
    label = f['label']
    data = xyzRGB[::].reshape(1,-1,6)
    seg = label[::].reshape(1,-1)
    print("data shape, ", data.shape, " seg shape", seg.shape)
    seg_pred_np, pred_np = extractor(data)
    print(pred_np.shape, f['label'][::].reshape(1,-1).shape)
    print(calculate_sem_IoU(pred_np, f['label'][::].reshape(1,-1)))
    print(visualization(data, seg, pred_np))
