import os
import argparse
import torch
import torch.nn as nn
from data import BENCHMARK
from model import DGCNN_semseg, DGCNN_semseg_xyz
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

def load_color_free_mug():
    colors = []
    labels = []
    f = open("prepare_data/meta/free_mug_colors.txt")
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
            if color_index >= 2:
                cv2.imwrite("prepare_data/meta/benchmark_seg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return np.array(colors)
            elif (column_index >= 1280):
                break
        row_index = row_index + int(color_size * 1.3)
        if (row_index >= img_size):
            break  

def calculate_sem_IoU(pred_np, seg_np, visual=False):
    num_classes = 2
    I_all = np.zeros(num_classes)
    U_all = np.zeros(num_classes)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(num_classes):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    if visual:
        for sem in range(num_classes):
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
    # semseg_colors = load_color_benchmark_seg()
    semseg_colors = load_color_free_mug()


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
    filepath_gt = 'testdata'+'/'+'free_mug'+'_gt.ply'
    xyzRGB = [(xyzRGB[i, 0], xyzRGB[i, 1], xyzRGB[i, 2], xyzRGB[i, 3], xyzRGB[i, 4], xyzRGB[i, 5]) for i in range(xyzRGB.shape[0])]
    xyzRGB_gt = [(xyzRGB_gt[i, 0], xyzRGB_gt[i, 1], xyzRGB_gt[i, 2], xyzRGB_gt[i, 3], xyzRGB_gt[i, 4], xyzRGB_gt[i, 5]) for i in range(xyzRGB_gt.shape[0])]
    vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
    PlyData([vertex]).write(filepath)
    print('PLY visualization file saved in', filepath)
    vertex = PlyElement.describe(np.array(xyzRGB_gt, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
    PlyData([vertex]).write(filepath_gt)
    print('PLY visualization file saved in', filepath_gt)

## activation hook added
## https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/4
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

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
    args.model_root = 'outputs/free_mug_am_cos/models/'


    ## Model related paramters
    args.emb_dims = 1024
    args.k = 20
    args.dropout = 0.25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
    #Try to load models
    # model = DGCNN_semseg(args).to(device)
    model = DGCNN_semseg_xyz(args).to(device)
    model.conv4.register_forward_hook(get_activation('conv4')) ## middle layer extract
    model.conv5.register_forward_hook(get_activation('conv5')) ## middle layer extract
    model.conv6.register_forward_hook(get_activation('conv6')) ## middle layer extract
    model.conv7.register_forward_hook(get_activation('conv7')) ## middle layer extract
    model.conv8.register_forward_hook(get_activation('conv8')) ## middle layer extract
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_all.t7')))
    model = model.eval()



    data = torch.from_numpy(data)
    ## data needs to be torch.float32 
    data = data.float()
    data = data.permute(0, 2, 1)
    seg_pred = model(data)
    seg_pred = seg_pred.permute(0, 2, 1).contiguous()
    
    # print("activation")
    # print(activation['conv8'].shape)

    conv4 = activation['conv4'].detach().cpu().numpy()
    conv4 = np.squeeze(conv4)
    conv5 = activation['conv5'].detach().cpu().numpy()
    conv5 = np.squeeze(conv5)
    conv6 = activation['conv6'].detach().cpu().numpy()
    conv6 = np.squeeze(conv6)
    conv7 = activation['conv7'].detach().cpu().numpy()
    conv7 = np.squeeze(conv7)
    conv8 = activation['conv8'].detach().cpu().numpy()
    conv8 = np.squeeze(conv8)


    ### different types of embeddings
    m = nn.Sigmoid()
    sigmoid_embeddings = m(seg_pred)
    # print(f"seg_pred {seg_pred[:3]}, sigmoid_embeddings {sigmoid_embeddings[:3]}")
    m = nn.ReLU()
    relu_embeddings = m(seg_pred)
    m = nn.Tanh()
    tanh_embeddings = m(seg_pred)
    m = nn.Softmax(dim=2)
    softmax_embeddings = m(seg_pred)


    pred = seg_pred.max(dim=2)[1] 
    seg_pred_np = seg_pred.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()

    sigmoid_embeddings = sigmoid_embeddings.detach().cpu().numpy()
    relu_embeddings = relu_embeddings.detach().cpu().numpy()
    tanh_embeddings = tanh_embeddings.detach().cpu().numpy()
    softmax_embeddings = softmax_embeddings.detach().cpu().numpy()
    
    f = h5py.File('test_2_embeddings_am_cos.h5', 'w')
    f.create_dataset('embeddings', data = seg_pred_np)
    f.create_dataset('sigmoid_embeddings', data = sigmoid_embeddings)
    f.create_dataset('relu_embeddings', data = relu_embeddings)
    f.create_dataset('tanh_embeddings', data = tanh_embeddings)
    f.create_dataset('softmax_embeddings', data = softmax_embeddings)
    f.create_dataset('conv4', data = conv4)
    f.create_dataset('conv5', data = conv5)
    f.create_dataset('conv6', data = conv6)
    f.create_dataset('conv7', data = conv7)
    f.create_dataset('conv8', data = conv8)

    # print(f"output from network {seg_pred_np}")
    # print("seg_pred_shape, ",seg_pred_np.shape, pred_np.shape) #, seg_pred_np, pred_np)
    return seg_pred_np, pred_np

if __name__ == "__main__":
    # N = 2000
    # data = np.random.randn(1, N ,6).astype(float)
    # extractor(data)

    ## test case 1
    # f = h5py.File('testdata/data_002.h5', 'r')
    # xyzRGB = f['data']
    # label = f['label']
    # data = xyzRGB[::].reshape(1,-1,6)
    # seg = label[::].reshape(1,-1)
    # print("data shape, ", data.shape, " seg shape", seg.shape)
    # seg_pred_np, pred_np = extractor(data)
    # print(pred_np.shape, f['label'][::].reshape(1,-1).shape)
    # print(calculate_sem_IoU(pred_np, f['label'][::].reshape(1,-1)))
    # print(visualization(data, seg, pred_np))


    ## test case 2
    f = h5py.File('data/xyz_data_test/data_021.h5', 'r')
    xyzRGB = f['data'][0]
    label = f['label'][0]
    data = xyzRGB[::].reshape(1,-1,3)
    seg = label[::].reshape(1,-1)
    print("data shape, ", data.shape, " seg shape", seg.shape)
    seg_pred_np, pred_np = extractor(data)
    print(pred_np.shape, f['label'][::].reshape(1,-1).shape)
    print(calculate_sem_IoU(pred_np, f['label'][0].reshape(1,-1)))
    print(visualization(data, seg, pred_np))