#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao, Pengliang Ji
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
@File: main_semseg.py
@Time: 2021/7/20 7:49 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import FREEMUGSPOON, FREEMUGSPOON_PART
from model import DGCNN_semseg_xyz
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from plyfile import PlyData, PlyElement
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer


global room_seg
room_seg = []
global room_pred
room_pred = []
global visual_warning
global counter
counter = 0
visual_warning = True

def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    os.system('cp main_semseg.py outputs'+'/'+args.exp_name+'/'+'main_semseg.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')


def calculate_sem_IoU(pred_np, seg_np, visual=False):
    num_classes = 3
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
    return I_all / U_all 

def visualization(visu, visu_format, data, seg, pred, semseg_colors):
    global room_seg, room_pred
    global visual_warning
    global counter
    visu = visu.split('_')
    # data (16, 6, 4096)
    data = torch.squeeze(data).T
    # data (4096, 6, 16)
    # seg = torch.squeeze(seg)
    # pred = torch.squeeze(pred)

    f = open('outputs/'+args.exp_name+'/'+'visualization'+'/'+'line_'+str(counter)+'.txt', "a")
    f_gt = open('outputs/'+args.exp_name+'/'+'visualization'+'/'+'line_'+str(counter)+'_gt.txt', "a")

    # data 4096, 6, 16
    # pred: 16, 4096
    # seg: 16, 4096

    for i in range(0, data.shape[0]):
        RGB = []
        RGB_gt = [] 
        skip = False
        if not os.path.exists('outputs/'+args.exp_name+'/'+'visualization'):
            os.makedirs('outputs/'+args.exp_name+'/'+'visualization')
        
        data = data.cpu()
        # print(f"data shape: {data.shape}, pred shape: {pred.shape}, seg shape: {seg.shape}, pred i shpae, {pred[:, i].shape}")
        # print(f"data i, {data[i][:3].shape}")
        RGB.append(semseg_colors[int(pred[:,i])])
        RGB_gt.append(semseg_colors[int(seg[:,i])])
        # print("print data shape before ", data.shape, data[i].shape)
        # data = data[:,[1,2,0]] ##################
        # print("print data shape after ", data.shape)
        # print("shapes, ", data[i].shape, np.array(RGB).shape)
        xyzRGB = np.concatenate((data[i][:3].reshape(1,3), np.array(RGB)), axis=1)
        xyzRGB_gt = np.concatenate((data[i][:3].reshape(1,3), np.array(RGB_gt)), axis=1)

        # f = open('outputs/'+args.exp_name+'/'+'visualization'+'/'+'line_'+str(i)+'.txt', "a")
        # f_gt = open('outputs/'+args.exp_name+'/'+'visualization'+'/'+'line_'+str(i)+'_gt.txt', "a")
        np.savetxt(f, xyzRGB, fmt='%s', delimiter=' ') 
        np.savetxt(f_gt, xyzRGB_gt, fmt='%s', delimiter=' ') 
        
    mIoU = np.mean(calculate_sem_IoU(np.array(pred.cpu()), np.array(seg.cpu()), visual=True))
    mIoU = str(round(mIoU, 4))

    if visu_format == 'ply':
        # filepath = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+'line_'+str(i)+'_pred_'+mIoU+'.ply'
        # filepath_gt = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+'line_'+str(i)+'_gt.ply'
        filepath = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+'line_'+'_pred_'+mIoU+'_'+str(counter)+'.ply'
        filepath_gt = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+'line_'+str(counter)+'_gt.ply'
        xyzRGB = np.loadtxt('outputs/'+args.exp_name+'/'+'visualization'+'/'+'line_'+str(counter)+'.txt')
        xyzRGB_gt = np.loadtxt('outputs/'+args.exp_name+'/'+'visualization'+'/'+'line_'+str(counter)+'_gt.txt')
        xyzRGB = [(xyzRGB[i, 0], xyzRGB[i, 1], xyzRGB[i, 2], xyzRGB[i, 3], xyzRGB[i, 4], xyzRGB[i, 5]) for i in range(xyzRGB.shape[0])]
        xyzRGB_gt = [(xyzRGB_gt[i, 0], xyzRGB_gt[i, 1], xyzRGB_gt[i, 2], xyzRGB_gt[i, 3], xyzRGB_gt[i, 4], xyzRGB_gt[i, 5]) for i in range(xyzRGB_gt.shape[0])]
        vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
        PlyData([vertex]).write(filepath)
        print('PLY visualization file saved in', filepath)
        vertex = PlyElement.describe(np.array(xyzRGB_gt, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
        PlyData([vertex]).write(filepath_gt)
        print('PLY visualization file saved in', filepath_gt)
        os.system('rm -rf '+'outputs/'+args.exp_name+'/visualization'+'/*.txt')
        counter += 1

    else:
        filename = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+'line_'+str(i)+'.txt'
        filename_gt = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+'line_'+str(i)+'_gt.txt'
        filename_mIoU = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+'line_'+str(i)+'_pred_'+mIoU+'.txt'
        os.rename(filename, filename_mIoU)
        print('TXT visualization file saved in', filename_mIoU)
        print('TXT visualization file saved in', filename_gt)

        # elif visu_format != 'ply' and visu_format != 'txt':
        #     print('ERROR!! Unknown visualization format: %s, please use txt or ply.' % \
        #     (visu_format))
        #     exit()

        
def train(args, io):
    train_loader = DataLoader(FREEMUGSPOON_PART(partition='train', num_points=args.num_points), 
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(FREEMUGSPOON_PART(partition='test', num_points=args.num_points), 
                            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_semseg_xyz(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, args.epochs)

    criterion = cal_loss

    best_test_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        cnt = 0
        for data, seg in train_loader:
            data, seg = data.to(device), seg.to(device)
            # print("data shape0, ", data.size())
            data = data[:, :, :3] ## only xyz
            # print("data shape1, ", data.size()) 
            data = data.permute(0, 2, 1) ## CHECK why permute
            # print("data shape, ", data.size()) ## torch.Size([32, 9, 4096])
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            # print("seg_pred, view, seg view: ", seg_pred.size(), "!!", seg_pred.view(-1,13).size(), "!!", seg.size() , "!!", seg.view(-1,1).size())
            ## seg_pred, view, seg view:  torch.Size([32, 4096, 13]) !! torch.Size([131072, 13]) !! torch.Size([32, 4096]) !! torch.Size([131072, 1])
            ## print(f"seg_pred {seg_pred.size()} seg_pred view {seg_pred.view(-1,7).size()} seg {seg.size()} seg.view {seg.view(-1,1).size()} seg view squeeze {seg.view(-1,1).squeeze().size()}")
            ## seg_pred torch.Size([32, 4096, 7]) seg_pred view torch.Size([131072, 7]) seg torch.Size([32, 4096]) seg.view torch.Size([131072, 1]) seg view squeeze torch.Size([131072])
            # print(f"seg {seg_pred.view(7,-1).size()}, label {seg.view(-1,1).squeeze().size()}")
            # print(seg_pred.view(-1,3).shape, seg.view(-1,1).squeeze().shape, np.unique(seg.detach().cpu().numpy()), batch_size)
            # loss_tri = loss_func(seg_pred.view(-1, 2), seg.view(-1,1).squeeze())
            loss = criterion(seg_pred.view(-1, 3), seg.view(-1,1).squeeze()) ## CHECK shapes comparing (N X 4096, 1) against (N X 4096)
            # loss = loss_tri + loss_cri
            if cnt % 100 == 1:
                # print(f"epoch {epoch}, batch {cnt}, loss_tri {loss_tri}, loss_cri {loss_cri}, loss {loss}")
                print(f"epoch {epoch}, batch {cnt}, loss {loss}")
            cnt += 1
            # print(seg.view(-1, 1).dtype)
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch, 
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        for data, seg in test_loader:
            data, seg = data.to(device), seg.to(device)
            data = data[:,:,:3] ## test only xyzrgb
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred = model(data)
            # print("seg_pred ", seg_pred.shape)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            # print("after permute ", seg_pred.shape)
            # loss_tri = loss_func(seg_pred.view(-1, 2), seg.view(-1,1).squeeze())
            loss = criterion(seg_pred.view(-1, 3), seg.view(-1,1).squeeze()) ## CHECK shapes comparing (N X 4096, 1) against (N X 4096)
            # loss = loss_tri + loss_cri

            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                              test_loss*1.0/count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious))
        io.cprint(outstr)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            torch.save(model.state_dict(), 'outputs/%s/models/model_%s.t7' % (args.exp_name, args.test_area))


def test(args, io):
    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []
    test_area = args.test_area
    if (args.test_area == 'all') or (test_area == args.test_area):
        # test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=test_area),
        #                          batch_size=args.test_batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(FREEMUGSPOON_PART(partition='test', num_points=args.num_points),
                                    batch_size=args.test_batch_size, shuffle=False, drop_last=False)

        device = torch.device("cuda" if args.cuda else "cpu")
                    
        #Try to load models
        free_mug_colors = test_loader.dataset.free_mug_colors
        if args.model == 'dgcnn':
            model = DGCNN_semseg_xyz(args).to(device)
        else:
            raise Exception("Not implemented")
            
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_%s.t7' % args.test_area)))
        model = model.eval()
        test_acc = 0.0
        count = 0.0
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        for data, seg in test_loader:
            data, seg = data.to(device), seg.to(device)
            data = data[:,:,:3] ## data changed (16, 4096, 6)
            data = data.permute(0, 2, 1) # (16, 6, 4096)
            batch_size = data.size()[0]
            seg_pred = model(data) # (16, 7, 4096)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous() # (16, 4096, 7)
            pred = seg_pred.max(dim=2)[1] # (16, 4096) # return index has maximum value.
            seg_np = seg.cpu().numpy() # (16, 4096)

            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
            # visiualization
            print("test data shape, ", data.shape, seg.shape, pred.shape)
            visualization(args.visu, args.visu_format, data, seg, pred, free_mug_colors) 
        if visual_warning and args.visu != '':
            print('Visualization Failed: You can only choose a room to visualize within the scope of the test area')
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        outstr = 'Test :: test area: %s, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_area,
                                                                                                test_acc,
                                                                                                avg_per_class_acc,
                                                                                                np.mean(test_ious))
        io.cprint(outstr)
        all_true_cls.append(test_true_cls)
        all_pred_cls.append(test_pred_cls)
        all_true_seg.append(test_true_seg)
        all_pred_seg.append(test_pred_seg)

    if args.test_area == 'all':
        all_true_cls = np.concatenate(all_true_cls)
        all_pred_cls = np.concatenate(all_pred_cls)
        all_acc = metrics.accuracy_score(all_true_cls, all_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(all_true_cls, all_pred_cls)
        all_true_seg = np.concatenate(all_true_seg, axis=0)
        all_pred_seg = np.concatenate(all_pred_seg, axis=0)
        all_ious = calculate_sem_IoU(all_pred_seg, all_true_seg)
        outstr = 'Overall Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (all_acc,
                                                                                         avg_per_class_acc,
                                                                                         np.mean(all_ious))
        io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='FREEMUG', metavar='N',
                        choices=['S3DIS, BENCHMARK, FREEMUG'])
    parser.add_argument('--test_area', type=str, default=None, metavar='N',
                        choices=['1', '2', '3', '4', '5', '6', 'all'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_root', type=str, default='', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--visu', type=str, default='',
                        help='visualize the model')
    parser.add_argument('--visu_format', type=str, default='ply',
                        help='file format of visualization')
    args = parser.parse_args()

    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.test_area = 'all'
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
