from plyfile import PlyData, PlyElement
import h5py
import numpy as np

f = h5py.File('data_000.h5', 'r')
print(f.keys())
xyzRGB = f['data'][0]
print(xyzRGB.shape)
xyzRGB = [(xyzRGB[i, 0], xyzRGB[i, 1], xyzRGB[i, 2], xyzRGB[i, 3]*255, xyzRGB[i, 4]*255, xyzRGB[i, 5]*255) for i in range(xyzRGB.shape[0])]
print(xyzRGB[9])
# xyzRGB_gt = [(xyzRGB_gt[i, 0], xyzRGB_gt[i, 1], xyzRGB_gt[i, 2], xyzRGB_gt[i, 3], xyzRGB_gt[i, 4], xyzRGB_gt[i, 5]) for i in range(xyzRGB_gt.shape[0])]
filepath = 'data_000.ply'
vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
PlyData([vertex]).write(filepath)
print('PLY visualization file saved in', filepath)
# vertex = PlyElement.describe(np.array(xyzRGB_gt, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
# PlyData([vertex]).write(filepath_gt)
# print('PLY visualization file saved in', filepath_gt)