import h5py
import numpy as np

for i in range(15,20):
    f = h5py.File('data/freemugspoon_data_test/fms_data_'+str(i).zfill(3)+'.h5', 'r+')     # open the file
    data = f['label']       # load the data
    data[:,900:1300] = 2                      # assign new values to data
    f.close()                          # close the file
