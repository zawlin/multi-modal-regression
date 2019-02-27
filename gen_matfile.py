import scipy.io as spio
import os
import numpy as np
import glob
from helperFunctions import classes
import utils

def setup_renderforcnn():
    src = 'data/renderforcnn_orig'
    dst = 'data/renderforcnn'

    classes_map = {
        '02691156':'aeroplane',
        '02834778':'bicycle',
        '02858304':'boat',
        '02876657':'bottle',
        '02924116':'bus',
        '02958343':'car',
        '03001627':'chair',
        '03211117':'diningtable',
        '03790512':'motorbike',
        '04256520':'sofa',
        '04379243':'train',
        '04468005':'tvmonitor'
        }
    for i in classes_map.keys():
        if not os.path.exists('%s/%s'%(dst,classes_map[i])):
            cmd = 'ln -rs %s/%s %s/%s'%(src,i,dst,classes_map[i])
            print(cmd)
            os.system(cmd)

def gen_info_pkl_files(root):
    for cls in classes:
        print(cls)
        files = glob.glob(root+cls+'/**/*.png',recursive=True)
        files = [f.replace('.png','') for f in files]
        pkl = {'image_names':np.asarray(files,dtype='object')}
        spio.savepkl(root+cls+'_info.pkl',pkl)

def gen_info_pkl_files(root):
    for cls in classes:
        print(cls)
        files = glob.glob(root+cls+'/**/*.png',recursive=True)
        files = [f.replace('.png','') for f in files]
        pkl = {'image_names':np.asarray(files,dtype='object')}
        utils.save(root+cls+'_info.pkl',pkl)

#setup_renderforcnn()
#gen_info_pkl_files('data/renderforcnn/')
#gen_info_pkl_files('data/augmented2/')
#gen_info_pkl_files('data/flipped_new/train/')
#gen_info_pkl_files('data/flipped_new/test/')
#gen_info_pkl_files('data/test/')
#gen_info_pkl_files('data/augmented2/')

gen_info_pkl_files('data/flipped_new/test/')
gen_info_pkl_files('data/flipped_new/train/')
gen_info_pkl_files('data/renderforcnn/')
gen_info_pkl_files('data/augmented2/')
