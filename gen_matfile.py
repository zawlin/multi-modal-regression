import scipy.io as spio
from helperFunctions import parse_name
import os
import numpy as np
from helperFunctions import classes

def gen_renderforcnn():
    pass

def gen_renderforcnn():
    src = 'data/renderforcnn_orig/'
    dst = 'data/renderforcnn/'

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
        os.system('ln -rs %s/%s %s/%s'%(src,i,dst,classes_map[i]))

    dst = 'data/renderforcnn/'

    for cls in classes:
        for root, dirs, files in os.walk(dst+cls):
            pass



def gen_pascal_flipped():
    root = 'data/flipped_new/train/'
    for cls in classes:
        #print(classes)
        for _, _, files in os.walk(root+cls):
            files = [f.replace('.png','') for f in files]
            mat = {'image_names':np.asarray(files,dtype='object')}
            spio.savemat(root+cls+'_info.mat',mat)
            tmp = spio.loadmat(root+cls+'_info.mat', squeeze_me=True)

#gen_renderforcnn()
gen_pascal_flipped()