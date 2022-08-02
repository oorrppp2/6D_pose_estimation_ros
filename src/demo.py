#!/home/user/anaconda3/envs/mcts/bin/python
import os
import lib.particle_filter as particle_filter
import numpy as np
import cv2
from PIL import Image
import scipy.io as scio
import numpy.ma as ma
import argparse
from utils import str2bool
from tqdm import tqdm
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ycb', help='dataset')
parser.add_argument('--dataset_root_dir', type=str, default='/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/YCB_Video_Dataset', help='dataset root dir')
parser.add_argument('--save_path', type=str, default='',  help='save results path')
parser.add_argument('--input_mask', type=str, default='pvnet',  help='save results path')
parser.add_argument('--visualization', type=str2bool, default=True,  help='visualization')
parser.add_argument('--gaussian_std', type=float, default=0.1,  help='gaussian_std')
parser.add_argument('--max_iteration', type=int, default=20,  help='max_iteration')
parser.add_argument('--tau', type=float, default=0.1,  help='tau')
parser.add_argument('--num_particles', type=int, default=180,  help='num_particles')
opt = parser.parse_args()

if __name__ == '__main__':
    pf = particle_filter.ParticleFilter(opt.dataset, opt.dataset_root_dir, visualization=opt.visualization,
    gaussian_std=opt.gaussian_std, max_iteration=opt.max_iteration, tau=opt.tau, num_particles=opt.num_particles)
    
    processing_time_record = ""
    if opt.dataset == "ycb":
        for now in tqdm(range(2949)):
            processing_time = time.time()

            img = cv2.imread('{0}/{1}-color.png'.format(pf.dataset_root_dir, pf.testlist[now]))
            depth = np.array(Image.open('{0}/{1}-depth.png'.format(pf.dataset_root_dir, pf.testlist[now])))
            posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(pf.ycb_toolbox_dir, '%06d' % (now)))
            label = np.array(posecnn_meta['labels'])
            posecnn_rois = np.array(posecnn_meta['rois'])

            labels = label[label > 0]
            labels = np.unique(labels)

            for itemid in labels:
                if itemid not in posecnn_meta['rois'][:,1]:
                    labels = np.delete(labels, np.where(labels == itemid))

            objects_region = np.zeros((480,640))
            for labels_ in labels:
                label_region = ma.getmaskarray(ma.masked_equal(label, labels_))
                objects_region[label_region] = 1

            for itemid in labels:
                if itemid not in posecnn_meta['rois'][:,1]:
                    continue
                best_score, pose = pf.start(itemid, now, img, depth, label, objects_region, dataset=opt.dataset, posecnn_meta=posecnn_meta)
                if best_score != 0:
                    np.save(opt.save_path+pf.models[itemid-1]+"_"+str(now), pose)

            processing_time_record_str = "Finish No.{0} image, Processing time : {1}".format(now, time.time() - processing_time)
            processing_time_record += processing_time_record_str + '\n'
