
import os
import sys
import cv2
import time

libpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(libpath)
sys.path.append(libpath + '/../CenterFindNet/lib')
import render
import objloader
# import ctypes
from PIL import Image
import numpy as np
import scipy.io as scio
import random
import time
from transforms3d.quaternions import axangle2quat, mat2quat, qmult, qinverse
from transforms3d.euler import quat2euler, mat2euler, euler2quat
import matplotlib.pyplot as plt
import numpy.ma as ma
import open3d as o3d

from filterpy.monte_carlo import systematic_resample
from filterpy.monte_carlo import stratified_resample

from sample import *
from utils import quaternion_matrix, calc_pts_diameter, draw_object
from scipy import spatial

import torch
from torch.autograd import Variable
from centroid_prediction_network import CentroidPredictionNetwork

np.random.seed(0)

class ParticleFilter():
    def __init__(self, cam_fx, cam_fy, cam_cx, cam_cy, visualization=False, gaussian_std=0.1, max_iteration=20, tau=0.1, num_particles=180):
        self.cam_cx = cam_cx
        self.cam_cy = cam_cy
        self.cam_fx = cam_fx
        self.cam_fy = cam_fy
        self.cad_model_root_dir = libpath + '/../models/ycb/'

        self.models = [
            "004_sugar_box",            # 1
            "006_mustard_bottle",       # 2
            "005_tomato_soup_can",      # 3
            "024_bowl",                 # 4
            "water_bottle",             # 5
            "coca_cola"                 # 6
        ]

        # Same number of particles (180) for all objects.
        self.num_particles = [num_particles * 2, num_particles, num_particles * 2, num_particles, num_particles, num_particles]

        self.taus = [tau, tau * 2, tau, tau, tau, tau, tau, tau, tau, tau,
                    tau, tau, tau, tau, tau * 2, tau, tau * 2, tau, tau * 2, tau * 2, tau]


        self.info = {'Height':480, 'Width':640, 'fx':self.cam_fx, 'fy':self.cam_fy, 'cx':self.cam_cx, 'cy':self.cam_cy}
        render.setup(self.info)

        self.visualization = visualization
        self.num_points = 1000
        self.gaussian_std = gaussian_std
        self.max_iteration = max_iteration

        self.min_particles = 256

        self.K = [[self.cam_fx, 0, self.cam_cx],
            [0, self.cam_fy, self.cam_cy],
            [0, 0, 1]]
        self.K = np.array(self.K)


        """ Load Centroid Prediction Network """
        model_path = libpath + '/../CenterFindNet/trained_model/CPN_model_91_0.00023821471899932882.pth'
        # model_path = os.path.dirname(os.path.abspath(__file__)) +
        self.estimator = CentroidPredictionNetwork(num_points = self.num_points)
        self.estimator.load_state_dict(torch.load(model_path))
        self.estimator.eval()

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.context = {}
        self.diameters = {}
        self.model_points = {}
        self.np_model_points = {}
        self.corners = {}
        for model in self.models:
            print("*** " + model + " adding... ***")
            cad_model_path = self.cad_model_root_dir + '{}/textured_simple.obj'.format(model)
            pcd_path = self.cad_model_root_dir + '{}/textured_simple.pcd'.format(model)

            V, F = objloader.LoadTextureOBJ_VF_only(cad_model_path)
            self.context[model] = render.SetMesh(V, F)

            pcd = o3d.io.read_point_cloud(pcd_path)
            pcd = np.asarray(pcd.points)

            x_max = np.max(pcd[:,0])
            x_min = np.min(pcd[:,0])
            y_max = np.max(pcd[:,1])
            y_min = np.min(pcd[:,1])
            z_max = np.max(pcd[:,2])
            z_min = np.min(pcd[:,2])

            corner = np.array([[x_min, y_min, z_min],
                                        [x_max, y_min, z_min],
                                        [x_max, y_max, z_min],
                                        [x_min, y_max, z_min],

                                        [x_min, y_min, z_max],
                                        [x_max, y_min, z_max],
                                        [x_max, y_max, z_max],
                                        [x_min, y_max, z_max]])
            self.corners[model] = corner

            self.diameters[model] = calc_pts_diameter(pcd)

            dellist = [j for j in range(0, len(pcd))]
            dellist = random.sample(dellist, len(pcd) - self.num_points)

            model_point = np.delete(pcd, dellist, axis=0)
            np_model_point = model_point.copy()

            model_point = torch.from_numpy(model_point.astype(np.float32)).view(1, self.num_points, 3)
            self.model_points[model] = model_point
            self.np_model_points[model] = np_model_point

        self.rotation_samples = {}
        for i, model in enumerate(self.models):
            self.rotation_samples[model] = get_rotation_samples(model, num_samples=self.num_particles[i])


    def mat2pdf_np(self, distance_matrix, mean, std):
        coeff = 1/(np.sqrt(2*np.pi) * std)
        pdf = coeff * np.exp(- (distance_matrix - mean)**2 / (2 * std**2))
        return pdf

    def mat2pdf(self, distance_matrix, mean, std):
        coeff = torch.ones_like(distance_matrix) * (1/(np.sqrt(2*np.pi) * std))
        mean = torch.ones_like(distance_matrix) * mean
        std = torch.ones_like(distance_matrix) * std
        pdf = coeff * torch.exp(- (distance_matrix - mean)**2 / (2 * std**2))
        return pdf

    def estimate(self, pos, weights):
        """returns mean and variance of the weighted particles"""
        mean = np.average(pos, weights=weights, axis=0)
        var  = np.average((pos - mean)**2, weights=weights, axis=0)
        return mean, var

    def start(self, itemid, img, depth, label, objects_region, posecnn_meta="", rois=[]):
        # cv2.imshow("other_objects_region", other_objects_region)
        # cv2.waitKey(0)
        model = self.models[itemid-1]
        context = self.context[model]
        diameter = self.diameters[model]
        model_point = self.model_points[model]
        np_model_point = self.np_model_points[model]
        corner = self.corners[model]
        num_particles = self.num_particles[itemid-1]

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(label, 1))
        mask = mask_label * mask_depth
        masked_depth = mask * depth

        if len(rois) != 0:
            cmin, rmin, cmax, rmax = rois
        else:
            ret, masks, stats, centroid = cv2.connectedComponentsWithStats(label * mask_label)
            stats_argsort = np.argsort(stats[:,4])
            best_box_index = stats_argsort[len(stats_argsort)-2]
            cmin, rmin, width, height, _ = stats[best_box_index]
            cmax = cmin + width
            rmax = rmin + height

        masked_depth[:rmin, :] = 0
        masked_depth[rmax:, :] = 0
        masked_depth[:, :cmin] = 0
        masked_depth[:, cmax:] = 0

        mask_label[:rmin, :] = 0
        mask_label[rmax:, :] = 0
        mask_label[:, :cmin] = 0
        mask_label[:, cmax:] = 0
        # cv2.imshow("masked_depth", masked_depth)
        # cv2.waitKey(0)
        masked_depth_copy = masked_depth.copy()
        masked_depth_copy = masked_depth_copy[masked_depth_copy > 0]

        var = np.var(masked_depth_copy)
        mean = np.mean(masked_depth_copy)
        masked_depth[masked_depth -mean > var + diameter] = 0

        other_objects_region = objects_region.copy()
        other_objects_region[mask_label] = 0
        depth_zero_in_mask = np.logical_and(mask_label, np.logical_not(depth))
        other_objects_region[depth_zero_in_mask] = 1

        choose = masked_depth[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > self.num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            print("The number of detected pixels is insufficient.")
            return 0, 0

        depth_masked = masked_depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        pt2 = depth_masked #* cam_scale
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        while not torch.cuda.is_available():
            time.sleep(0.001)

        cloud = torch.from_numpy(cloud.astype(np.float32)).view(1, self.num_points, 3)

        """ Centroid prediction for generating the initial translation of the particle filter system. """
        centroid = self.estimator(cloud, model_point)
        centroid = centroid[0,0,:].cpu().data.numpy()

        best_score = 0
        final_pose = None
        pose_distribution = None

        """ Initial pose hypotheses """
        poses = []
        for sample_ryp in self.rotation_samples[model]:
            quat = list(euler2quat(sample_ryp[0], sample_ryp[1], sample_ryp[2]))
            pose = np.hstack([centroid[0], centroid[1], centroid[2], quat])
            poses.append(pose)
        print("num of particles : ", len(poses))
        render.setSrcDepthImage(self.info, masked_depth.copy(), other_objects_region.copy())

        for iters in range(self.max_iteration):

            # if iters < 5:
            #     threshold = self.taus[itemid-1] / (iters * 2+1)
            # else:
            threshold = self.taus[itemid-1] / 10.0

            render.setNumOfParticles(len(poses), int(threshold * 1000000))
            for i in range(len(poses)):
                pose = poses[i]

                transform_matrix = quaternion_matrix(pose[3:]).astype(np.float32)
                transform_matrix[:3,3] = pose[:3]
                render.render(context, transform_matrix)
                render.calcMatchingScore(self.info)

            """ Access to the CUDA memory to get the scores of each pose hypothesis. """
            scores = render.getMatchingScores(len(poses))
            if len(scores[scores > 0]) == 0:
                print("All of scores are 0.")
                return 0, 0

            """ Likelihood calculation """
            pdf_matrix = self.mat2pdf_np(scores / max(scores), 1, self.gaussian_std)
            if iters == 0:
                pose_distribution = pdf_matrix / np.sum(pdf_matrix + 1e-8)
                weights = pose_distribution
            else:
                pose_distribution = np.exp(np.log(pdf_matrix + 1e-8) + np.log(pose_distribution))
                weights = pose_distribution / np.sum(pose_distribution + 1e-8)
                pose_distribution = weights


            # print("="*50)
            # print("Current iteration : ", iters)
            # print("Best score : ", best_score)
            # print("Best matching score : ", max(scores))
            # print("particles size : ", len(poses))
            # print("best pose : ", final_pose)

            """ Current state estimation by weighted average. """
            mu, var = self.estimate(poses, weights)

            render.setNumOfParticles(1, int(threshold * 1000000))
            transform_matrix = quaternion_matrix(mu[3:]).astype(np.float32)
            transform_matrix[:3,3] = mu[:3]
            render.render(context, transform_matrix)
            render.calcMatchingScore(self.info)
            matching_score = render.getMatchingScores(1)

            # matching_score = max(scores)
            if matching_score > best_score:
                best_score = matching_score
                final_pose = mu
                # Taking pose of the max score method. This has lower performance than the weighted average method.
                # final_pose = poses[scores.argmax()]


            """
                Visualization
            """
            if self.visualization == True:
                transform_matrix = quaternion_matrix(final_pose[3:]).astype(np.float32)
                transform_matrix[:3,3] = final_pose[:3]

                draw_box = img.copy()
                draw_object(itemid, self.K, transform_matrix[:3,:3], transform_matrix[:3,3], draw_box, np_model_point, corner)
                r, g, b = cv2.split(draw_box)
                draw_box = cv2.merge((b, g, r))
                cv2.imshow("draw_box", draw_box)

                # if iters == self.max_iteration-1:
                trial = 1
                cv2.imwrite("/home/robot/IROS_Experiment/exp_video/{0}/6D_box_{1}_iters{2}.png".format(str(itemid), str(trial), str(iters)), draw_box)
                np.save("/home/robot/IROS_Experiment/exp_video/{0}/estimated_pose_{1}_iters{2}.npy".format(str(itemid), str(trial), str(iters)), transform_matrix)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    exit(0)
                    break

            """ Resampling"""

            N = len(weights)
            positions = (np.random.random() + np.arange(N)) / N
            indexes = np.zeros(N, "i")
            cumulative_sum = np.cumsum(weights)
            i, j = 0, 0
            while i < N and j < N:
                if positions[i] < cumulative_sum[j]:
                    indexes[i] = j
                    i += 1
                else:
                    j += 1
            values, counts = np.unique(indexes, return_counts=True)

            reduce_count = 0
            reduce_index = -1

            """ Reducing the number of particles. It has to be used carefully. """
            # for i, count in enumerate(counts):
            #     if len(poses) > min_particles and count > (len(poses) / 10):
            #         reduce_count = count / 2
            #         if len(poses) - reduce_count < min_particles:
            #             reduce_count = len(poses) - min_particles
            #         reduce_index = i


            """ Propagation """
            prev_poses = poses.copy()
            prev_pose_distribution = pose_distribution.copy()
            poses = []
            pose_distribution = []

            for count_i, index in enumerate(values):
                count = counts[count_i]
                pose_copy = prev_poses[index]
                score = scores[index]
                if count_i == reduce_index:
                    count -= reduce_count
                count = int(count)

                mu, var = pose_copy[0], 0.05 * pow(1-score, 2)
                t1 = np.random.normal(mu, var, count)
                mu, var = pose_copy[1], 0.05 * pow(1-score, 2)
                t2 = np.random.normal(mu, var, count)
                mu, var = pose_copy[2], 0.05 * pow(1-score, 2)
                t3 = np.random.normal(mu, var, count)

                trans = np.asarray([t1, t2, t3])

                mu, var = pose_copy[3], (np.pi / 2.) * pow(1-score, 2)
                q0 = np.random.normal(mu, var, count)
                mu, var = pose_copy[4], (np.pi / 2.) * pow(1-score, 2)
                q1 = np.random.normal(mu, var, count)
                mu, var = pose_copy[5], (np.pi / 2.) * pow(1-score, 2)
                q2 = np.random.normal(mu, var, count)
                mu, var = pose_copy[6], (np.pi / 2.) * pow(1-score, 2)
                q3 = np.random.normal(mu, var, count)

                quat = np.asarray([q0, q1, q2, q3])

                for q in quat:
                    for i in range(len(q)):
                        if q[i] < -1:
                            q[i] = np.trunc(q[i])-1 - q[i]
                        if q[i] > 1:
                            q[i] = np.trunc(q[i])+1 - q[i]

                pose = np.vstack((trans, quat)).T
                poses.extend(pose)

                pose_dist_copy = [prev_pose_distribution[index] for i in range(count)]
                pose_distribution.extend(pose_dist_copy)

        pose = final_pose

        pose_matrix = quaternion_matrix(pose[3:])
        pose_matrix[:3,3] = pose[:3]
        detected_pose = pose_matrix.flatten()
        return best_score, pose, detected_pose

        # return best_score, pose, pose_rot, pose[:3]
