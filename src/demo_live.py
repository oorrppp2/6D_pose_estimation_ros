#!/home/robot/anaconda3/envs/pytorch_env/bin/python
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
import rospy
from vision_msgs.msg import SegResult
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Empty, String, Bool, Header, Float64, Int8

from living_lab_robot_perception.msg import PoseEstimationAction, PoseEstimationResult
# from tf.transformations import *
import actionlib
from utils import quaternion_from_matrix


class PoseEstimator:
    def __init__(self):
        visualization = rospy.get_param("visualization", default=True)
        gaussian_std = rospy.get_param("gaussian_std", default=0.1)
        max_iteration = rospy.get_param("max_iteration", default=20)
        tau = rospy.get_param("tau", default=0.1)
        num_particles = rospy.get_param("num_particles", default=180)

        self.camera_param_msg = rospy.wait_for_message("/camera/aligned_depth_to_color/camera_info", CameraInfo)
        self.camera_param = self.camera_param_msg.K

        self.detection_done = True
        self.detected_pose = np.zeros(16)       # Flattened matrix[4,4]
        self.target_id = -1

        self.seg_sub = rospy.Subscriber('/segmentation_result', SegResult, self.Seg_result_callback)
        self.target_id_pub = rospy.Publisher("/target_id", Int8, queue_size=1)

        self.action_pose_estimation = actionlib.SimpleActionServer('/pose_estimation', PoseEstimationAction, execute_cb=self.pose_estimation_cb, auto_start = False)
        self.action_pose_estimation.start()

        self.pf = particle_filter.ParticleFilter(self.camera_param[0], self.camera_param[4],
        self.camera_param[2], self.camera_param[5], visualization=visualization,
        gaussian_std=gaussian_std, max_iteration=max_iteration, tau=tau, num_particles=num_particles)

        rospy.loginfo('%s ready...'%rospy.get_name())

    def pose_estimation_cb(self, goal):
        print("Goal id : ", goal.target_id)
        self.target_id_pub.publish(goal.target_id)
        self.target_id = goal.target_id
        result = PoseEstimationResult()
        self.detection_done = False

        while(not self.detection_done):
            rospy.sleep(0.1)

        result.estimated_pose = self.detected_pose.tolist()
        print(result.estimated_pose)
        result.result = True
        self.action_pose_estimation.set_succeeded(result)


    def Seg_result_callback(self, seg_message):
        if self.detection_done != True:
            if self.target_id == seg_message.frame_id:
                frame_id = seg_message.frame_id
                rois = seg_message.roi
                label_msg = seg_message.target_region
                other_region_msg = seg_message.other_region
                img_msg = seg_message.color
                depth_msg = seg_message.depth
                rois = seg_message.roi

                img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1).copy()
                depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width, -1).copy() * 0.001
                label = np.frombuffer(label_msg.data, dtype=np.uint8).reshape(label_msg.height, label_msg.width, -1).copy()
                other_region = np.frombuffer(other_region_msg.data, dtype=np.uint8).reshape(other_region_msg.height, other_region_msg.width, -1).copy()

                pose_estimation_time = time.time()
                # best_score, pose, estimated_rot, estimated_trans = self.pf.start(frame_id+1, img, depth, label, other_region, rois=rois)
                best_score, pose, self.detected_pose = self.pf.start(frame_id, img, depth, label, other_region, rois=rois)
                rospy.loginfo('Pose estimation time: {0}sec'.format(time.time() - pose_estimation_time))

                self.detection_done = True
            else:
                return

        else:
            return


if __name__ == '__main__':

    rospy.init_node('pose_estimation', anonymous=False)
    pe = PoseEstimator()

    rospy.spin()
