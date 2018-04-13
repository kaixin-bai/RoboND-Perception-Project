#!/usr/bin/env python
"""
PR2 Perception.

To Run:
    roslaunch pr2_robot pr2_robot pick_place_project.launch
    rosrun pr2_robot pr2_perception.py
"""

import os
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

import pickle

import rospy
import rospkg

from sensor_stick.srv import GetNormals
from pr2_robot.features import cloud2feature
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

from pr2_robot import seg_utils
from pr2_robot.svm_classifier import SVMClassifier
from pr2_robot.srv import SetTarget, SetTargetResponse, SetTargetRequest
import tf

class PR2Perception(object):
    def __init__(self):
        rospy.init_node('pr2_perception')

        rospack = rospkg.RosPack()
        pkg_root = rospack.get_path('pr2_robot')
        fname = os.path.join(pkg_root, 'config', 'model.sav') 

        self._model_path = rospy.get_param('~model_path', default=fname)
        self._pcl_topic = rospy.get_param('~pcl_topic', default='/pr2/world/points')
        self._seg_topic = rospy.get_param('~seg_topic', default='~segmented_objects')
        self._tf = tf.TransformListener()

        self._cloud_b = np.empty(shape=(0,4))

        self._clf = SVMClassifier(model_path=self._model_path)

        self._n_srv = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)

        self._target = '' # do not exclude anything from map cloud
        self._target_srv = rospy.Service('~set_target', SetTarget, self.target_cb)
        self._map_pub = rospy.Publisher('/pr2/3d_map/points', PointCloud2, queue_size=10)

        self._do_pub = rospy.Publisher('~detected_objects', DetectedObjectsArray, queue_size=10)
        self._mk_pub = rospy.Publisher('~label_markers', Marker, queue_size=10)
        self._pcl_sub = rospy.Subscriber(self._pcl_topic,
                pc2.PointCloud2, self.pcl_cb, queue_size=1)
        self._seg_pub = rospy.Publisher(self._seg_topic,
                pc2.PointCloud2, queue_size=1)

    def target_cb(self, req):
        self._target = req.target.data
        return SetTargetResponse(success=True)

    def pcl_cb(self, msg):
        try:
            cloud = ros_to_pcl(msg)
            cloud_os, cloud_o, cloud_t, cloud_b1, cloud_b2 = self.segment(cloud)

            # bins ...
            cloud_b = []
            for c in [self._cloud_b, cloud_b1, cloud_b2]:
                if np.ndim(c) == 2:
                    cloud_b.append(c)
            cloud_b = np.concatenate(cloud_b, axis=0)
            self._cloud_b = pcl.PointCloud_PointXYZRGB()
            self._cloud_b.from_list(cloud_b)
            self._cloud_b = seg_utils.downsample(self._cloud_b, leaf=0.01)

            labels, objects = self.classify(cloud_os)
            self.publish(labels, objects, cloud_o, cloud_os, cloud_t)
        except Exception as e:
            rospy.logerr_throttle(2.0, "Invalid or Empty Cloud : {}".format(e))

    @staticmethod
    def segment(cloud):
        cloud = seg_utils.downsample(cloud, leaf=0.01)
        cloud = seg_utils.passthrough(cloud, ax='y', axmin=-2.0, axmax=2.0)
        cloud = seg_utils.passthrough(cloud, ax='z', axmin=0.6, axmax=3.0)
        cloud = seg_utils.denoise(cloud, k=50, x=1e-1)

        cloud_b1 = seg_utils.passthrough(cloud, ax='y', axmin=0.5, axmax=2.0)
        cloud_b2 = seg_utils.passthrough(cloud, ax='y', axmin=-2.0, axmax=-0.5)

        cloud_t, cloud_o = seg_utils.ransac(cloud, dmax=0.02)
        cloud_os = seg_utils.cluster(cloud_o, as_list=True)
        cloud_o = seg_utils.cluster(cloud_o, as_list=False)
        return cloud_os, cloud_o, cloud_t, cloud_b1, cloud_b2

    def classify(self, cloud_os):
        # prepare feature
        features = []
        for cloud in cloud_os:
            cloud_ros = pcl_to_ros(cloud)
            normals = self._n_srv(cloud_ros).cluster
            feature = cloud2feature(cloud_ros, normals,
                    self._clf._hsv, bins=self._clf._bin
                    )
            features.append(feature)
        features = np.stack(features, axis=0)
        classes = self._clf.predict(features)

        pos     = np.float32([np.mean(cloud, axis=0) for cloud in cloud_os])
        pos[:,2] += 0.4

        classes = classes.tolist()
        pos = pos.tolist()

        labels = [make_label(c,p,i) for i, (c,p) in enumerate(zip(classes, pos))]
        objects = [DetectedObject(label=l, cloud=pcl_to_ros(c)) for (l,c) in 
            zip(classes, cloud_os)]
        return labels, objects

    def publish(self, labels, objects, cloud_o, cloud_os, cloud_t):
        # construct map cloud ...
        if self._target:
            map_cloud_list = [self._cloud_b, cloud_t]
            map_cloud_list += [c for (l,c) in zip(labels, cloud_os) if l.text != self._target]
            map_cloud_list = np.concatenate(map_cloud_list, axis=0)
            #t, q = self._tf.lookupTransform('camera_link', 'world', rospy.Time(0))
            #T = np.reshape(t, (1,3))
            #R = tf.transformations.quaternion_matrix(q)[:3,:3]

            ## collect cloud ...
            #map_cloud_list = [np.asarray(cloud_t)]
            #i0,i1 = len(map_cloud_list[0]),0
            #for l, c in zip(labels, cloud_os):
            #    c = np.asarray(c)
            #    map_cloud_list.append(c)
            #    if (l.text != self._target):
            #        if i1 == 0:
            #            i0 += len(c)
            #    else:
            #        i1 = i0 + len(c)
            #map_cloud_list = np.concatenate(map_cloud_list, axis=0)

            ## apply transformations ...
            ## this is important in order for the octomap to clear properly
            ## since it uses raycasting internally.
            #map_cloud_list[:,:3] = map_cloud_list[:,:3].dot(R.T) # == R.dot(x.T).T
            #map_cloud_list[:,:3] += T

            ## "clear" objects ...
            #if i1 > i0:
            #    clear_len = 1.5 # TODO - kind of arbitrary; enough clearance to clear target from octomap
            #    lens = np.linalg.norm(map_cloud_list[i0:i1,:3], axis=1, keepdims=True)
            #    map_cloud_list[i0:i1,:3] *= (clear_len/lens)

            # (N,4)
            map_cloud = pcl.PointCloud_PointXYZRGB()
            map_cloud.from_list(map_cloud_list)
            map_cloud = pcl_to_ros(map_cloud)
            #map_cloud.header.frame_id = 'camera_link'
            self._map_pub.publish(map_cloud)

        for l in labels:
            self._mk_pub.publish(l)
        self._do_pub.publish(objects)
        self._seg_pub.publish(pcl_to_ros(cloud_o))

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()

def main():
    get_color_list.color_list = []
    app = PR2Perception()
    app.run()

if __name__ == '__main__':
    main()
