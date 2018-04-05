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

class PR2Perception(object):
    def __init__(self):
        rospy.init_node('pr2_perception')

        rospack = rospkg.RosPack()
        pkg_root = rospack.get_path('pr2_robot')
        fname = os.path.join(pkg_root, 'config', 'model.sav') 

        self._model_path = rospy.get_param('~model_path', default=fname)
        self._pcl_topic = rospy.get_param('~pcl_topic', default='/pr2/world/points')
        self._seg_topic = rospy.get_param('~seg_topic', default='~segmented_objects')

        self._clf = SVMClassifier(model_path=self._model_path)

        self._n_srv = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
        self._do_pub = rospy.Publisher('~detected_objects', DetectedObjectsArray, queue_size=10)
        self._mk_pub = rospy.Publisher('~label_markers', Marker, queue_size=10)
        self._pcl_sub = rospy.Subscriber(self._pcl_topic,
                pc2.PointCloud2, self.pcl_cb, queue_size=1)
        self._seg_pub = rospy.Publisher(self._seg_topic,
                pc2.PointCloud2, queue_size=1)

    def pcl_cb(self, msg):
        cloud = ros_to_pcl(msg)
        cloud_os, cloud_o = self.segment(cloud)
        labels, objects = self.classify(cloud_os)
        self.publish(labels, object, cloud_o)

    @staticmethod
    def segment(cloud):
        cloud = seg_utils.downsample(cloud, leaf=0.01)
        cloud = seg_utils.passthrough(cloud, ax='y', axmin=-0.5, axmax=0.5)
        cloud = seg_utils.passthrough(cloud, axmin=0.6, axmax=1.1)
        cloud = seg_utils.denoise(cloud, k=50, x=0.01)
        cloud_t, cloud_o = seg_utils.ransac(cloud, dmax=0.02)
        cloud_os = seg_utils.cluster(cloud_o, as_list=True)
        cloud_o = seg_utils.cluster(cloud_o, as_list=False)
        return cloud_os, cloud_o

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

    def publish(self, labels, objects, cloud_o):
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
