#!/usr/bin/env python
"""
Performs data collection.

To Run:
    roslaunch sensor_stick training.launch
    roslaunch pr2_robot data_collection.launch
"""
import numpy as np
import pickle
import rospy
import os

from sensor_stick.pcl_helper import *
from sensor_stick.training_helper import spawn_model
from sensor_stick.training_helper import delete_model
from sensor_stick.training_helper import initial_setup
from sensor_stick.training_helper import capture_sample
from pr2_robot.features import cloud2feature
from sensor_stick.srv import GetNormals
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

class Collector(object):
    """
    Simple data collection class that interfaces with sensor_stick/training.launch.
    Produces color/normal histogram features from point clouds.
    """
    def __init__(self):
        """
        Initializes ROS Node 'collector'
        and loads parameters from the parameter server.
        """
        rospy.init_node('collector')

        # see pr2_robot/config/pick_list_3.yaml for format reference.
        self._models = rospy.get_param('~object_list', default=[])
        self._models = [m['name'] for m in self._models]

        # data collection parameters
        self._as_feature = rospy.get_param('~as_feature', default=False)
        self._as_hsv = rospy.get_param('~as_hsv', default=False)
        self._steps = rospy.get_param('~steps', default=64) # steps per model
        self._max_try = rospy.get_param('~max_try', default=8)
        self._nbins = rospy.get_param('~nbins', default=16)

        # encode parameter information in default path
        s_feat  = ('feat' if self._as_feature else 'raw')
        s_color = ('hsv' if self._as_hsv else 'rgb')
        default_path = '/tmp/training_set_{}_{}_{}_{}x{}.sav'.format(
                s_feat, s_color, self._nbins,
                self._steps, self._max_try)

        self._path = rospy.get_param('~path', default=default_path)

    def run(self):
        """
        Produces training_set.sav
        Formatted as [param, data], where:
        - param : {'hsv':bool, 'bin':int}
            hsv : hsv colorspace flag (otherwise rgb)
            bin : number of histogram bins for feature extraction
        - data : [{name : [features]} for name in models]
        """
        initial_setup()
        # save collection parameters
        param = {
                'hsv' : self._as_hsv,
                'bin' : self._nbins
                }
        data = {}
        m_n = len(self._models)
        for m_i, model_name in enumerate(self._models):
            rospy.loginfo('[{}/{}] Processing Model Name : {}'.format(m_i, m_n, model_name))
            model_data = []
            spawn_model(model_name)

            for i in range(self._steps):
                # get_cloud()
                sample_cloud = None
                for j in range(self._max_try):
                    sample_cloud = capture_sample()
                    sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()
                    if sample_cloud_arr.shape[0] == 0:
                        rospy.loginfo('')
                        print('Invalid cloud detected')
                    else:
                        break
                # save_data()
                if sample_cloud is not None:
                    if self._as_feature:
                        # Extract histogram features
                        normals = get_normals(sample_cloud)
                        feature = cloud2feature(sample_cloud, normals,
                                hsv=self._as_hsv, bins=self._nbins)
                        model_data.append(feature)
                    else:
                        model_data.append(sample_cloud)

            data[model_name] = model_data
            delete_model()

        # save data with pickle
        with open(self._path, 'wb') as f:
            pickle.dump([param, data], f)

def main():
    collector = Collector()
    collector.run()

if __name__ == "__main__":
    main()
