#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

class PR2PickPlaceClient(object):
    """
    Simple PR2 Pick-and-Place Client, composed of three stages:
    1. Init:
        Wait for a specified duration, for stability.
        The duration is specified as `wait_init`
    2. Map:
        Build collision map for the robot.
        The duration of each maneuver is specified as `wait_turn`
    3. Data:
        Collect Pose-Model association data.
        if `static` option is set, this stage will be skipped,
        and the data will be loaded from the supplied `yaml_file` parameter.
    4. Move:
        While objects continue to be detected, the client will request
        the `pick_place_routine` to move objects from the table to the bins.
    5. Done:
        After all the objects have been placed properly, the client will
        save the collected data if `save_yaml` option is set, and terminate the program.
    """
    def __init__(self):
        """ initialize the client + load parameters """
        rospy.init_node('pr2_pick_place_client')

        self._wait_init = rospy.get_param('~wait_init', default=10.0) # wait 10 sec. at startup
        self._wait_turn = rospy.get_param('~wait_turn', default=10.0) # wait 10 sec. at startup
        self._wait_data = rospy.get_param('~wait_data', default=10.0) # wait 10 sec. at startup

        # TODO : grab bin pose data

        self._yaml_file = rospy.get_param('~yaml_file', default='/tmp/target.yaml')
        self._static = rospy.get_param('~static', default=False)
        self._save_yaml = rospy.get_param('~save_yaml', default=False)

        self._object_list = rospy.get_param('~object_list', default=[])
        self._object_names = [o['name'] for o in self._object_list]

        self._det_sub = rospy.Subscriber('/pr2_perception/detected_objects',
                DetectedObjectsArray, self.det_cb, queue_size=1)
        self._move_srv = rospy.ServiceProxy('pick_place_routine', PickPlace)

        self._data = []
        self._rate = rospy.Rate(50)

        self._state = 'wait'
        if self._static:
            # TODO(@yoonyoungcho) : load from yaml here
            pass

    def det_cb(self, msg):
        """ Process and Save data published from pr2_perception() """
        objects = msg.objects
        now = rospy.Time.now().to_sec()
        if self._state == 'data':
            # collect data
            for o in objects:
                if o.label not in self._object_names:
                    continue
                pos = np.mean(ros_to_pcl(o.cloud), axis=0)
                name = o.label
                self._data.append( (pos, name) )

    def wait_for(self, duration):
        """ Wait for duration """
        start = rospy.Time.now().to_sec()
        while True:
            now = rospy.Time.now().to_sec()
            if (now - start) > duration:
                break
            self._rate.sleep()

    def turn_to(self, duration, angle):
        """ Turn to angle over duration """
        # TODO : compute duration from speed?
        self._j_pub.publish(angle)
        wait_for(duration)

    def move(self, object):
        """ Pick-and-place object via `pick_place_routine` """
        # TODO : implement
        rospy.wait_for_service('pick_place_routine')
        # which_arm \element {right, left}
        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)
            print ("Response: ",resp.success)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def save(self):
        """ Save object location parameters as a YAML file """
        # TODO : implement

    def log(self, message):
        """ Log current state """
        rospy.loginfo(message)

    def run(self):
        """ Run PR2 Pick-and-place client """
        # for stability, wait for specified duration
        self._state = 'init'
        self.log('State : {}'.format(self._state))
        wait_for(self._wait_init)

        # initial scouting to build map
        self._state = 'map'
        self.log('State : {}'.format(self._state))
        self.log()
        turn_to(self._wait_turn, -np.pi/2)
        turn_to(self._wait_turn, np.pi/2)
        turn_to(self._wait_turn, 0.0)

        # collect data over a window, skipped if `static` option is set
        # (i.e. loading from YAML)
        self._state = 'data'
        self.log('State : {}'.format(self._state))
        if not self._static:
            wait_for(self._wait_data)
            # 1. cluster_by_location()
            # pseudocode : 
            # self._data = KMeans(data.pos, len(self._object_names))
            # 2. count_names_per_clusters()
            # 3. assign_name_to_location()
            # (by count; row : cluster, column : object name)
            # self._data = linear_sum_assignment( ... )

        # perform pick-and-place
        # technically this would overlook the object
        # which was hidden behind others, but it's okay for now
        self._state = 'move'
        self.log('State : {}'.format(self._state))
        for object in self._object_names:
            self.move(object)

        # done!
        self._state = 'done'
        self.log('State : {}'.format(self._state))
        if self._save_yaml:
            self.save()

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    
    # TODO: Statistical Outlier Filtering

    # TODO: Voxel Grid Downsampling

    # TODO: PassThrough Filter

    # TODO: RANSAC Plane Segmentation

    # TODO: Extract inliers and outliers

    # TODO: Euclidean Clustering

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately

    # TODO: Convert PCL data to ROS messages

    # TODO: Publish ROS messages

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)

        # Grab the points for the cluster

        # Compute the associated feature vector

        # Make the prediction

        # Publish a label into RViz

        # Add the detected object to the list of detected objects.

    # Publish the list of detected objects

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects_list)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables

    # TODO: Get/Read parameters

    # TODO: Parse parameters into individual variables

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list

        # TODO: Get the PointCloud for a given object and obtain it's centroid

        # TODO: Create 'place_pose' for the object

        # TODO: Assign the arm to be used for pick_place

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file



if __name__ == '__main__':

    # TODO: ROS node initialization

    # TODO: Create Subscribers

    # TODO: Create Publishers

    # TODO: Load Model From disk

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
