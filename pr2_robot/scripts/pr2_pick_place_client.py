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
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

class PR2PickPlaceClient(object):
    """
    Simple PR2 Pick-and-Place Client, composed of five stages:
    1. Init:
        Wait for a specified duration, for stability.
        The duration is specified as `wait_init`
    2. Map:
        Build collision map for the robot.
        (Deprecated) The duration of each maneuver is specified as `wait_turn`
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

        self._wait_init = rospy.get_param('~wait_init', default=0.0) # wait 10 sec. at startup
        self._wait_turn = rospy.get_param('~wait_turn', default=5.0) # wait 10 sec. at startup
        self._wait_data = rospy.get_param('~wait_data', default=5.0) # wait 10 sec. at startup

        self._yaml_file = rospy.get_param('~yaml_file', default='/tmp/target.yaml')
        self._static = rospy.get_param('~static', default=False)
        self._save_yaml = rospy.get_param('~save_yaml', default=False)
        self._rate = rospy.Rate(rospy.get_param('~rate', default=50))

        # WARNING : fetch object_list + dropbox from global parameter server
        self._object_list = rospy.get_param('/object_list', default=[])
        self._object_names = [o['name'] for o in self._object_list]
        self._dropbox = rospy.get_param('/dropbox', default=[]) # [group, name, position]
        self._scene = rospy.get_param('~scene', default=1)

        self._det_sub = rospy.Subscriber('/pr2_perception/detected_objects',
                DetectedObjectsArray, self.det_cb, queue_size=1)

        self._j = None
        self._j_pub = rospy.Publisher('/pr2/world_joint_controller/command', Float64, queue_size=1)
        self._j_sub = rospy.Subscriber('/pr2/joint_states', JointState, self.joint_cb, queue_size=1)

        #self._move_srv = rospy.ServiceProxy('pick_place_routine', PickPlace)

        self._data = []
        self._yaml_data = []

        self._state = 'wait'
        if self._static:
            # TODO(@yoonyoungcho) : load from yaml here
            pass

    def joint_cb(self, msg):
        """ Get + Save World Joint Feedback Info for turn_to() """
        # only care about world joint
        jname = 'world_joint'
        if jname in msg.name:
            idx = msg.name.index(jname)
            self._j = msg.position[idx]

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

    def turn_to(self, angle, tol=np.deg2rad(1.0)):
        """ Turn to angle over duration """
        while True:
            if self._j is not None:
                if np.abs(angle - self._j) < tol:
                    break
            self._j_pub.publish(angle)
            self._rate.sleep()
        #self.wait_for(duration)

    def move(self, object):
        """ Pick-and-place object via `pick_place_routine` """
        rospy.loginfo("Waiiting for pick_place_routine service ... ")
        # TODO : support service timeout
        rospy.wait_for_service('pick_place_routine')
        # which_arm \element {right, left}
        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            test_scene_num = Int32(self._scene)

            # 1 : search for object in object_list
            for o in self._object_list:
                if o['name'] == object:
                    group = o['group']
                    break
            else:
                rospy.logerr_throttle(1.0, "Object {} not found in object_list".format(object))
                return False

            # 2 : search for group in dropbox
            for box in self._dropbox:
                if box['group'] == group:
                    which_arm = box['name']
                    place_pose = box['position']
                    break
            else:
                rospy.logerr_throttle(1.0, "Group {} not found in dropbox".format(group))
                return False

            # 3 : search for pick_pose
            ## TODO : improve heuristic
            pick_poses = [d[0] for d in self._data if(d[1]==object)]
            pick_pose = np.mean(pick_poses, axis=0)

            if np.any(np.isnan(pick_pose)):
                rospy.loginfo_throttle(1.0, "Invalid Pick Pose for {}".format(object))
                return False

            rospy.loginfo_throttle(1.0, "[{}] Pick Pose : {}".format(object, pick_pose))

            # format ...
            pick_pose_msg = Pose()
            pick_pose_msg.position.x = float(pick_pose[0])
            pick_pose_msg.position.y = float(pick_pose[1])
            pick_pose_msg.position.z = float(pick_pose[2])
            pick_pose_msg.orientation.w = 1.0

            place_pose_msg = Pose()
            place_pose_msg.position.x = float(place_pose[0]) - 0.1 # more interior
            place_pose_msg.position.y = float(place_pose[1])
            place_pose_msg.position.z = float(place_pose[2])
            place_pose_msg.orientation.w = 1.0

            object_msg = String(object)
            which_arm_msg = String(which_arm)

            resp = pick_place_routine(test_scene_num, object_msg, which_arm_msg, pick_pose_msg, place_pose_msg)
            yaml_dict = make_yaml_dict(test_scene_num, object_msg, which_arm_msg, pick_pose_msg, place_pose_msg)
            self._yaml_data.append(yaml_dict)

            print ("Response: ",resp.success)
            return resp.success
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
            return False

        # TODO : other failure cases?
        return True

    def save(self):
        rospy.loginfo('yaml data : {}'.format(self._yaml_data))
        send_to_yaml(self._yaml_file, self._yaml_data)
        """ Save object location parameters as a YAML file """

    def log(self, message):
        """ Log current state """
        rospy.loginfo(message)

    def run(self):
        """ Run PR2 Pick-and-place client """
        # for stability, wait for specified duration
        self._state = 'init'
        self.log('State : {}'.format(self._state))
        self.wait_for(self._wait_init)

        # initial scouting to build map
        self._state = 'map'
        self.log('State : {}'.format(self._state))

        self.turn_to(0.0)
        self.turn_to(-np.pi/2)
        self.turn_to(np.pi/2)
        self.turn_to(0.0)

        ## collect data over a window, skipped if `static` option is set
        ## (i.e. loading from YAML)
        self._state = 'data'
        self.log('State : {}'.format(self._state))
        if not self._static:
            self.wait_for(self._wait_data)
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
            self._data = [] # clear data, in case stuff got knocked over
            self._state = 'data' # temporarily collect data
            self.wait_for(self._wait_data)
            self._state = 'move'

            suc = self.move(object)
            rospy.loginfo_throttle(1.0, "[{}] Move Success : {}".format(object, suc))

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

def main():
    # Initialize color_list
    get_color_list.color_list = []
    client = PR2PickPlaceClient()
    client.run()
    
if __name__ == '__main__':
    main()
