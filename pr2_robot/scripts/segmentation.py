#!/usr/bin/env python

# Import modules
from pcl_helper import *
from pr2_robot import seg_utils

# Initialize color_list
get_color_list.color_list = []

class SegmenterROS(object):
    def __init__(self):
        rospy.init_node('segmenter', anonymous=True)
        self._pcl_sub = rospy.Subscriber('/sensor_stick/point_cloud',
                pc2.PointCloud2, self.pcl_cb, queue_size=1)
        self._obj_pub = rospy.Publisher('/pcl_objects',
                PointCloud2, queue_size=1)
        self._tbl_pub = rospy.Publisher('pcl_table',
                PointCloud2, queue_size=1)

    def pcl_cb(self, msg):
        cloud = ros_to_pcl(msg)
        cloud = seg_utils.downsample(cloud)
        cloud = seg_utils.passthrough(cloud)
        cloud_t, cloud_o = seg_utils.ransac(cloud)
        cloud_o = seg_utils.cluster(cloud_o)
        self._tbl_pub.publish(pcl_to_ros(cloud_t))
        self._obj_pub.publish(pcl_to_ros(cloud_o))

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()

def main():
    app = SegmenterROS()
    app.run()

if __name__ == '__main__':
    main()
