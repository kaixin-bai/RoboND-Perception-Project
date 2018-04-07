#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import make_label

class GroundTruthMarker(object):
    def __init__(self):
        rospy.init_node('ground_truth_marker')
        self._gt_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.gt_cb, queue_size=1)
        self._gt_pub = rospy.Publisher('~label_markers', Marker, queue_size=10)

        self._models= rospy.get_param('~object_list', default=[])
        self._models = [m['name'] for m in self._models]

    def gt_cb(self, msg):
        labels = []
        for i, (n,p) in enumerate(zip(msg.name, msg.pose)):
            if not (n in self._models):
                continue
            pos = [p.position.x, p.position.y, p.position.z + 0.3]
            label = make_label(n, pos, i, color=[0.0,1.0,0.0])
            labels.append(label)

        for l in labels:
            self._gt_pub.publish(l)

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()

def main():
    app = GroundTruthMarker()
    app.run()

if __name__ == "__main__":
    main()
