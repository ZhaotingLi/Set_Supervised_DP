#!/usr/bin/python3

"""
Authors:
    Rodrigo Perez-Dattari <r.j.perezdattari@tudelft.nl>
    Willem Momma <w.j.momma@student.tudelft.nl>
"""
import logging

logger = logging.getLogger(__name__)


from sensor_msgs.msg import JointState
from iiwa_robotics_toolbox import iiwa
from spatialmath import UnitQuaternion
from geometry_msgs.msg import PoseStamped, Pose
import numpy as np
import rospy
import sys
import json

ROBOT = sys.argv[1]
## This code is not written yet. To be fixed in the future
class CartesianStatePrinter:
    def __init__(self):
        self.q = None
        self.q_dot = None
        self.effort = None  # Add this line to store the joint efforts
        self.robot = iiwa(model=ROBOT)
        rospy.Subscriber('/%s/joint_states' % ROBOT, JointState, self._callback_joint_states, queue_size=10)
        self.pose = None
        rospy.Subscriber('/vrpn_client_node/probe/pose', PoseStamped, self._callback_pose, queue_size=10)

    
    def _callback_pose(self, data):
        self.pose = data

    def _callback_joint_states(self, data):
        self.q = np.array(data.position)
        self.q_dot = np.array(data.velocity)
        self.effort = np.array(data.effort)  

    def _get_pose(self, end_link='iiwa_link_tool'):
        pose = self.robot.fkine(self.q, end=end_link, start='iiwa_link_0')
        position = pose.t
        orientation = pose.R
        return position, orientation

    def _get_cartesian_state(self, end_link='iiwa_link_tool'):
        # Get pose
        position_end_effector, orientation_end_effector = self._get_pose(end_link=end_link)

        # Transform orientation matrix to quaternion
        orientation_quaternion = UnitQuaternion(orientation_end_effector)

        return position_end_effector, orientation_quaternion

    def _print_pose(self):
        if self.pose is None:
            logger.debug('No Optitrack data received yet.')
            return [0, 0, 0], [0, 0, 0, 0]  # Return default position and orientation

        position = self.pose.pose.position
        orientation = self.pose.pose.orientation
        logger.debug('%s %s', 'Optitrack Position:', position)
        logger.debug('%s %s', 'Optitrack Orientation:', orientation)
        logger.debug('\n')
        return [position.x, position.y, position.z], [orientation.x, orientation.y, orientation.z, orientation.w] 

    def _print_cartesian_state(self, end_link):
        position, orientation = self._get_cartesian_state(end_link=end_link)
        logger.debug('%s %s', 'Position %s:' % end_link, position)
        logger.debug('%s %s', 'Orientation unit quaternion %s:' % end_link, orientation)
        logger.debug('%s %s', 'Orientation RPY %s:' % end_link, orientation.rpy())
        logger.debug('%s %s', 'Joint positions:', self.q.tolist())  # Add this line to print the joint positions
        logger.debug('%s %s', 'Joint efforts:', self.effort.tolist())  # Add this line to print the joint efforts
        logger.debug('\n')
        # Get and print the optitrack pose
        optitrack_position, optitrack_orientation = self._print_pose()
        return position.tolist(), str(orientation), orientation.rpy().tolist(), self.q.tolist(), self.effort.tolist(), optitrack_position, optitrack_orientation

    # Function to adjust Optitrack coordinates to match the robot's coordinate system
    def adjust_optitrack_coordinates(self, optitrack_position):
        # Swap X and Y (keeping the sign)
        adjusted_x = -optitrack_position[1]
        adjusted_y = optitrack_position[0]
        adjusted_z = optitrack_position[2]
        return [adjusted_x, adjusted_y, adjusted_z]

    def run(self):
        rospy.sleep(0.1)
        
        file_name = input("Enter the file name (without extension): ")
        positions = {}

        while True:
            # Ask the user for the pose name
            pose_name = input("Enter the pose name (or 'quit' to stop): ")
            if pose_name.lower() == 'quit':
                break

            # Get and print the end effector's state and the optitrack pose
            position_ee, orientation_str_ee, orientation_rpy_ee, joint_positions, joint_efforts, optitrack_position, optitrack_orientation = self._print_cartesian_state(end_link='iiwa_link_ee')
            position_tool, orientation_str_tool, orientation_rpy_tool, _, _, _, _ = self._print_cartesian_state(end_link='iiwa_link_tool')

            # Adjust the Optitrack coordinates to match the robot's coordinate system
            adjusted_optitrack_position = self.adjust_optitrack_coordinates(optitrack_position)

            positions[pose_name] = {
                'kuka_ee_position': position_ee,
                'kuka_ee_orientation_str': orientation_str_ee,
                'kuka_ee_orientation_rpy': orientation_rpy_ee,
                'kuka_tool_position': position_tool,
                'kuka_joint_values': joint_positions,
                'kuka_joint_torques': joint_efforts,
                'optitrack_position': adjusted_optitrack_position,
                'optitrack_orientation': optitrack_orientation
            }

        # Write the positions to a JSON file in the 'data' directory
        with open(f'data/point_recordings/{file_name}.json', 'w') as f:
            json.dump(positions, f, indent=4)

if __name__ == '__main__':
    # Init ROS
    rospy.init_node('print_cartesian_state')
    control_server = CartesianStatePrinter()
    control_server.run()