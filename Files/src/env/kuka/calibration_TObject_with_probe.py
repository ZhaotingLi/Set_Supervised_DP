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
import os

# ROBOT = sys.argv[1]

class CartesianStatePrinter:
    def __init__(self):
        self.q = None
        self.q_dot = None
        self.effort = None  # Add this line to store the joint efforts
        # self.robot = iiwa(model=ROBOT)
        # rospy.Subscriber('/%s/joint_states' % ROBOT, JointState, self._callback_joint_states, queue_size=10)
        self.pose = None
        
        rospy.Subscriber('/vrpn_client_node/MeasurementProbe/pose', PoseStamped, self._callback_pose, queue_size=10)
        self.Tobject_pose = None
        # rospy.Subscriber('/vrpn_client_node/TObject/pose', PoseStamped, self._callback_TObject_pose, queue_size=10)

        rospy.Subscriber('/vrpn_client_node/UObject/pose', PoseStamped, self._callback_TObject_pose, queue_size=10)

    
    def _callback_pose(self, data):
        self.pose = data

    def _callback_TObject_pose(self, data):
        self.Tobject_pose = data


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

    def _print_Tobject_pose(self):
        if self.Tobject_pose is None:
            logger.debug('No Tobject_pose data received yet.')
            return [0, 0, 0], [0, 0, 0, 0]  # Return default position and orientation

        position = self.Tobject_pose.pose.position
        orientation = self.Tobject_pose.pose.orientation
        # print('Tobject_pose Position:', position)
        # print('Tobject_pose Orientation:', orientation)
        # print('\n')
        return [position.x, position.y, position.z], [orientation.x, orientation.y, orientation.z, orientation.w] 


    # def _print_cartesian_state(self, end_link):
    #     position, orientation = self._get_cartesian_state(end_link=end_link)
    #     print('Position %s:' % end_link, position)
    #     print('Orientation unit quaternion %s:' % end_link, orientation)
    #     print('Orientation RPY %s:' % end_link, orientation.rpy())
    #     print('Joint positions:', self.q.tolist())  # Add this line to print the joint positions
    #     print('Joint efforts:', self.effort.tolist())  # Add this line to print the joint efforts
    #     print('\n')
    #     # Get and print the optitrack pose
    #     optitrack_position, optitrack_orientation = self._print_pose()
    #     return position.tolist(), str(orientation), orientation.rpy().tolist(), self.q.tolist(), self.effort.tolist(), optitrack_position, optitrack_orientation

    # Function to adjust Optitrack coordinates to match the robot's coordinate system
    def adjust_optitrack_coordinates(self, optitrack_position):
        # Swap X and Y (keeping the sign)
        adjusted_x = -optitrack_position[1]
        adjusted_y = optitrack_position[0]
        adjusted_z = optitrack_position[2]
        return [adjusted_x, adjusted_y, adjusted_z]

    def run(self):
        rospy.sleep(0.1)
        
        # Ask for file name to save the results
        file_name = input("Enter the file name (without extension): ")
        position_all = {}

        while True:
            # Ask for a main pose name to store a set of object points
            pose_name = input("Enter the pose name (or 'quit' to stop): ")
            if pose_name.lower() == 'quit':
                break

            # Capture Optitrack data for the main pose
            Object_local2_p_w_to_L2, Object_local2_R_w_to_L2 = self._print_Tobject_pose()

            # T-shape
            positions_local1 ={
                'Object_local1_p_w_to_o': [0,0,0],
                'Object_local1_p_w_1': [-0.015,0.13,0],   # meter
                'Object_local1_p_w_2': [0.015,0.13,0],
                'Object_local1_p_w_3': [-0.015,0.03,0],
                'Object_local1_p_w_4': [0.015,0.03,0],
                'Object_local1_p_w_5': [0.055,0.03,0],
                'Object_local1_p_w_6': [0.055,0.0,0],
                'Object_local1_p_w_7': [-0.055,0.03,0],
                'Object_local1_p_w_8': [-0.055,0.0,0],
            }
            # Dictionary to hold positions for each pose, with initial values
            positions = {
                'Object_local2_p_w_to_L2': Object_local2_p_w_to_L2,
                'Object_local2_R_w_to_L2': Object_local2_R_w_to_L2,
                'Object_local1_p_w_to_o': None,
                'Object_local1_p_w_1': None,
                'Object_local1_p_w_2': None,
                'Object_local1_p_w_3': None,
                'Object_local1_p_w_4': None,
                'Object_local1_p_w_5': None,
                'Object_local1_p_w_6': None,
                'Object_local1_p_w_7': None,
                'Object_local1_p_w_8': None,
            }

            # Loop to fill all required object points in 'positions'
            for key in positions.keys():
                if 'Object_local1' in key:
                    while True:
                        object_point_name = input(f"Enter data for '{key}' (or 'skip' to leave it as None): ")
                        if object_point_name.lower() == 'skip':
                            break
                        # Capture the Optitrack data for this specific object point
                        object_position, _ = self._print_pose()
                        if object_position:
                            positions[key] = object_position
                            break
                        else:
                            logger.debug("No data received. Please ensure Optitrack is running and try again.")
            
            # Store the completed pose's data in the main dictionary
            position_all[pose_name] = positions

        # Ensure the directory exists before saving
        os.makedirs('data/U_shape_point_recordings', exist_ok=True)
        # Save the data to a JSON file
        with open(f'data/U_shape_point_recordings/{file_name}.json', 'w') as f:
            json.dump(position_all, f, indent=4)
        logger.debug(f"Data successfully saved to data/T_shape_point_recordings/{file_name}.json")

    def tranform_local_to_world_Tobject(self, P7_L1 = np.array([[-0.055, 0.0, 0.0]]).T):
        def quaternion_to_matrix(q):
                """Convert a quaternion (x, y, z, w) to a 3x3 rotation matrix."""
                x, y, z, w = q
                return np.array([
                    [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
                ])
            # Given values

        # T-shape object results, tested 
        # R_L2_to_L1 = np.array([[ 0.95211778, -0.30562785,  0.0079597 ],
        #                     [-0.30568869, -0.95209741,  0.00805851],
        #                     [ 0.00511551, -0.01010585, -0.99993585]]
        #                     )
        # P_L2_to_L1 = np.array([[ 0.00029164],
        #                     [-0.05317678],
        #                     [ 0.0035383 ]])


        # U-shape object results
        R_L2_to_L1 = np.array( [[ 0.99044297 , 0.13789649 ,-0.00269775],
                            [-0.13787775 , 0.99042981,  0.00620666],
                            [ 0.00352781, -0.00577539 , 0.9999771 ]])
        P_L2_to_L1 = np.array([[ 0.00156376],
                                [-0.04893837],
                                [-0.00310259]]
                                )


        # Define P7 in L1 frame
        # P7_L1 = np.array([[-0.055, 0.0, 0.0]]).T  # Shape (3, 1)

        # # Inverse R_L2_to_L1 to use for L1 to L2 transformation
        R_L1_to_L2 = np.linalg.inv(R_L2_to_L1)

        # Transform P7 from L1 to L2 frame
        P_L2 = R_L1_to_L2 @ P7_L1 + ( R_L1_to_L2 @ P_L2_to_L1)

        # Given measured values of R_w_to_L2 and P_w_to_L2 (assuming quaternion conversion if needed)
        # R_w_to_L2 = np.array([[...], [...], [...]])  # Replace with measured 3x3 rotation matrix for R_w_to_L2
        # P_w_to_L2 = np.array([[...], [...], [...]]).T  # Replace with measured 3x1 position vector for P_w_to_L2
        P_w_to_L2, R_w_to_L2 = self._print_Tobject_pose() 
        # print("P_w_to_L2: ", P_w_to_L2)
        R_w_to_L2 = quaternion_to_matrix(R_w_to_L2)
        # Transform P_L2 to world frame
        # print("R_w_to_L2 @ P_L2: ", R_w_to_L2 @ P_L2)
        R_w_to_L2 = np.array(R_w_to_L2)  # Ensure R_w_to_L2 is a NumPy array
        P_L2 = np.array(P_L2)            # Ensure P_L2 is a NumPy array
        P_w_to_L2 = np.array(P_w_to_L2)  # Ensure P_w_to_L2 is a NumPy array

        # Perform the transformation with reshaping
        P_w = (R_w_to_L2 @ P_L2.reshape(3, 1)) + P_w_to_L2.reshape(3, 1)

        logger.debug('%s %s', "Transformed position of P7 in world frame:", P_w.flatten())


    def test_estimated_transformationos(self):
        while True:
            rospy.sleep(0.1)
            ## Test T-shape results
            # P_local = np.array([[-0.055, 0.0, 0.0]]).T
            # print("P8")
            # self.tranform_local_to_world_Tobject(P_local)
            
            # P_local = np.array([[0.055, 0.0, 0.0]]).T
            # print("P6")
            # self.tranform_local_to_world_Tobject(P_local)

            # P_local = np.array([[0.015, 0.13, 0.0]]).T
            # print("P2")
            # self.tranform_local_to_world_Tobject(P_local)

            ## test U-shape results
            P_local = np.array([[0.039, 0.0, 0]]).T
            logger.debug("P8")
            self.tranform_local_to_world_Tobject(P_local)
            
            P_local = np.array([[-0.039, 0.0, 0]]).T
            logger.debug("P7")
            self.tranform_local_to_world_Tobject(P_local)

            P_local = np.array([[-0.019, 0.105, 0]]).T
            logger.debug("P2")
            self.tranform_local_to_world_Tobject(P_local)
            
            

if __name__ == '__main__':
    # Init ROS
    rospy.init_node('print_cartesian_state')
    control_server = CartesianStatePrinter()
    # control_server.run()  # for calibration 
    control_server.test_estimated_transformationos()  # for test the results of calibration