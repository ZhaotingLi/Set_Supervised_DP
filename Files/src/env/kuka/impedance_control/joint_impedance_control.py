"""
Authors:
    Rodrigo Perez-Dattari <r.j.perezdattari@tudelft.nl>
    Micah Prendergast <j.m.prendergast@tudelft.nl>
"""

import std_msgs.msg as std_msg
from sensor_msgs.msg import JointState
import numpy as np
import rospy


class JointImpedanceController:
    def __init__(self, robot_name, stiffness=1.0, alpha=1.0):
        # Parameters
        self.stiffness = np.array([600, 600, 500, 450, 180, 100, 40]) * stiffness
        self.damping = np.array([2.5, 2.5, 1.5, 1.5, 1.5, 1.2, 1.2]) * np.sqrt(self.stiffness)
        # self.max_q_delta = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10])

        self.max_q_delta = np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])

        # Init variables
        self.alpha = alpha
        self.q = None
        self.q_dot = None
        self.q_d_prev = None
        self.q_dot_d_prev = None

        # Start ROS publishers and subscribers
        self.command = rospy.Publisher('/%s/server_torque_command' % robot_name, std_msg.Float64MultiArray, queue_size=10)
        rospy.Subscriber('/%s/joint_states' % robot_name, JointState, self._callback_joint_states, queue_size=10)

    def _callback_joint_states(self, data):
        self.q = np.array(data.position)
        self.q_dot = np.array(data.velocity)

    def saturate_q_delta(self, q_delta):
        q_delta[np.abs(q_delta) > self.max_q_delta] = self.max_q_delta[np.abs(q_delta) > self.max_q_delta] \
                                                      * np.sign(q_delta)[np.abs(q_delta) > self.max_q_delta]
        return q_delta

    # def send_torque_request(self, q_d):
    #     # Smooth signal
    #     if self.q_d_prev is None:
    #         self.q_d_prev = q_d
    
    #     q_d_filtered = self.alpha * q_d + (1 - self.alpha) * self.q_d_prev
    #     self.q_d_prev = q_d_filtered

    #     # Get angle error
    #     q_delta = q_d_filtered - self.q

    #     # Saturate error if too large
    #     q_delta_saturated = self.saturate_q_delta(q_delta)

    #     # Compute desired torque with PD control
    #     torque = self.stiffness * q_delta_saturated - self.damping * self.q_dot

    #     # Create and publish ROS message
    #     msg = std_msg.Float64MultiArray()
    #     msg.data = torque
    #     self.command.publish(msg)
    #     return True

    def send_torque_request(self, q_d, q_dot_d=0):
        # Smooth signal
        if self.q_d_prev is None:
            self.q_d_prev = q_d

        if self.q_dot_d_prev is None:
            self.q_dot_d_prev = q_dot_d
    
        # q_d_filtered = self.alpha * q_d + (1 - self.alpha) * self.q_d_prev
        # self.q_d_prev = q_d_filtered

        # q_dot_d_filtered = self.alpha * q_dot_d + (1 - self.alpha) * self.q_dot_d_prev
        # self.q_dot_d_prev = q_dot_d_filtered
        q_d_filtered = q_d
        q_dot_d_filtered = q_dot_d
        # Get angle error
        q_delta = q_d_filtered - self.q

        # Saturate error if too large
        q_delta_saturated = self.saturate_q_delta(q_delta)

        # Get angle velocity error
        q_dot_delta = q_dot_d_filtered - self.q_dot

        # Compute desired torque with PD control
        torque = self.stiffness * q_delta_saturated + self.damping * q_dot_delta

        # Create and publish ROS message
        msg = std_msg.Float64MultiArray()
        msg.data = torque
        self.command.publish(msg)
        return True