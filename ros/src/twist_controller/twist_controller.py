from lowpass import LowPassFilter
from pid import PID
import rospy
from yaw_controller import YawController
#############################################
# twist_controller.py
# First cut: Udacity
# rtarkunde: Modified as per the walkthrough
#############################################

GAS_DENSITY = 2.858
ONE_MPH     = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass_in, fuel_capacity_in, brake_deadband_in, 
                 decel_limit_in, accel_limit_in, wheel_radius_in, wheel_base_in,
                 steer_ratio_in, max_lat_accel_in, max_steer_angle_in):
        self.yaw_controller = YawController(wheel_base_in, steer_ratio_in, 0.1, max_lat_accel_in, max_steer_angle_in)
        
        kp = 0.3
        ki = 0.1
        kd = 0.0
#        ki = 0.01
#        kd = 0.1
        min_throttle = 0.0
        max_throttle = 0.10
        self.throttle_controller = PID(kp, ki, kd, min_throttle, max_throttle)
        
        tau           = 0.5
        sampling_time = 0.02  # 50 Hz
        self.vel_lpf = LowPassFilter(tau, sampling_time)
        
        self.vehicle_mass   = vehicle_mass_in
        self.fuel_capacity  = fuel_capacity_in
        self.brake_deadband = brake_deadband_in 
        self.decel_limit    = decel_limit_in
        self.accel_limit    = accel_limit_in
        self.wheel_radius   = wheel_radius_in
        self.last_time      = rospy.get_time()
        
    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0. ,0.

        current_vel = self.vel_lpf.filt(current_vel)
        
#        rospy.logwarn("Angular vel: {0}".format(angular_vel))
#        rospy.logwarn("Target vel: {0}".format(linear_vel))
#        rospy.logwarn("Current vel: {0}".format(current_vel))
        
        steering  = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
#        rospy.logwarn("Steering : {0}".format(steering))
        
        vel_error = linear_vel - current_vel
        self.last_vel  = current_vel
        #rospy.logwarn("vel_error: {0}".format(vel_error))
        
        current_time   = rospy.get_time()
        sample_time    = current_time - self.last_time
        self.last_time = current_time
        
        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake    = 0
        # Changed to 0.015 fix car stopping way beyond stopline
        if linear_vel == 0. and current_vel < 0.015:
            throttle   = 0.  # Testing purpose did not work
            #brake     = 400 # N*m Hold car at stop light
            brake      = 700 # N*m Hold car at stop light
            ## To be changed to 700 above for Carla. Done.
        elif throttle < 0.1 and vel_error < 0:
            throttle  = 0.
            decel     = max(vel_error, self.decel_limit)
            brake     = abs(decel)*self.vehicle_mass*self.wheel_radius
            #rospy.logwarn("Decel: {0}".format(decel))
            #rospy.logwarn("Brake: {0}".format(brake))
        return throttle, brake, steering


