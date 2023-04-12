# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:18:37 2023

@author: Rob Chambers
"""
import numpy as np

class Beam(object):
    def __init__(self, contour_region):
        (self.x_min, self.x_max, self.y_min, self.y_max) = contour_region
        self.x = self.x_min
        self.y = self.y_min
        self.theta = 0
        self.distance_from_cancer = 0
        self.distance_from_cord = 0

    def get_position(self):
        return (self.x, self.y)

    def move(self, del_x, del_y):
        """
        Moves the x,y values by some prescribed amount. The position is then 
        bounded by the values set by the observation space.
        """
        self.x += del_x
        self.y += del_y

        self.x = self.bound(self.x, self.x_min, self.x_max)
        self.y = self.bound(self.y, self.y_min, self.y_max)

    def rotate(self, del_theta):
        """
        Increases the value of theta in the anti-clockwise direction. Measured 
        in degrees, modular(360). 
        """
        self.theta = (self.theta + del_theta)%360

    def bound(self, n, minn, maxn):
        """
        Bounds a given number, n, between some given minimum and maximum, 
        (minn, maxn). If n exceeds the boundary, returns the closest allowed
        value.
        """
        return max(min(maxn,n), minn)

    def update_search_space(self, contour_map):
        """
        Runs through each element in binary map
        """
        min_x = 512
        max_x = 0
        min_y = 512
        max_y = 0
        for i in range(512):
            for j in range(512):
                if contour_map[j,i] == 1:
                    if i < min_x:
                        min_x = i
                    if i > max_x:
                        max_x = i
                    if j < min_y:
                        min_y = j
                    if j > max_y:
                        max_y = j

        (self.x_min, self.x_max, self.y_min, self.y_max) = (min_x, max_x, min_y, max_y)

    def find_proton_energy():
        """
        Finds the proton energy required to place the Bragg peak of the spot at
        the required location in the brain given the location of the beam
        """
    def calculate_dose_gradient(self, distance):
        """
        A WET approximation calculation to give the intensity of dose as a
        function of distance from beam spot
        """
        return 0.1 * np.exp(-(distance**2)/3)
