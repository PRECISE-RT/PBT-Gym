# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:02:14 2023

@author: Rob Chambers

This file contains the ParallelBeamEnv class, inheriting from openAI gym's Env
class. It's purpose is to create a basic model of a proton beam dosing a patient
so that reinforcing learning techniques. 

Adding user input step functions for use with any user defined dosing model.
"""

from .requirements import *
from .beam import Beam
from .dose_model import complex_dose, bit_wise_dose
from .brain_region import BrainRegion

class ParallelBeamEnv(Env):
    def __init__(self, dose_choice: str, action_space_type: str, CANCER: BrainRegion, EXTERNAL: BrainRegion, user_step: function):
        super(ParallelBeamEnv, self).__init__()
        self.cancer = CANCER
        self.external = EXTERNAL
        self.dose_choice = dose_choice
        self.user_step = user_step
        if dose_choice == "complex":
            self.apply_dose = complex_dose
        elif dose_choice == "bit_wise":
            self.apply_dose = bit_wise_dose
        else:
            raise Exception(f"No dosing method found under name: {dose_choice}. Please enter a valid dose method e.g. 'complex', 'bit_wise'.")
        
        self.dose.canvas = -0.7 * np.array(self.cancer.boolean_map, dtype=np.float32)
        self.match_uid(self.cancer, self.external)
        self.search_space = self.bound_search_space(self.cancer)
        self.learn_space = self.bound_search_space(self.external)
        self.beam = Beam(self.search_space) # Can be reduced by finding the bounds of the relevant contours and only permitting the beam to operate within these regions. NOTE: there will be issues when treating cancers with multiple regions
        self.action_space = spaces.MultiDiscrete([3,3,3,2])# Position of spot, angle of beam, beam intensity
        self.x_min = self.learn_space[0]
        self.x_max = self.learn_space[1]
        self.y_min = self.learn_space[2]
        self.y_max = self.learn_space[3]
        self.x_size = self.x_max - self.x_min
        self.y_size = self.y_max - self.y_min
        self.observation_shape = (self.x_size, self.y_size) # (175, 333, 123, 320), (242, 310, 140, 240), (167, 339, 109, 330)
        self.target_dose = 0.3 # Dose prescription measured in Gy
        self.observation_space = spaces.Dict({
                                "position": spaces.Box(low = np.array(self.search_space[0:3:2]), 
                                                       high = np.array(self.search_space[1:4:2]),
                                                       dtype=int),
                                "cancer bool": spaces.Discrete(2),
                                "dose distribution": spaces.Box(low = -1 * np.ones(self.observation_shape),
                                                                high = np.ones(self.observation_shape),
                                                                dtype=np.float32)})
        self.treatment_period = 1024 # Number of steps per learning perdiod
        self.state = {
            "position": [self.search_space[0], self.search_space[2]],
            "cancer bool": 0,
            "dose distribution": -0.7 * np.array(self.cancer.boolean_map[self.x_min:self.x_max,
                                                                        self.y_min:self.y_max],
                                                                        dtype=np.float32)}
        self.theta_resolution = 45 # How many degrees changed per step
        self.default_dose = 0.1 # Default amount to be used until variable dosing is implemented. Can also be a reference value for decrete choice dosing e.g choices are 0, 0.25, 0.5, 1, 1.5, 2, 2.5x reference dose when choosing
        self.step_size = 3
        self.other_brain_regions = []
        self.reward_multipliers = []
        self.reward_stack = self.calculate_reward_stack()
        self.reward_map = self.reward_stack[0]
        self.update_reward_map()

    def default_step(self, action):
        """
        Applies the action chosen by the Agent. Calculates 'reward'. Updates the
        'state'. Decreases the treatment period by 1. Updates the 'done' status.
        Returns 'info'. 
        """
        # Initially set reward to zero so if no action taken, reward does not carry over from previous step
        reward = 0
        # Check if plan is done
        if self.treatment_period <= 0: 
            done = True
        else:
            done = False
        
        # Translational Shift
        del_xy = (action[0:2] - 1) * self.step_size
        self.beam.move(del_xy[0], del_xy[1])
        self.state["position"] = np.array([self.beam.x, self.beam.y])
        
        if self.is_beam_over_cancer() == True:
            self.state["cancer bool"] = 1
        else:
            self.state["cancer bool"] = 0

        # Angular Shift
        del_theta = (action[2] - 1) * self.theta_resolution
        self.beam.rotate(del_theta)
        # Apply_dose and calculate corresponding reward
        if action[3] == 1:
            reward = self.apply_dose()# Change the dosing methods here
            self.state["dose distribution"] = self.dose.canvas[self.x_min:self.x_max, self.y_min:self.y_max]

        # Check for overdosing
        if (np.max(self.state["dose distribution"]) > self.target_dose):
            reward -= 10
            done = True
        
        # Reduce treatment period by 1; Retaining treatment period approach in case method changes in the future
        self.treatment_period -= 1

        info = {}
        # Rendering for dose checks
        # if done == True:
        #     self.render()
        return self.state, reward, done, info

    def step(self, action):
        """
        Step function now dependent on user defined
        """
        if callable(self.user_step):
            return self.user_step(self, action)
        else:
            return self.default_step(self, action)
        
    def reset(self):
        """
        Resets environment to initial state. Currently set to zeros but could also
        be set to either base CT image. This will also have to change if we move
        from 'Box' to 'Dict' space. 
        """
        # Random Slice
        self.cancer.slice_number = np.random.randint(0, high=len(self.cancer.r_uids)) # 38 being number of slices in the dataset with the CTV cancer involved, 33 in the external
        self.match_uid(self.cancer, self.external)
        self.cancer.update_slices()
        self.external.update_slices()
        self.beam.update_search_space(self.cancer.contour_map)
        #(self.beam.x_min, self.beam.x_max, self.beam.y_min, self.beam.y_max) = self.bound_search_space()
        # If other brain region objects added as vector, can iterator through them here

        self.update_reward_map()
        #print(f"Environment has reset using slice number {x}")

        self.treatment_period = 1024
        self.state = {
            "position": np.array([self.search_space[0], self.search_space[2]], dtype=int),
            "cancer bool": 0,
            "dose distribution": -0.7 * np.array(self.cancer.boolean_map[self.x_min:self.x_max,
                                                                         self.y_min:self.y_max],
                                                                         dtype=np.float32)}
        self.beam.x = np.random.randint(self.beam.x_min, self.beam.x_max)
        self.beam.y = np.random.randint(self.beam.y_min, self.beam.y_max)
        # self.beam.x = int((self.beam.x_min + self.beam.x_max) / 2)
        # self.beam.y = int((self.beam.y_min + self.beam.y_max) / 2)
        self.dose.canvas = -0.7 * np.array(self.cancer.boolean_map, dtype=np.float32)
        return self.state
    
    def apply_dose(self) -> float:
        """
        Creates a line of a given thickness between two points and applies a 
        custom gradient to said line (function of distance along the line).
        Then calculates a normalised reward based on the overlap of dose and
        contour.
        """
        x = self.env.beam.x
        y = self.env.beam.y
        theta = self.env.beam.theta
        # Defining properties of each circle to be drawn
        radius = 2
        n=0
        score = 0
        if not self.env.within_patient(x,y):# Just double check this later
        #     print("Not within patient")
            score -= 0.1
        # else:
        #     print("Within target")

        if theta == 0:
            while(self.env.within_patient(x+n,y)):
                value = self.env.beam.calculate_dose_gradient(n)
                self.env.canvas[y+n,x-radius:x+radius+1] += value
                score += np.sum(self.reward_map[y+n,x-radius:x+radius+1]) * value / (2*radius + 1)
                n += 1

        elif theta == 180:
            while(self.env.within_patient(x-n,y)):
                value = self.env.beam.calculate_dose_gradient(n)
                self.env.canvas[y-n,x-radius:x+radius+1] += value
                score += np.sum(self.env.reward_map[y-n,x-radius:x+radius+1]) * value / (2*radius + 1)
                n += 1

        elif theta == 90:
            while(self.env.within_patient(x,y+n)):
                value = self.env.beam.calculate_dose_gradient(n)
                self.env.canvas[y-radius:y+radius+1,x+n] += value
                score += np.sum(self.env.reward_map[y-radius:y+radius+1,y+n]) * value / (2*radius + 1) # Multiplicative term at the end here is a normalisation factor (value [0,1], 2x Radius + 1 pixels per step)
                n += 1

        elif theta == 270:
            while(self.env.within_patient(x,y-n)):
                value = self.env.beam.calculate_dose_gradient(n)
                self.env.canvas[y-radius:y+radius+1,y-n] += value
                score += np.sum(self.env.reward_map[y-radius:y+radius+1,x-n]) * value / (2*radius + 1)
                n += 1

        elif theta < 90 or theta > 270:
            m = np.rint(np.tan(theta*np.pi/180))
            while(self.env.within_patient(x+n,int(y+m*n))):
                distance = n * np.sqrt(1 + m**2) # should always be sqrt(2) x n for this model but writing like this in case theta precision changes
                value = self.env.beam.calculate_dose_gradient(distance)
                for r in range(2*radius+1):
                    k = r - radius
                    self.env.canvas[y+n-k, int(x+m*(n+k))] += value
                    score += np.sum(self.env.reward_map[y+n-k, int(x+m*(n+k))]) * value / (2*radius + 1)  
                for r in range(2*radius):
                    k = r - radius
                    self.env.canvas[y+n+k, int(x+m*(n-k-1))] += value
                    score += np.sum(self.env.reward_map[y+n+k, int(x+m*(n-k-1))]) * value / (2*radius)
                n += 1

        elif theta > 90 and theta < 270:
            m = np.rint(np.tan(theta*np.pi/180))
            while(self.env.within_patient(x-n,int(y-m*n))):
                distance = n * np.sqrt(1 + m**2) # should always be sqrt(2) x n for this model but writing like this in case theta precision changes
                value = self.env.beam.calculate_dose_gradient(distance)
                for r in range(2*radius+1):
                    k = r - radius
                    self.env.canvas[y-n-k,int(x-m*(n-k))] += value
                    score += np.sum(self.env.reward_map[y-radius:y+radius+1,x-n]) * value / (2*radius + 1)

                for r in range(2*radius):
                    k = r - radius
                    self.env.canvas[y-n+k, int(x-m*(n+k+1))] += value
                    score += np.sum(self.env.reward_map[y-n+k, int(x-m*(n+k+1))]) * value / (2*radius)
                n += 1
        return score

    def render(self, mode = "human"):
        """
        Renders the dosing information of the AI upon a CT image.
        """
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            #s_s = self.search_space
            plt.imshow(self.dose.canvas)#[s_s[0]:s_s[2] + 1, s_s[1]:s_s[3]+ 1])
            plt.savefig("LatestDose")# Maybe have the contour outputted for humans as well
            return None
    
        elif mode == "rgb_array":
            return self.dose.canvas

    def is_beam_over_cancer(self):
        """
        Checks if the whole beam is over the cancer contour and if the dose
        exceeds the maximum.
        """
        x = self.beam.x
        y = self.beam.y        
        beam_over_cancer = self.cancer.boolean_map[(y-1):(y+2),(x-1):(x+2)]
        dose = np.max(self.dose.canvas[(y-1):(y+2),(x-1):(x+2)])
        if (np.sum(beam_over_cancer) == 9) and (dose < 0.7):# Just a placeholder for the dose limit
            return True
        else:
            return False

    def bound_search_space(self, region):
        """
        Runs through each element in binary map for each slice in a region and
        outputs the maximum and minimum boundaries for each contour region.
        """
        min_x = 512
        max_x = 0
        min_y = 512
        max_y = 0
        for k in range(len(region.binary_stack)):
            for i in range(512):
                for j in range(512):
                    if region.binary_stack[k][j,i] == 1:
                        if i < min_x:
                            min_x = i
                        if i > max_x:
                            max_x = i
                        if j < min_y:
                            min_y = j
                        if j > max_y:
                            max_y = j

        return (int(min_x), int(max_x), int(min_y), int(max_y))

    def within_cancer(self, i, j):
        """
        Returns a True if the coordinates are within the contour, else returns
        False.
        """
        
        if max(i,j) > 512 or min(i,j) < 0:
            return False
        elif self.cancer.contour_map[j,i] == 1:
            return True
        else:
            return False

    def within_patient(self, i, j):
        """
        Returns a True if the coordinates are within the contour, else returns
        False.
        """
        if max(i,j) > 512 or min(i,j) < 0:
            return False
        elif self.external.contour_map[j,i] == 1:
            return True
        else:
            return False
    
    def calculate_reward_stack(self):
        """
        Calculates the reward map corresponding to every slice of the cancer.
        """
        reward_arrays = []
        
        for i in range(len(self.cancer.boolean_stack)):
            self.cancer.slice_number = i
            self.match_uid(self.cancer, self.external)
            self.cancer.update_slices()
            self.external.update_slices()
            ith_reward_map = self.calculate_reward_map()
            reward_arrays.append(ith_reward_map)
            
        reward_stack = np.stack(reward_arrays)
        return reward_stack

    def calculate_reward_map(self):
        """
        Calculates a reward map that priritises the cancer contour. Other 
        contours can be added via the add_to_reward_map() function.
        """
        reward_map = np.zeros((512,512))
        # Calculates lowest hierachy first
        reward_map[self.external.boolean_stack[self.external.slice_number]] = -1
        
        # for i in range(len(other_brain_regions)):
        #     reward_map[other_brain_regions[i].boolean_map] = multipliers[i]

        # Calculates highest hierachy last
        reward_map[self.cancer.boolean_stack[self.cancer.slice_number]] = 1

        return reward_map
    
    def update_reward_map(self):
        """
        Updates the reward map to the givn slice index.
        """
        self.reward_map = self.reward_stack[self.cancer.slice_number]

    def match_uid(self, region_1, region_2):
        """
        Takes region and a slice index of that region and finds the 
        corresponding slice in a second given region using referenced UID
        matching. If a match is found it updates the second region's slice 
        number.
        """
        slice_id = region_1.r_uids[region_1.slice_number]
        for i in range(len(region_2.r_uids)):
            if region_2.r_uids[i] == slice_id:
                region_2.slice_number = i
                return None
                
        raise Exception(f"Slice {region_1.slice_number} from {region_1.contour_name}"
                        "could not be matched with {region_2.contour_name}.")

