# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:38:47 2023

@author: Rob Chambers
"""
import beam
import BrainRegion

class Patient(object):
    def __init__(self, directory, structure, cancer_name="CTV 70", threshold,
                 ext_name="External", ):
        self.cancer = BrainRegion(directory, structure, cancer_name, threshold)
        self.external = BrainRegion(ext_name, threshold)
        self.match_uid(self.cancer, self.external)
        self.search_space = self.bound_search_space(self.cancer)
        self.learn_space = self.bound_search_space(self.external)


        
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
