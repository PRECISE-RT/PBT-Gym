# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:32:16 2023

@author: Rob Chambers
"""
import os
import pydicom as dcm
import numpy as np
import cv2 as cv
import preprocessing

class BrainRegion(object):
    def __init__(self, directory, structure, contour_name, threshold):
        self.directory = directory
        self.contour_name = contour_name
        self.threshold = threshold
        self.structure = structure
        self.roi_num = self.get_roi_index(contour_name)
        self.slice_number = 0 # Will be removed when moving to multiple layers of CT
        self.contour_coords = self.map_contour_to_ct(self.roi_num)
        self.boolean_map, self.contour_map = self.create_contour_map(self.contour_coords)
        self.coordinate_stack, self.r_uids = self.get_coordinate_stack()
        self.boolean_stack, self.binary_stack = self.get_contour_stack()

    def get_roi_index(self, roi_name):
        """
        Takes a string and outputs the roi location within the structure file of
        corresponding contour e.g CTV 70, brainstem etc.
        """
        roi_num = 0
        while roi_name != self.structure.StructureSetROISequence[roi_num].ROIName:
            roi_num += 1
            if roi_num > len(self.structure.StructureSetROISequence) - 1:
                raise Exception("Sorry, that name is not contained in this Structure Sequence")
        print(f"The ROI index for {roi_name} is {roi_num}")
        return roi_num

    def get_coordinate_stack(self):
        """
        Returns the set of contours present in a 2D CT scan.
        """
        roi_contours = []
        referenced_uids = []
        for slice in os.scandir(self.directory):
            if slice.is_file():
                ds_ct = dcm.dcmread(slice.path)
                slice_id = ds_ct.SOPInstanceUID
                # Plotting iteratively through all the contours of one ROI on one CT scan
                for j in range(len(self.structure.ROIContourSequence[self.roi_num].ContourSequence)):
                    for k in range(len(self.structure.ROIContourSequence[self.roi_num].ContourSequence[j].ContourImageSequence)):
                        if self.structure.ROIContourSequence[self.roi_num].ContourSequence[j].ContourImageSequence[k].ReferencedSOPInstanceUID == slice_id:
                            point_number = self.structure.ROIContourSequence[self.roi_num].ContourSequence[j].NumberOfContourPoints

                            contour_array_old_coords = np.reshape(self.structure.ROIContourSequence[self.roi_num].ContourSequence[j].ContourData, (point_number, 3))
                            new_x = (contour_array_old_coords[:,0] - ds_ct.ImagePositionPatient[0])/ ds_ct.PixelSpacing[0]
                            new_y = (contour_array_old_coords[:,1] - ds_ct.ImagePositionPatient[1])/ ds_ct.PixelSpacing[1]
                            #ct_z_coord = contour_array_old_coords[:,2]
                            contour_data = np.array([new_x, new_y])
                            roi_contours.append(contour_data)
                            referenced_uids.append(slice_id)
                            continue

        return roi_contours, referenced_uids

    def get_contour_stack(self):
        """
        Returns a 3D stack of contours in binary and boolean data types.
        """
        binary_arrays = []
        boolean_arrays = []
        for i in range(len(self.coordinate_stack)):
            ith_boolean, ith_binary = self.create_contour_map(self.coordinate_stack[i])
            binary_arrays.append(ith_binary)
            boolean_arrays.append(ith_boolean)

        binary_stack = np.stack(binary_arrays)
        boolean_stack = np.stack(boolean_arrays)
        
        return boolean_stack, binary_stack

    def map_contour_to_ct(self, roi_num):
        """
        Maps the list of contour values to (3,n) dimensional array of x,y,z
        coords that match CT space.
        """
        ct_files = os.scandir(self.directory)
        ct_file = dcm.dcmread(ct_files[0].path)
        no_points = self.structure.ROIContourSequence[roi_num].ContourSequence[self.slice_number].NumberOfContourPoints
        contour_array_old_coords = np.reshape(np.array(self.structure.ROIContourSequence[roi_num].ContourSequence[self.slice_number].ContourData),
                                              (no_points, 3))
        patient_pos_x = np.full(np.shape(contour_array_old_coords[:,0]), float(ct_file.ImagePositionPatient[0]))
        patient_pos_y = np.full(np.shape(contour_array_old_coords[:,1]), float(ct_file.ImagePositionPatient[1]))
        new_x = (contour_array_old_coords[:,0] - patient_pos_x) / float(ct_file.PixelSpacing[0])
        new_y = (contour_array_old_coords[:,1] - patient_pos_y) / float(ct_file.PixelSpacing[1])
        #ct_z_coord = contour_array_old_coords[:,2]
        contour_data = np.array([new_x, new_y]) #np.array([new_x, new_y, ct_z_coord])

        return contour_data

    def create_contour_map(self, contour):
        """
        Takes a list of x,y coordinates and creates a map of size (512,512)
        containing the contour. The map has values of -1 outside the contour, 0 
        on the contour and +1 inside the contour.
        """

        x_vals = contour[0,:]
        y_vals =  contour[1,:]
        contour_xy = list(map(lambda x, y:(int(np.rint(x)),int(np.rint(y))), x_vals, y_vals))

        src = np.zeros((512, 512), dtype=np.uint8)
        num_points = len(x_vals)

        for i in range(num_points):
            cv.line(src, contour_xy[i], contour_xy[(i+1)%num_points], ( 255 ), 3)
        # Get the contours
        contours, hierarchy = cv.findContours(src, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        boolean_map = np.empty(src.shape, dtype=bool)
        binary_map = np.empty(src.shape)
        
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                if cv.pointPolygonTest(contours[0], (j,i), False)>0:
                    binary_map[i,j] = 1
                    boolean_map[i,j] = True
                else:
                    binary_map[i,j] = -1
                    boolean_map[i,j] = False

        raw_dist = np.empty(src.shape)
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                raw_dist[i,j] = cv.pointPolygonTest(contours[0], (j,i), True)
        
        
        return boolean_map, binary_map

    def update_slices(self):
        """
        Updates the active slice in the brain regions (to the one chosen in the 
        reset method). 
        """
        self.boolean_map = self.boolean_stack[self.slice_number]
        self.contour_map = self.binary_stack[self.slice_number]