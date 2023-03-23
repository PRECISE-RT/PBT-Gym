# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:33:53 2023

@author: Rob Chambers
"""

import pydicom as dcm

DOSE_PATH = "/workspaces/MPhys Gamification/HNSCC-01-0001_DICOM/RD.HNSCC-01-0001.4MPhys.dcm"
STRUCTURE_PATH = "/workspaces/MPhys Gamification/HNSCC-01-0001_DICOM/RS.HNSCC-01-0001.Oropharynx.dcm"
CT_PATH = "/workspaces/MPhys Gamification/HNSCC-01-0001_DICOM/CT.HNSCC-01-0001.Image 22.dcm"
DIRECTORY = "/workspaces/MPhys Gamification/HNSCC-01-0001_DICOM"
CT_FILE = dcm.dcmread(CT_PATH)
STRUCTURE = dcm.dcmread(STRUCTURE_PATH)
DOSE_FILE = dcm.dcmread(DOSE_PATH)
