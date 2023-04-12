from __future__ import annotations
from . import util

def data_loader_2D(CT_fp: str, RS_fp: str, split:float) -> tuple[list, list]:
    pass

def data_loader_3D(CT_fp: str, RS_fp: str, split:float) -> tuple[list, list]:
    pass

class BaseDataLoader():
    pass

class DataLoader2D(BaseDataLoader):
    """
    This accounts for slice based training and will split for data for all available slices.
    """
    def __init__(self):
        self.CT_fp = None
        self.RS_fp = None
        self.CT = None
        self.split = None

class DataLoader3D(BaseDataLoader):
    """
    This accounts for patient based training and will split for data for all available patients.
    """
    def __init__(self):
        self.CT_fp = None
        self.RS_fp = None
        self.CT = None
        self.split = None