import os
import numpy as np

# Subclasses
from src.beamforming import Beamforming
from src.seismic_array import SeismicArray
from src.travel_time import TravelTime
from src.utils import Utils


class MUSIC(Beamforming, SeismicArray, TravelTime, Utils):

    def __init__(self):
        super().__init__()
        pass

    def set_params(self, params):
        self.__dict__.update(params)

    def sanity_check(self):
        print("  Performing sanity check")
        return True

    def initialise(self):
        print("Initialising...")
        self.sanity_check()
        self.construct_grid()
        self.read_stations()
        self.read_time_grid()
        self.construct_time_table()
        print("Initialisation complete")
        pass
