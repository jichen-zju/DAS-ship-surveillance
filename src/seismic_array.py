import os

import obspy
import geopy.distance
import pandas as pd


class SeismicArray:

    def __init__(self):
        pass

    def read_stations(self):
        print("  Reading seismic station inventory")
        stations = obspy.read_inventory(self.station_inventory_file)[0]
        station_df = pd.DataFrame(columns=("station", "lat", "lon", "dist"))

        for station in stations:
            station_loc = (station.latitude, station.longitude)
            dist = geopy.distance.distance((self.origin["lat"], self.origin["lon"]), station_loc).km
            station_df = station_df.append({
                "station": station.code,
                "lat": station.latitude,
                "lon": station.longitude,
                "dist": dist,
            }, ignore_index=True)

        station_df.sort_values(by=["dist", "lat", "lon"], inplace=True)
        station_df.reset_index(inplace=True, drop=True)
        station_df = station_df.iloc[self.station_selection]
        station_df.reset_index(inplace=True, drop=True)
        self.stations = station_df
        return self.stations
