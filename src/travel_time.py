import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d, interp2d


class TravelTime:

    def __init__(self):
        pass

    @staticmethod
    def distance(lat1, lon1, lat2, lon2):
        scale = np.pi / 180.0
        lat1, lon1 = lat1 * scale, lon1 * scale
        lat2, lon2 = lat2 * scale, lon2 * scale
        a = np.sin(0.5 * (lat2 - lat1))**2 + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * (lon2 - lon1))**2
        rng = (180.0 / np.pi) * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return rng

    def time_diff(self, lat, lon):
        origin = self.origin
        stations = self.stations
        t_int = self.t_int
        # Distance between hypocentre and stations
        r1 = self.distance(origin["lat"], origin["lon"], stations["lat"].values, stations["lon"].values)
        # Distance between source location and stations
        r2 = self.distance(lat, lon, stations["lat"].values, stations["lon"].values)
        # Arrival times
        t1 = t_int(r1)
        t2 = t_int(r2)
        # Arrival time difference
        dt = t2 - t1
        return dt

    def construct_spatial_grid(self):
        print("  Constructing source location grid")
        Nx, Ny = self.grid_size
        dlon, dlat = self.grid_extent
        x = np.linspace(-dlon, dlon, Nx) + self.origin["lon"]
        y = np.linspace(-dlat, dlat, Ny) + self.origin["lat"]
        self.grid = {
            "grid_size": self.grid_size,
            "x": x,
            "y": y,
        }
        return self.grid

    def construct_slowness_grid(self):
        print("  Constructing slowness grid")
        Ntheta, Nslow = self.grid_size
        vmin, vmax = self.v_range
        theta_min, theta_max = self.theta_range

        # Linear grid over azimuth and slowness
        theta_grid = np.linspace(theta_min, theta_max, Ntheta)
        v_grid = np.linspace(vmin, vmax, Nslow)
        slow_grid = 1 / v_grid #输入是速度范围

        # Combine linear grids
        X, Y = np.meshgrid(theta_grid, slow_grid)
        theta, slowness = X.ravel(), Y.ravel()

        # Compute slowness in E/N directions
        Sx, Sy = -slowness * np.sin(theta), -slowness * np.cos(theta)
        Sx = Sx.reshape(X.shape).T
        Sy = Sy.reshape(X.shape).T

        self.grid = {
            "grid_size": self.grid_size,
            "grid_theta": theta_grid,
            "grid_slow": slow_grid,
            "slowness": (Sx, Sy),
        }
        return self.grid

    def construct_times_backprojection(self):
        print("  Constructing travel time look-up table")
        Ns = len(self.stations)
        Nx, Ny = self.grid["grid_size"]
        x, y = self.grid["x"], self.grid["y"]

        dt = np.zeros((Nx, Ny, Ns))

        for i in range(Nx):
            for j in range(Ny):
                dt_loc = self.time_diff(y[j], x[i])
                dt[i, j] = dt_loc - dt_loc.mean()

        self.dt = dt
        return dt

    def construct_times_beamforming(self):
        print("  Constructing time delay look-up table")
        Ns = len(self.stations)
        Ntheta, Nslow = self.grid["grid_size"]
        Sx, Sy = self.grid["slowness"]
        station_dist = self.station_dist
        dt = np.zeros((Ntheta, Nslow, Ns)) # 存储各个台站每个方向、慢度对应的时延

        for i in range(Ntheta):
            for j in range(Nslow):
                for k in range(Ns):
                    dt_x = Sx[i, j] * station_dist[k, 0]
                    dt_y = Sy[i, j] * station_dist[k, 1]
                    dt[i, j, k] = dt_x + dt_y
                dt[i, j] = dt[i, j] - dt[i, j].mean() # 在每个方向、慢度条件下减去对应的走时均值，获得每个台站时延

        self.dt = dt
        return dt

    def read_time_grid(self):
        print("  Reading velocity model")
        Ptimes = loadmat(self.ptimes_file)
        t = Ptimes["Ptt"]
        r = Ptimes["dis"].ravel()
        z = Ptimes["dep"].ravel()
        t_dep = interp2d(x=r, y=z, z=t, kind="linear")(r, self.origin["depth"])
        self.t_int = interp1d(x=r, y=t_dep, kind="linear")
        return self.t_int

    def compute_station_dist(self):
        # Earth's radius in km
        R_earth = 6371.0
        # Station inventory
        stations = self.stations
        N_stations = len(stations)
        station_codes = np.sort([station.code for station in stations])
        # Buffer for station-pair distances
        station_dist = np.zeros((N_stations, 2))
        # Reference station 阵列中心？
        st0 = stations.select(station=self.station_ref)[0]
        lat0, lon0 = st0.latitude, st0.longitude

        # Loop over all stations
        for i, station_code in enumerate(station_codes):
            # Select station i
            st1 = stations.select(station=station_code)[0]
            lat, lon = st1.latitude, st1.longitude
            # Compute distance between stations using small-angle approximation
            station_dist[i, 0] = R_earth * (lon - lon0) * np.pi / 180.
            station_dist[i, 1] = R_earth * (lat - lat0) * np.pi / 180.

        self.station_dist = station_dist
        return station_dist

