import numpy as np
from numpy.fft import rfft, irfft
from scipy.signal import  butter, filtfilt
from scipy.signal.windows import tukey
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


class Utils:

    def __init__(self):
        pass

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=4):

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        if low < 0:
            Wn = high
            btype = "lowpass"
        elif high < 0:
            Wn = low
            btype = "highpass"
        else:
            Wn = [low, high]
            btype = "bandpass"

        b, a = butter(order, Wn, btype=btype)

        return b, a

    def taper_filter(self, arr, fmin, fmax, samp):
        b_DAS, a_DAS = self.butter_bandpass(fmin, fmax, samp)
        window_time = tukey(arr.shape[1], 0.1)
        arr_wind = arr * window_time
        arr_wind_filt = filtfilt(b_DAS, a_DAS, arr_wind, axis=-1)
        return arr_wind_filt

    @staticmethod
    def xcorr(x, y):
        # Compute norm of data
        norm_x_sq = (x**2).sum() # 二范数
        norm_y_sq = (y**2).sum()
        norm = np.sqrt(norm_x_sq * norm_y_sq)

        # FFT of x and conjugation
        X_bar = rfft(x).conj()
        # FFT of y
        Y = rfft(y)
        # Correlation coefficients
        R = irfft(X_bar * Y)
        # Index of maximum correlation coefficient (= shift)
        shift = np.argmax(R)
        # Select normalised maximum correlation coefficient (0 < R <= 1)
        R_max = R[shift] / norm
        # Return shift and correlation coefficient
        return shift, R_max

    @staticmethod
    def unwrap(x, N):
        x = x % N
        x -= int(x.mean())
        wrap_inds = (x < N // 2) # // 向下取整
        x[wrap_inds] += N
        wrap_inds = (x > N // 2)
        x[wrap_inds] -= N
        x -= int(x.mean())
        return x

    def align_hypocentre(self, data, hypo_dt, start=200, win=300):
        shifted_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            shifted_data[i] = np.roll(data[i], shift=-int(hypo_dt[i] * self.fsamp))

        select_data = shifted_data[:, start:start + win]
        select_shifts, _, _ = self.delaysum(select_data)
        select_shifts = self.unwrap(select_shifts, win)

        for i in range(data.shape[0]):
            shifted_data[i] = np.roll(shifted_data[i], -select_shifts[i])

        return shifted_data, select_shifts

    def verify_locations(self):

        lats = self.stations["lat"].values
        lons = self.stations["lon"].values

        x, y = self.grid["x"], self.grid["y"]
        X, Y = np.meshgrid(x, y)

        plt.figure(figsize=(5, 3))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.stock_img()
        plt.plot(lons, lats, ".", transform=ccrs.Geodetic(), markersize=1, alpha=0.5, c="k")
        plt.plot(X.ravel(), Y.ravel(), ".", transform=ccrs.Geodetic(), markersize=1, alpha=0.5, c="r")
        plt.show()
        pass

