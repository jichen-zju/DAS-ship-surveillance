import math
import numpy as np
import pandas as pd
from pyproj import Proj
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, date2num,num2date,MinuteLocator
import os
import cartopy.crs as ccrs
import h5py
import warnings
from scipy.signal import butter, filtfilt
from scipy.signal.windows import tukey
import math
import matplotlib.pyplot as plt
from pyproj import Transformer
music_dir = os.path.join(os.getcwd(), "PyMUSIC")
sys.path.append(music_dir)
from src.base import MUSIC
from matplotlib.colorbar import Colorbar
warnings.filterwarnings("ignore")
from scipy.ndimage import gaussian_filter as gf
from math import sin, asin, cos, radians, fabs, sqrt, tan,atan
from scipy.signal import spectrogram
import cartopy.mpl.ticker as cticker
import time
import matplotlib as mpl
import xarray as xr

def read_elevation(path):
    data = xr.open_dataset(path, engine='netcdf4')

    lat_list = data.lat.to_series().values  #纬度
    lon_list = data.lon.to_series().values  #经度


    # fig=plt.figure()
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax.contourf(lon_list, lat_list, data["elevation"].data,
    #             cmap="Greens")
    return lon_list, lat_list, data["elevation"].data

def butter_bandpass(lowcut, highcut, fs, order=2): # 带通滤波
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


def taper_filter(arr, fmin, fmax, samp_DAS):
    b_DAS, a_DAS = butter_bandpass(fmin, fmax, samp_DAS)
    window_time = tukey(arr.shape[-1], 0.1) # 给阵列数据加了turkey窗
    arr_wind = arr * window_time
    arr_wind_filt = filtfilt(b_DAS, a_DAS, arr_wind, axis=-1)
    return arr_wind_filt

#二倍角公式
def hav(theta):
    s = sin(theta / 2)
    return s * s
#
def get_distance_hav(lat0, lng0, lat1, lng1):
    "用haversine公式计算球面两点间的距离。"
    # 经纬度转换成弧度
    lat0 = float(lat0)
    lat1 = float(lat1)
    lng0 = float(lng0)
    lng1 = float(lng1)
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * 6371 * asin(sqrt(h))
    return distance

# 计算研究时段内船和ref point的平均距离
def cal_average_dis(ship_info,t1,t2,ref_lon,ref_lat):
    dis = []
    for i in range(0,ship_info.shape[0]):
        if ship_info[i,2] >= t1 and ship_info[i,2]<= t2:
            dis_now = 1000 * get_distance_hav(ship_info[i,1],ship_info[i,0],ref_lat,ref_lon)
            dis.append(dis_now)
    average = np.mean(dis)
    return average

def get_ship_info(all_data,name):
    length_all = all_data.shape[0]
    k = 0
    for i in range(0,length_all):
        if all_data[i,0] == name:
            k = k + 1
    ship_info = np.zeros((k,3))
    m = 0
    for i in range(0,length_all):
        if all_data[i,0] == name:
            ship_info[m,0] = all_data[i,5] # lon
            ship_info[m,1] = all_data[i,6] # lat
            ship_info[m,2] = all_data[i,7] # time
            m = m + 1
        if m > k+1:
            break
    return ship_info

def FK_filter(data_in, fs, dx, cmin):
    ''' F-K filtering between cmin and cmax contours '''
    Nx = data_in.shape[0]
    Ns = data_in.shape[1]
    f0 = np.fft.fftshift(np.fft.fftfreq(Ns, d=1. / fs))
    k0 = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    ft2 = np.fft.fftshift(np.fft.fft2(data_in))
    F, K = np.meshgrid(f0, k0)


    C = F / K
    filt = np.zeros(ft2.shape)
    filt[np.logical_or(C > cmin, C < -cmin)] = 1

    filt = gf(filt, 3)  # blur the filter a little to reduce Gibbs ringing

    ft2f = ft2 * filt
    data_out = np.fft.ifft2(np.fft.fftshift(ft2f)).astype(float)

    return data_out

def cal_source(waveform,channel_loc,start_time,freq_band,bft):

    dx = 4
    fs = 500
    N_theta = 360
    N_slow = 2
    bf_time = bft
    sample_number= bf_time * fs
    center_lon,center_lat = np.mean(channel_loc, axis=0)

    # filter
    waveform_filted = taper_filter(waveform,freq_band[0],freq_band[1],fs)

    dist = np.zeros((channel_loc.shape[0],2))
    for i in range(0,channel_loc.shape[0]):
        if channel_loc[i,0] >= center_lon:
            dist[i, 0] = 1000 * get_distance_hav(channel_loc[i, 1], channel_loc[i, 0], channel_loc[i, 1], center_lon)
        if channel_loc[i,0] < center_lon:
            dist[i, 0] = -1000 * get_distance_hav(channel_loc[i, 1], channel_loc[i, 0], channel_loc[i, 1], center_lon)
        if channel_loc[i,1] >= center_lat:
            dist[i, 1] = 1000 * get_distance_hav(channel_loc[i, 1], channel_loc[i, 0], center_lat, channel_loc[i, 0])
        if channel_loc[i,1] < center_lat:
            dist[i, 1] = -1000 * get_distance_hav(channel_loc[i, 1], channel_loc[i, 0], center_lat, channel_loc[i, 0])

    # print(dist)

    # Compute next power of 2
    NFFT = 2 ** int(np.log2(sample_number) + 1) + 1
    # Compute real FFT frequencies
    freqs = np.fft.rfftfreq(n=NFFT, d=1. / fs)
    # Parameter dictionary
    params = {
        # Source location grid (Nx, Ny)
        "grid_size": (N_theta, N_slow),
        # Sampling frequency of the waveforms
        "fsamp": fs,
        # Range of velocities (= 1/slowness)
        "v_range": (1500, 1500),
        # Range of azimuths (full circle in this case)
        "theta_range": (0, 2*np.pi),
    }
    # Dummy for station list
    stations = np.zeros(channel_loc.shape[0])

    # Instantiate beamformer and set parameters
    jazz = MUSIC()
    jazz.set_params(params)

    # Set station list (dummy)
    jazz.stations = stations # 源文件stations存储了lon lat dis数据，但这边只会用到长度，所以简化了
    # Construct the slowness space
    jazz.construct_slowness_grid()
    # Set the relative distances
    jazz.station_dist = dist

    # Compute the travel times for each slowness grid element
    jazz.construct_times_beamforming()
    print("Done")
    # ----------------------------------------------
    #  Beamforming ship track
    index_start = int(fs * start_time)
    index_stop = int(index_start + sample_number)
    # Select indices of frequencies that fall within the selected target band
    inds = (freqs >= freq_band[0]) & (freqs < freq_band[1])
    freqs_select = freqs[inds]
    jazz.precompute_A(freqs_select)
    Cxy = jazz.CMTM(
        waveform_filted[:, index_start:index_stop], Nw=2, freq_band=freq_band,
        fsamp=jazz.fsamp, scale=True, jit=True
    )
    # Project onto noise space of covariance matrix
    P = jazz.noise_space_projection(Cxy, sources=1, mode="MUSIC").reshape((N_theta, N_slow))
    source = 1/P
    theta = jazz.grid["grid_theta"]
    v = 1 / jazz.grid["grid_slow"]


    return theta, v, source


ship_name = 'CORAL ACROPORA'
channel2_selected = [2459,2469]
channel3_selected = [2470,2480]
standard_data_time = '2023-08-16 01:50:01'
# 转换成时间数组
timeArray_data = time.strptime(standard_data_time, "%Y-%m-%d %H:%M:%S")
# 转换成时间戳
timestamp_data = int(time.mktime(timeArray_data)) # unixtime
print('data start time is: ' + str(timestamp_data))
standard_start_time = '2023-08-16 01:50:30'
# 转换成时间数组
timeArray_start = time.strptime(standard_start_time, "%Y-%m-%d %H:%M:%S")
# 转换成时间戳
timestamp_start = int(time.mktime(timeArray_start)) # unixtime
print('start time is: ' + str(timestamp_start))
standard_end_time = '2023-08-16 01:53:50'

timeArray_end = time.strptime(standard_end_time, "%Y-%m-%d %H:%M:%S")

timestamp_end = int(time.mktime(timeArray_end)) # unixtime
print('end time is: ' + str(timestamp_end))






waveform1 = np.load(r"E:\DAS-ship-surveillance\data\CORAL ACROPORA2459-2469.npy")
waveform2 = np.load(r"E:\DAS-ship-surveillance\data\CORAL ACROPORA2470-2480.npy") # load DAS data of 2 selected segments here


channel_loc_file1 = r"E:\DAS-ship-surveillance\data\segment1_loc.csv" # channel location
channel_loc1 = pd.read_csv(channel_loc_file1)
channel_loc1 = np.asarray(channel_loc1)
center_lon1,center_lat1 = np.mean(channel_loc1, axis=0)
print('The current ref point1 -- lon: ' + str(center_lon1) + ' lat: ' + str(center_lat1))

channel_loc_file2 = r"E:\DAS-ship-surveillance\data\segment2_loc.csv" # channel location
channel_loc2 = pd.read_csv(channel_loc_file2)
channel_loc2 = np.asarray(channel_loc2)
center_lon2,center_lat2 = np.mean(channel_loc2, axis=0)
print('The current ref point2 -- lon: ' + str(center_lon2) + ' lat: ' + str(center_lat2))
print(channel_loc1)


center_lon = (sum(channel_loc1[:,0]) + sum(channel_loc2[:,0]) )/(channel_loc1.shape[0]+channel_loc2.shape[0])
center_lat = (sum(channel_loc1[:,1]) + sum(channel_loc2[:,1]) )/(channel_loc1.shape[0]+channel_loc2.shape[0])
print(center_lat)
print(center_lon)
DAS_dx = get_distance_hav(channel_loc1[0,1],channel_loc1[0,0],channel_loc1[0,1],channel_loc2[-1,0])
DAS_dy = get_distance_hav(channel_loc1[0,1],channel_loc2[-1,0],channel_loc2[-1,1],channel_loc2[-1,0])
DAS_orientation = np.pi/2 - atan(DAS_dy/DAS_dx)

# ship info
ship_file = r"E:\DAS-ship-surveillance\data\tanker_081417-081706.csv" # AIS ship information
ship_data = pd.read_csv(ship_file)
ship_data = np.asarray(ship_data)
ship_info = get_ship_info(ship_data,ship_name) # lon,lat,time

# theoretical azimuth
orientation = []
orie = []
time_point = []
time_point_index = []
degree = 180/np.pi

for i in range(0,ship_info.shape[0]):

    if ship_info[i,2] >= timestamp_start and ship_info[i,2]<= timestamp_end:
        if ship_info[i, 0] > center_lon and ship_info[i, 1] > center_lat:
            dx = get_distance_hav(center_lat, center_lon,center_lat,ship_info[i,0])
            dy = get_distance_hav(center_lat, ship_info[i,0],ship_info[i,1],ship_info[i,0])
            ori = 90 - atan(dy/dx)*degree
            angle = 90 - atan((ship_info[i,1] - center_lat)/(ship_info[i,0] - center_lon))*degree

        if ship_info[i, 0] > center_lon and ship_info[i, 1] < center_lat:
            dx = get_distance_hav(center_lat, center_lon,center_lat,ship_info[i,0])
            dy = get_distance_hav(center_lat, ship_info[i,0],ship_info[i,1],ship_info[i,0])
            ori = 90 + atan(dy/dx)*degree
            angle = 90 - atan((ship_info[i,1] - center_lat)/(ship_info[i,0] - center_lon))*degree

        if ship_info[i, 0] < center_lon and ship_info[i, 1] < center_lat:
            dx = get_distance_hav(center_lat, center_lon,center_lat,ship_info[i,0])
            dy = get_distance_hav(center_lat, ship_info[i,0],ship_info[i,1],ship_info[i,0])
            ori = 270 - atan(dy/dx)*degree
            angle = 270 - atan((ship_info[i,1] - center_lat)/(ship_info[i,0] - center_lon))*degree

        if ship_info[i, 0] < center_lon and ship_info[i, 1] > center_lat:
            dx = get_distance_hav(center_lat, center_lon,center_lat,ship_info[i,0])
            dy = get_distance_hav(center_lat, ship_info[i,0],ship_info[i,1],ship_info[i,0])
            ori = 270 + atan(dy/dx)*degree
            angle = 270 - atan((ship_info[i,1] - center_lat)/(ship_info[i,0] - center_lon))*degree

        orientation.append(ori)
        orie.append(angle)
        time_point.append(ship_info[i,2])
        time_point_index.append(i)

estimated_ori = []
bias = []
for i in range(0,len(time_point_index)):

    freq_band = [48.8,49.4]

    bf_time = 3
    select_index = i
    start_time = time_point[select_index] - timestamp_data
    print('start time is: ' + str(start_time))
    print('theoretical orientation is: ' + str(orientation[select_index]))
    theta, v, source2 = cal_source(waveform1,channel_loc1,start_time,freq_band,bf_time)

    theta, v, source3 = cal_source(waveform2,channel_loc2,start_time,freq_band,bf_time)
    source = 1/((1/source2 + 1/source3)/2)
    source -= source.min()
    source /= source.max()

    fig, axes = plt.subplots(
        nrows=1, ncols=1, figsize=(10, 8), subplot_kw={"projection": "polar"}
    )
    ax = axes
    # Maximum contour level to plot
    Pmax = 1
    # Minimum contour level for main panels
    Pmin = 0.8
    lvls = np.linspace(Pmin, Pmax, 50)

    CS = ax.contourf(theta, [0,1], source.T, lvls,cmap="GnBu")
    theoretical_orientation = orientation[select_index]/degree
    ax.plot([theoretical_orientation,theoretical_orientation],[0,1],color = 'red',linewidth = 3)
    ax.plot([DAS_orientation, DAS_orientation + np.pi], [1, 1], color='black', linewidth=3, ls = '--')

    # Fix to avoid PDF rendering artifacts
    for c in CS.collections:
        c.set_edgecolor("face")
    axins = inset_axes(
                ax, width='2%', height='100%', loc='lower left',
                bbox_to_anchor=(1.1, 0, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0
            )
    cb = Colorbar(mappable=CS, ax=axins, orientation="vertical")
    # cb.set_label('dB', fontsize=14)
    # ax.set_yticks(ticks=np.arange(1490, 1511, 10))
    # ax.set_yticklabels(labels=np.arange(1490, 1511, 10), color="r")
    ax.set_rlabel_position(-67.5)
    ax.tick_params(labelsize=15)

    # Zero azimuth points north, then rotates clock-wise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)


    ref_index = int((DAS_orientation + np.pi)*degree)
    if i < 7:
        if i == 4:
            max_index = np.where(source == np.max(source))
            estimation = np.mean(theta[max_index[0][:]]) * degree
            ref = DAS_orientation * degree + 180 # 对不同对象需确认是否加180 下半区需加
            if estimation > ref and orientation[select_index] < ref:
                estimation = ref - (estimation - ref)
            if estimation < ref and orientation[select_index] > ref:
                estimation = ref + (ref - estimation)
        else:
            max_index = np.where(source == np.max(source[:ref_index,0]))
            # print(max_index)
            estimation = np.mean(theta[max_index[0][:]]) * degree
            # print(theta[max_index[0][:]])
    else:
        max_index = np.where(source == np.max(source[ref_index:, 0]))
        estimation = np.mean(theta[max_index[0][:]]) * degree

    estimated_ori.append(estimation)
    bias_now = abs(estimation - orientation[select_index])
    bias.append(bias_now)

    ax.plot([estimation/degree, estimation/degree], [0, 1], color='red', linewidth=3, ls = '--')





# plot
average_bias = np.mean(bias)
print(len(bias))
print('average bias is: ' + str(average_bias))

time_plt = []
for i in range(0,len(time_point)):
    time_plt.append(time_point[i]-time_point[0])

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
ax = axes
ax.plot(time_plt,orientation, color = 'black', zorder = 1)
ax.scatter(time_plt,estimated_ori,marker = 'v',color = 'red', zorder = 2, s = 150)


ax.set_xlabel('time (s)',labelpad=0.5,fontsize=14)
ax.set_ylabel('orientation (degree)',labelpad=0.5,fontsize=14)
ax.set_title('average error = ' + str(average_bias) + 'degree',fontsize=14)
plt.show()
plt.close()