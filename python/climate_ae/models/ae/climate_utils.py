import os
import numpy as np
from netCDF4 import Dataset, num2date, date2num


def save_ncdf_file_high_res_prec(dat_array, years, months, days, fname, out_dir):
    ny, nx = (128, 128)
    lon = np.linspace(-13.3049926, 0.6650085, nx)
    lat = np.linspace(-8.744998, 5.225001, ny)
    months_str = []
    for i in range(len(months)):
        tmp = str(int(months[i]))
        if len(tmp) == 1:
            months_str.append("0"+tmp)
        elif len(tmp) == 2:
            months_str.append(tmp)
    days_str = []
    for i in range(len(days)):
        tmp = str(int(days[i]))
        if len(tmp) == 1:
            days_str.append("0"+tmp)
        elif len(tmp) == 2:
            days_str.append(tmp)
    dates = ['X'+str(int(i))+'.'+j+'.'+k for (i,j,k) in zip(years, months_str, days_str)]

    ncout = Dataset(os.path.join(out_dir, fname), 'w', 'NETCDF4') 
    ncout.createDimension('lon', nx)
    ncout.createDimension('lat', ny)
    ncout.createDimension('time', len(dates))
    lonvar = ncout.createVariable('lon', 'float32', ('lon'))
    lonvar[:] = lon
    latvar = ncout.createVariable('lat', 'float32', ('lat'))
    latvar[:] = lat
    timevar = ncout.createVariable('time', 'str', ('time'))
    timevar[:] = np.array(dates)
    myvar = ncout.createVariable('myvar', 'float32', ('time', 'lat','lon'))
    myvar[:] = dat_array[:,:,:,0]
    ncout.close()