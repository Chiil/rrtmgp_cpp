#import matplotlib.pyplot as pl
import numpy as np
from shutil import copyfile
import netCDF4 as nc
Tsfc = 300
ep = 0.622
wh2o_min = 5e-6

def interp_to_lev(tlay):
    nt,nz,ny,nx = tlay.shape
    tlev = np.zeros((nt,nz+1,ny,nx))
    tlev[:,1:-1] = (tlay[:,1:]+tlay[:,:-1]) / 2.
    tlev[:,0] = 2 * tlay[:,0] - tlev[:,1]
    tlev[:,-1] = 2 * tlay[:,-1] - tlev[:,-2]
    return tlev

def get_o3(p):
    p_hpa = p / 100.
    return np.maximum(5e-9,3.6478 * p_hpa**0.83209 * np.exp(-p_hpa/11.3515) * 1e-6)

qt   = nc.Dataset("qt.nc").variables['qt'][:]
ql   = nc.Dataset("ql.nc").variables['ql'][:]
qi   = nc.Dataset("qi.nc").variables['qi'][:]

Tlay = nc.Dataset("T.nc").variables['T'][:]
Tlev = interp_to_lev(Tlay)
pfile1 = nc.Dataset("rcemip.default.0000000.nc").groups['thermo']
pfile2 = nc.Dataset("rcemip.default.0008640.nc").groups['thermo']
pres_lay = np.append(pfile1.variables['phydro'][::24],pfile2.variables['phydro'][::24],axis=0)
pres_lev = np.append(pfile1.variables['phydroh'][::24],pfile2.variables['phydroh'][::24],axis=0)
nt,nz,ny,nx = qt.shape

qv  = qt-(ql+qi)
h2o = qv/(ep-ep*qv)
h2o = np.maximum(h2o,wh2o_min) 

##########################
nc_file = nc.Dataset("test_rcemip_input2D.nc", mode="w", datamodel="NETCDF4", clobber=True)

float_type = "f8"
co2 =  397.546967e-6
ch4 = 1831.47094727e-9
o2 = 0.20900001
cfc11 = 233.0798645e-12
cf4 = 81.09249115e-12
n2o =  326.98800659e-9
n2 = 0.78100002
co =  1.19999e-7

Nprofs = 1000 
Lprofs = []
iprof  = 0

nset = 1
nc_file.createDimension("set",nset)
for i in range(1,nset+1):
    print("radiation"+str(i))
    nc_group_rad = nc_file.createGroup("radiation"+str(i))
    nc_group_rad.createDimension("lay", pres_lay.shape[1])
    nc_group_rad.createDimension("lev", pres_lev.shape[1])
    nc_group_rad.createDimension("col", Nprofs)
         
    nc_p_lay = nc_group_rad.createVariable("p_lay", float_type, ("lay","col"))
    nc_p_lev = nc_group_rad.createVariable("p_lev", float_type, ("lev","col"))

    nc_T_lay = nc_group_rad.createVariable("t_lay", float_type, ("lay","col"))
    nc_T_lev = nc_group_rad.createVariable("t_lev", float_type, ("lev","col"))
    nc_T_sfc = nc_group_rad.createVariable("t_sfc", float_type, ("col"))
    nc_CO2   = nc_group_rad.createVariable("co2", float_type, ("lay"))
    nc_CH4   = nc_group_rad.createVariable("ch4", float_type, ("lay"))
    nc_N2O   = nc_group_rad.createVariable("n2o", float_type, ("lay"))
    nc_O3    = nc_group_rad.createVariable("o3" , float_type, ("lay","col"))
    nc_H2O   = nc_group_rad.createVariable("h2o", float_type, ("lay","col"))
    nc_N2    = nc_group_rad.createVariable("n2" , float_type, ("lay"))
    nc_O2    = nc_group_rad.createVariable("o2" , float_type, ("lay"))
    nc_CFC11 = nc_group_rad.createVariable("cfc11" , float_type, ("lay"))
    nc_CO    = nc_group_rad.createVariable("co" , float_type, ("lay"))
    nc_CF4   = nc_group_rad.createVariable("cf4" , float_type, ("lay"))

    nc_CO2[:]   = co2
    nc_CH4[:]   = ch4
    nc_N2O[:]   = n2o
    nc_N2 [:]   = n2
    nc_O2 [:]   = o2
    nc_CFC11[:] = cfc11
    nc_CO[:]    = co
    nc_CF4[:]   = cf4
    nc_T_sfc[:] = Tsfc

    while iprof < Nprofs:
        it = np.random.randint(0,nt,size=(1))
        ix =  np.random.randint(0,nx,size=(1))
        iy =  np.random.randint(0,ny,size=(1))
        if (it,ix,iy) not in Lprofs:
            Lprofs += [(it,ix,iy)]
            
            nc_p_lay[:,iprof] = pres_lay[it,:].flatten()
            nc_p_lev[:,iprof] = pres_lev[it,:].flatten()
            nc_T_lay[:,iprof] = Tlay[it,:,iy,ix].flatten()
            nc_T_lev[:,iprof] = Tlev[it,:,iy,ix].flatten()
            nc_O3[:,iprof]    = get_o3(pres_lay[it,:]).flatten()
            nc_H2O[:,iprof]   = h2o[it,:,iy,ix].flatten()

            iprof += 1
nc_file.close()




