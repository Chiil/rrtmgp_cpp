#import matplotlib.pyplot as pl
import numpy as np
from shutil import copyfile
import netCDF4 as nc
#location of Cabauw output 
datapath = "/archive/mveerman/mainruns/RRTMG_coarse/"

cp  = 1004.
rd    = 287.04
pres0 = 1.e5

def interp_to_lev(tlay):
    tlev = np.zeros(tlay.shape[0]+1)
    tlev[1:-1] = (tlay[1:]+tlay[:-1]) / 2.
    tlev[0] = 2 * tlay[0] - tlev[1]
    tlev[-1] = 2 * tlay[-1] - tlev[-2]
    return tlev

def get_o3(p):
    p_hpa = p / 100.
    return np.maximum(5e-9,3.6478 * p_hpa**0.83209 * np.exp(-p_hpa/11.3515) * 1e-6)

nc_idx  = [(np.random.randint(0,12),np.random.randint(0,12)) for i in range(10)]
atmfiles = [nc.Dataset(datapath+"threedheating.%03d.%03d.700.nc"%(idx[0],idx[1])) for idx in nc_idx]
sfcfiles = [nc.Dataset(datapath+"crossAGS.x%03dy%03d.700.nc"%(idx[0],idx[1])) for idx in nc_idx]
nt,nz,ny,nx = atmfiles[0].variables['ql'][:].shape
inp_psfile = np.loadtxt("pressures.dat")

pres_lay = np.reshape(inp_psfile[:,1] * 100., (600,228))
pres_lev = np.reshape(inp_psfile[:,2] * 100. ,(600,228))
pres_lev = np.append(pres_lev,np.reshape(pres_lay[:,-1]*2 - pres_lev[:,-1],(600,1)),axis=1)

##########################
nc_file = nc.Dataset("test_rcemip_input2D.nc", mode="w", datamodel="NETCDF4", clobber=True)
float_type = "f8"
co2 =  397.546967e-6
ch4 = 1831.47094727e-9
o2 = 0.20900001
cf4 = 81.09249115e-12
n2o =  326.98800659e-9
n2 = 0.78100002
cfc11 = 233.0798645e-12
co =  1.19999e-7

# load a couple of ncfiles (one file = 1 core - 16*16 columns)
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
    print(nc_p_lay.shape)
    #random location
    while iprof < Nprofs:
        iatm = np.random.randint(0,len(atmfiles),size=(1))[0]
        it   = np.random.randint(0,nt,size=(1))[0]
        iy   = np.random.randint(0,ny,size=(1))[0]
        ix   = np.random.randint(0,nx,size=(1))[0]
        if (iatm,it,ix,iy) not in Lprofs:
            Lprofs += [(iatm,it,ix,iy)]
            qt = atmfiles[iatm].variables['qt'][it,:,iy,ix] * 1e-5
            ql = atmfiles[iatm].variables['ql'][it,:,iy,ix] * 1e-5
            qv = np.maximum(0,qt-ql)
            h2o = np.maximum(1.6e-5,qv/(.622-.622*qv))
            th = atmfiles[iatm].variables['thl'][it,:,iy,ix]
            o3 = get_o3(pres_lay[it,:])
            exnf= (pres_lay[it,:]/pres0)**(rd/cp) #6*i
            tlay = exnf * (th + (2.53e6/1004.) * ql / exnf)
            tlev = interp_to_lev(tlay)
            tsfc = sfcfiles[iatm].variables['tskin'][it,iy,ix]
            print(iprof,it)
            nc_p_lay[:,iprof] = np.transpose(pres_lay[it])
            nc_p_lev[:,iprof] = np.transpose(pres_lev[it])
            nc_T_lay[:,iprof] = np.transpose(tlay[:])
            nc_T_lev[:,iprof] = np.transpose(tlev[:])
            nc_T_sfc[iprof] = np.transpose(tsfc)
            nc_O3   [:,iprof] = np.transpose(o3[:])
            nc_H2O  [:,iprof] = np.transpose(h2o[:])
            iprof += 1

nc_file.close()

