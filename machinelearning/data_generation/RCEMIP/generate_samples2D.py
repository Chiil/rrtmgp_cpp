##import matplotlib.pyplot as pl
import numpy as np
import netCDF4 as nc
#define constants
float_type = "f8"
ep         = 0.622
#skewed random numbers 
randskew = -np.log(1-np.random.random(1000000))
randskew /= np.max(randskew)
np.random.shuffle(randskew)
irand = -1

#Saturation specific humidity and interpolation functions
def fill_ncstring(nc_var,str_var):
    for i in range(len(str_var)):
        for j in range(len(str_var[i])):
            nc_var[i,j] = str_var[i][j]
        for j in range(len(str_var[i]),10):
            nc_var[i,j] = " "

def sample_v(vmin,vmax):
    return vmin + np.random.random(1) * (vmax-vmin) 

def sample_v_prof(p,p_prof,vmin_prof,vmax_prof): 
    vmin = np.interp(p,p_prof,vmin_prof) 
    vmax = np.interp(p,p_prof,vmax_prof) 
    if vmax/vmin >= 10. and np.random.random(1) > .5:
        global irand
        irand += 1
        return vmin + randskew[irand] * (vmax-vmin)
    else:
        return vmin + np.random.random(1) * (vmax-vmin)

############
# User defined fixes gas concentrations or gas profiles
############
gases_profiles = ['h2o','o3']
gases_constant = ['co2','n2o','ch4','o2','n2','cfc11','co','cf4']

def get_o3(p):    
    p_hpa = p / 100.
    return np.maximum(5e-9,3.6478 * p_hpa**0.83209 * np.exp(-p_hpa/11.3515) * 1e-6)

co2    = 397.546967e-6
ch4    = 1831.47094727e-9
n2o    = 326.98800659e-9
n2     = 0.78100002
o2     = 0.20900001
cfc11  = 233.0798645e-12
co     = 1.19999e-7
cf4    = 81.09249115e-12

############
# User defined values or profiles of minimum and maximum allowed temperature, h20 and possible other gas concentrations
############
data  = np.loadtxt("profiles.txt",skiprows=1)
pres  = data[::-1,0]
p_min = np.min(pres) - 100.
p_max = np.max(pres) + 100.
T_min = data[::-1,1] * 0.99
T_max = data[::-1,2] * 1.01
q_min = data[::-1,3] * 0.95
q_max = data[::-1,4] * 1.05
nz    = len(pres)

fullpres = np.zeros(len(pres)+2)
fullpres[1:-1] = pres
fullpres[0]  = fullpres[1]  + (fullpres[1] - fullpres[2])
fullpres[-1] = fullpres[-2] + (fullpres[-2]- fullpres[-3])

#### create data:
ncol = 1000
p_lay = np.zeros((nz,ncol))
T_lay = np.zeros((nz,ncol))
q_lay = np.zeros((nz,ncol))
o3_lay = np.zeros((nz,ncol))
for i in range(nz):
    for j in range(ncol):
        p_lay[i,j]   = sample_v(fullpres[i],fullpres[i+2])
p_lay = np.sort(p_lay,axis=0)
for i in range(nz):
    for j in range(ncol):
        T_lay[i,j]   = sample_v_prof(p_lay[i,j],pres,T_min,T_max)
        q_lay[i,j]   = sample_v_prof(p_lay[i,j],pres,q_min,q_max)

for i in range(nz):
    for j in range(ncol):
        o3_lay[i,j]  = get_o3(p_lay[i,j])

h2o_lay = q_lay/(ep-ep*q_lay)
h2o_lay = np.maximum(h2o_lay,5e-6)

#### get t_lev:
T_lev = np.zeros((nz+1,ncol))
T_lev[1:-1,:] = (T_lay[1:,:] + T_lay[:-1,:]) / 2.
T_lev[0,:]  = T_lay[0]  + (T_lay[0]-T_lev[1])
T_lev[-1,:] = T_lay[-1] + (T_lay[-1]-T_lev[-2])

#### create netcdf file
nc_file = nc.Dataset("samples_input.nc", mode="w", datamodel="NETCDF4", clobber=True)
nc_group_rad = nc_file.createGroup("radiation")

nc_group_rad.createDimension("Nchar", 10)
nc_group_rad.createDimension("Ncnst", len(gases_constant))
nc_group_rad.createDimension("Nprof", len(gases_profiles))
nc_v_cnst = nc_group_rad.createVariable("gas_const","S1",("Ncnst","Nchar"))
fill_ncstring(nc_v_cnst,gases_constant)
nc_v_prof = nc_group_rad.createVariable("gas_profs","S1",("Nprof","Nchar"))
fill_ncstring(nc_v_prof,gases_profiles)

nc_group_rad.createDimension("lay", nz)
nc_group_rad.createDimension("lev", nz+1)
nc_group_rad.createDimension("col", ncol)

nc_p_lay = nc_group_rad.createVariable("p_lay", float_type, ("lay","col"))
nc_p_lay[:] = p_lay[:]

nc_T_lay = nc_group_rad.createVariable("t_lay", float_type, ("lay","col"))
nc_T_lay[:] = T_lay[:]

nc_T_lev = nc_group_rad.createVariable("t_lev", float_type, ("lev","col"))
nc_T_lev[:] = T_lev[:]

nc_h2o = nc_group_rad.createVariable("h2o", float_type, ("lay","col"))
nc_h2o[:] = h2o_lay[:]

nc_o3 = nc_group_rad.createVariable("o3", float_type, ("lay","col"))
nc_o3[:] = o3_lay[:]

nc_CO2     = nc_group_rad.createVariable("co2", float_type, ("lay"))
nc_CH4     = nc_group_rad.createVariable("ch4", float_type, ("lay"))
nc_N2O     = nc_group_rad.createVariable("n2o", float_type, ("lay"))
nc_N2      = nc_group_rad.createVariable("n2" , float_type, ("lay"))
nc_O2      = nc_group_rad.createVariable("o2" , float_type, ("lay"))
nc_CFC11   = nc_group_rad.createVariable("cfc11" , float_type, ("lay"))
nc_CO      = nc_group_rad.createVariable("co" , float_type, ("lay"))
nc_CF4     = nc_group_rad.createVariable("cf4" , float_type, ("lay"))

nc_CO2[:]     = co2
nc_CH4[:]     = ch4
nc_N2O[:]     = n2o
nc_N2[:]      = n2
nc_O2[:]      = o2
nc_CFC11[:]   = cfc11
nc_CO[:]      = co
nc_CF4[:]     = cf4

nc_file.close()
