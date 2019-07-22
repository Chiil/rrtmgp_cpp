import numpy as np
import netCDF4 as nc

float_type = "f8"

nc_file = nc.Dataset("test_rcemip_input.nc", mode="w", datamodel="NETCDF4", clobber=True)

# Radiation profiles.
z_top = 100.e3
dz = 5.
z  = np.arange(dz/2, z_top, dz)
zh = np.arange(   0, z_top-dz/2, dz)
zh = np.append(zh, z_top)

def calc_p_q_T(z):
    q_0 = 0.01864 # for 300 K SST.
    z_q1 = 4.0e3
    z_q2 = 7.5e3
    z_t = 15.e3
    q_t = 1.e-14

    q = q_0 * np.exp(-z/z_q1) * np.exp(-(z/z_q2)**2)

    # CvH hack to remove moisture jump.
    q_t = q_0 * np.exp(-z_t/z_q1) * np.exp(-(z_t/z_q2)**2)

    i_above_zt = np.where(z > z_t)
    q[i_above_zt] = q_t
    
    T_0 = 300.
    gamma = 6.7e-3
    Tv_0 = (1. + 0.608*q_0)*T_0
    Tv = Tv_0 - gamma*z
    Tv_t = Tv_0 - gamma*z_t
    Tv[i_above_zt] = Tv_t
    T = Tv / (1. + 0.608*q)
    
    g = 9.79764
    Rd = 287.04
    p0 = 101480.

    p = p0 * (Tv / Tv_0)**(g/(Rd*gamma))
    
    p_tmp = p0 * (Tv_t/Tv_0)**(g/(Rd*gamma)) \
          * np.exp( -( (g*(z-z_t)) / (Rd*Tv_t) ) )
    
    p[i_above_zt] = p_tmp[i_above_zt]

    return p, q, T

p_lay,   h2o, T_lay = calc_p_q_T( z)
p_lev, dummy, T_lev = calc_p_q_T(zh)
print("nlev",p_lay.shape)
print(np.argmin(np.where(p_lay<9948.4,0,1)))
co2 =  400.e-6
ch4 = 1831.47094727e-9
n2o =  326.98800659e-9
n2 = 0.78100002
o2 = 0.20900001
cfc11 = 233.0798645e-12
cfc12 = 520.58099365
hcfc22=229.542e-12
co =  1.19999e-7
cf4 = 81.09249115e-12
hfc143 = 15.25278091e-12
hfc134 = 80.51573181e-12

g1 = 3.6478
g2 = 0.83209
g3 = 11.3515
p_hpa = p_lay/100.
o3 = np.minimum(1.4e-5,np.maximum(8.6e-9,g1 * p_hpa**g2 * np.exp(-p_hpa/g3) * 1e-6))

nc_group_rad = nc_file.createGroup("radiation")

nc_group_rad.createDimension("lay", p_lay.size)
nc_group_rad.createDimension("lev", p_lev.size)

nc_z_lay = nc_group_rad.createVariable("z_lay", float_type, ("lay"))
nc_z_lev = nc_group_rad.createVariable("z_lev", float_type, ("lev"))
nc_z_lay[:] = z [:]
nc_z_lev[:] = zh[:]

nc_p_lay = nc_group_rad.createVariable("p_lay", float_type, ("lay"))
nc_p_lev = nc_group_rad.createVariable("p_lev", float_type, ("lev"))
nc_p_lay[:] = p_lay[:]
nc_p_lev[:] = p_lev[:]

nc_T_lay = nc_group_rad.createVariable("t_lay", float_type, ("lay"))
nc_T_lev = nc_group_rad.createVariable("t_lev", float_type, ("lev"))
nc_T_lay[:] = T_lay[:]
nc_T_lev[:] = T_lev[:]

nc_CO2 = nc_group_rad.createVariable("co2", float_type, ("lay"))
nc_CH4 = nc_group_rad.createVariable("ch4", float_type, ("lay"))
nc_N2O = nc_group_rad.createVariable("n2o", float_type, ("lay"))
nc_O3  = nc_group_rad.createVariable("o3" , float_type, ("lay"))
nc_H2O = nc_group_rad.createVariable("h2o", float_type, ("lay"))
nc_N2  = nc_group_rad.createVariable("n2" , float_type, ("lay"))
nc_O2  = nc_group_rad.createVariable("o2" , float_type, ("lay"))
nc_X1  = nc_group_rad.createVariable("cfc11" , float_type, ("lay"))
nc_X2  = nc_group_rad.createVariable("cfc12" , float_type, ("lay"))
nc_X3  = nc_group_rad.createVariable("hcfc22" , float_type, ("lay"))
nc_X4  = nc_group_rad.createVariable("co" , float_type, ("lay"))
nc_X5  = nc_group_rad.createVariable("cf4" , float_type, ("lay"))
nc_X5  = nc_group_rad.createVariable("hfc143a" , float_type, ("lay"))
nc_X6  = nc_group_rad.createVariable("hfc134a" , float_type, ("lay"))

nc_CO2[:] = co2
nc_CH4[:] = ch4
nc_N2O[:] = n2o
nc_O3 [:] = o3 [:]
nc_H2O[:] = h2o[:]
nc_N2 [:] = n2
nc_O2 [:] = o2
nc_X1 [:] = cfc11
nc_X2 [:] = cfc12
nc_X3 [:] = hcfc22
nc_X4 [:] = co
nc_X4 [:] = cf4
nc_X5 [:] = hfc143
nc_X6 [:] = hfc134


print(np.max(h2o))
nc_file.close()
