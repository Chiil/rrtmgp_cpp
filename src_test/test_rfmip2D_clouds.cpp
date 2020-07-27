/*
 * This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
 * and Rapid Radiative Transfer Model for GCM applications Parallel (RRTMGP).
 *
 * The original code is found at https://github.com/RobertPincus/rte-rrtmgp.
 *
 * Contacts: Robert Pincus and Eli Mlawer
 * email: rrtmgp@aer.com
 *
 * Copyright 2015-2019,  Atmospheric and Environmental Research and
 * Regents of the University of Colorado.  All right reserved.
 *
 * This C++ interface can be downloaded from https://github.com/microhh/rrtmgp_cpp
 *
 * Contact: Chiel van Heerwaarden
 * email: chiel.vanheerwaarden@wur.nl
 *
 * Copyright 2019, Wageningen University & Research.
 *
 * Use and duplication is permitted under the terms of the
 * BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
 *
 */

#include <boost/algorithm/string.hpp>
#include <cmath>

#include "Netcdf_interface.h"
#include "Array.h"
#include "Gas_concs.h"
#include "Gas_optics.h"
#include "Cloud_optics.h"
#include "Optical_props.h"
#include "Source_functions.h"
#include "Fluxes.h"
#include "Rte_lw.h"
#include "Rte_sw.h"
#include <time.h>
#include <sys/time.h>

#ifdef FLOAT_SINGLE_RRTMGP
#define FLOAT_TYPE float
#else
#define FLOAT_TYPE double
#endif
    double get_wall_time()
    {
        struct timeval time;
        if (gettimeofday(&time,NULL))
        {
            //  Handle error
            return 0;
        }
        return (double)time.tv_sec + (double)time.tv_usec * .000001;
    }

namespace
{
    double starttime,endtime;
    std::vector<std::string> get_variable_string(
            const std::string& var_name,
            std::vector<int> i_count,
            Netcdf_handle& input_nc,
            const int string_len,
            bool trim=true)
    {
        // Multiply all elements in i_count.
        int total_count = std::accumulate(i_count.begin(), i_count.end(), 1, std::multiplies<>());

        // Add the string length as the rightmost dimension.
        i_count.push_back(string_len);

        // Multiply all elements in i_count.
        // int total_count_char = std::accumulate(i_count.begin(), i_count.end(), 1, std::multiplies<>());

        // Read the entire char array;
        std::vector<char> var_char;
        var_char = input_nc.get_variable<char>(var_name, i_count);

        std::vector<std::string> var;

        for (int n=0; n<total_count; ++n)
        {
            std::string s(var_char.begin()+n*string_len, var_char.begin()+(n+1)*string_len);
            if (trim)
                boost::trim(s);
            var.push_back(s);
        }

        return var;
    }

    template<typename TF>
    Cloud_optics<TF> load_and_init_cloud_optics(
         Master& master,
         const Gas_concs<TF>& gas_concs,
    	 const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(master, coef_file, Netcdf_mode::Read);

        // Read look-up table coefficient dimensions
        int n_band     = coef_nc.get_dimension_size("nband");
        int n_rghice   = coef_nc.get_dimension_size("nrghice");
        int n_size_liq = coef_nc.get_dimension_size("nsize_liq");
        int n_size_ice = coef_nc.get_dimension_size("nsize_ice");

        Array<TF,2> band_lims_wvn(coef_nc.get_variable<TF>("bnd_limits_wavenumber", {n_band, 2}), {2, n_band});

        // Read look-up table constants.
        TF radliq_lwr = coef_nc.get_variable<TF>("radliq_lwr");
        TF radliq_upr = coef_nc.get_variable<TF>("radliq_upr");
        TF radliq_fac = coef_nc.get_variable<TF>("radliq_fac");

        TF radice_lwr = coef_nc.get_variable<TF>("radice_lwr");
        TF radice_upr = coef_nc.get_variable<TF>("radice_upr");
        TF radice_fac = coef_nc.get_variable<TF>("radice_fac");

        Array<TF,2> lut_extliq(
                coef_nc.get_variable<TF>("lut_extliq", {n_band, n_size_liq}), {n_size_liq, n_band});
        Array<TF,2> lut_ssaliq(
                coef_nc.get_variable<TF>("lut_ssaliq", {n_band, n_size_liq}), {n_size_liq, n_band});
        Array<TF,2> lut_asyliq(
                coef_nc.get_variable<TF>("lut_asyliq", {n_band, n_size_liq}), {n_size_liq, n_band});

        Array<TF,3> lut_extice(
                coef_nc.get_variable<TF>("lut_extice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});
        Array<TF,3> lut_ssaice(
                coef_nc.get_variable<TF>("lut_ssaice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});
        Array<TF,3> lut_asyice(
                coef_nc.get_variable<TF>("lut_asyice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});

        return Cloud_optics<TF>(
                band_lims_wvn,
                radliq_lwr, radliq_upr, radliq_fac,
                radice_lwr, radice_upr, radice_fac,
                lut_extliq, lut_ssaliq, lut_asyliq,
                lut_extice, lut_ssaice, lut_asyice);
    }
    
    template<typename TF>
    Gas_optics<TF> load_and_init_gas_optics(
            Master& master,
            const Gas_concs<TF>& gas_concs,
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(master, coef_file, Netcdf_mode::Read);

        // Read k-distribution information.
        int n_temps = coef_nc.get_dimension_size("temperature");
        int n_press = coef_nc.get_dimension_size("pressure");
        int n_absorbers = coef_nc.get_dimension_size("absorber");
        int n_char = coef_nc.get_dimension_size("string_len");
        int n_minorabsorbers = coef_nc.get_dimension_size("minor_absorber");
        int n_extabsorbers = coef_nc.get_dimension_size("absorber_ext");
        int n_mixingfracs = coef_nc.get_dimension_size("mixing_fraction");
        int n_layers = coef_nc.get_dimension_size("atmos_layer");
        int n_bnds = coef_nc.get_dimension_size("bnd");
        int n_gpts = coef_nc.get_dimension_size("gpt");
        int n_pairs = coef_nc.get_dimension_size("pair");
        int n_minor_absorber_intervals_lower = coef_nc.get_dimension_size("minor_absorber_intervals_lower");
        int n_minor_absorber_intervals_upper = coef_nc.get_dimension_size("minor_absorber_intervals_upper");
        int n_contributors_lower = coef_nc.get_dimension_size("contributors_lower");
        int n_contributors_upper = coef_nc.get_dimension_size("contributors_upper");

        // Read gas names.
        Array<std::string,1> gas_names(
                get_variable_string("gas_names", {n_absorbers}, coef_nc, n_char, true), {n_absorbers});

        Array<int,3> key_species(
                coef_nc.get_variable<int>("key_species", {n_bnds, n_layers, 2}),
                {2, n_layers, n_bnds});
        Array<TF,2> band_lims(coef_nc.get_variable<TF>("bnd_limits_wavenumber", {n_bnds, 2}), {2, n_bnds});
        Array<int,2> band2gpt(coef_nc.get_variable<int>("bnd_limits_gpt", {n_bnds, 2}), {2, n_bnds});
        Array<TF,1> press_ref(coef_nc.get_variable<TF>("press_ref", {n_press}), {n_press});
        Array<TF,1> temp_ref(coef_nc.get_variable<TF>("temp_ref", {n_temps}), {n_temps});

        TF temp_ref_p = coef_nc.get_variable<TF>("absorption_coefficient_ref_P");
        TF temp_ref_t = coef_nc.get_variable<TF>("absorption_coefficient_ref_T");
        TF press_ref_trop = coef_nc.get_variable<TF>("press_ref_trop");

        Array<TF,3> kminor_lower(
                coef_nc.get_variable<TF>("kminor_lower", {n_temps, n_mixingfracs, n_contributors_lower}),
                {n_contributors_lower, n_mixingfracs, n_temps});
        Array<TF,3> kminor_upper(
                coef_nc.get_variable<TF>("kminor_upper", {n_temps, n_mixingfracs, n_contributors_upper}),
                {n_contributors_upper, n_mixingfracs, n_temps});

        Array<std::string,1> gas_minor(get_variable_string("gas_minor", {n_minorabsorbers}, coef_nc, n_char),
                {n_minorabsorbers});

        Array<std::string,1> identifier_minor(
                get_variable_string("identifier_minor", {n_minorabsorbers}, coef_nc, n_char), {n_minorabsorbers});

        Array<std::string,1> minor_gases_lower(
                get_variable_string("minor_gases_lower", {n_minor_absorber_intervals_lower}, coef_nc, n_char),
                {n_minor_absorber_intervals_lower});
        Array<std::string,1> minor_gases_upper(
                get_variable_string("minor_gases_upper", {n_minor_absorber_intervals_upper}, coef_nc, n_char),
                {n_minor_absorber_intervals_upper});

        Array<int,2> minor_limits_gpt_lower(
                coef_nc.get_variable<int>("minor_limits_gpt_lower", {n_minor_absorber_intervals_lower, n_pairs}),
                {n_pairs, n_minor_absorber_intervals_lower});
        Array<int,2> minor_limits_gpt_upper(
                coef_nc.get_variable<int>("minor_limits_gpt_upper", {n_minor_absorber_intervals_upper, n_pairs}),
                {n_pairs, n_minor_absorber_intervals_upper});

        Array<int,1> minor_scales_with_density_lower(
                coef_nc.get_variable<int>("minor_scales_with_density_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<int,1> minor_scales_with_density_upper(
                coef_nc.get_variable<int>("minor_scales_with_density_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<int,1> scale_by_complement_lower(
                coef_nc.get_variable<int>("scale_by_complement_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<int,1> scale_by_complement_upper(
                coef_nc.get_variable<int>("scale_by_complement_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<std::string,1> scaling_gas_lower(
                get_variable_string("scaling_gas_lower", {n_minor_absorber_intervals_lower}, coef_nc, n_char),
                {n_minor_absorber_intervals_lower});
        Array<std::string,1> scaling_gas_upper(
                get_variable_string("scaling_gas_upper", {n_minor_absorber_intervals_upper}, coef_nc, n_char),
                {n_minor_absorber_intervals_upper});

        Array<int,1> kminor_start_lower(
                coef_nc.get_variable<int>("kminor_start_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<int,1> kminor_start_upper(
                coef_nc.get_variable<int>("kminor_start_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<TF,3> vmr_ref(
                coef_nc.get_variable<TF>("vmr_ref", {n_temps, n_extabsorbers, n_layers}),
                {n_layers, n_extabsorbers, n_temps});

        Array<TF,4> kmajor(
                coef_nc.get_variable<TF>("kmajor", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                {n_gpts, n_mixingfracs, n_press+1, n_temps});

        // Keep the size at zero, if it does not exist.
        Array<TF,3> rayl_lower;
        Array<TF,3> rayl_upper;

        if (coef_nc.variable_exists("rayl_lower"))
        {
            rayl_lower.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_upper.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_lower = coef_nc.get_variable<TF>("rayl_lower", {n_temps, n_mixingfracs, n_gpts});
            rayl_upper = coef_nc.get_variable<TF>("rayl_upper", {n_temps, n_mixingfracs, n_gpts});
        }

        // Is it really LW if so read these variables as well.
        if (coef_nc.variable_exists("totplnk"))
        {
            int n_internal_sourcetemps = coef_nc.get_dimension_size("temperature_Planck");

            Array<TF,2> totplnk(
                    coef_nc.get_variable<TF>( "totplnk", {n_bnds, n_internal_sourcetemps}),
                    {n_internal_sourcetemps, n_bnds});
            Array<TF,4> planck_frac(
                    coef_nc.get_variable<TF>("plank_fraction", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                    {n_gpts, n_mixingfracs, n_press+1, n_temps});

            // Construct the k-distribution.
            return Gas_optics<TF>(
                    gas_concs,
                    gas_names,
                    key_species,
                    band2gpt,
                    band_lims,
                    press_ref,
                    press_ref_trop,
                    temp_ref,
                    temp_ref_p,
                    temp_ref_t,
                    vmr_ref,
                    kmajor,
                    kminor_lower,
                    kminor_upper,
                    gas_minor,
                    identifier_minor,
                    minor_gases_lower,
                    minor_gases_upper,
                    minor_limits_gpt_lower,
                    minor_limits_gpt_upper,
                    minor_scales_with_density_lower,
                    minor_scales_with_density_upper,
                    scaling_gas_lower,
                    scaling_gas_upper,
                    scale_by_complement_lower,
                    scale_by_complement_upper,
                    kminor_start_lower,
                    kminor_start_upper,
                    totplnk,
                    planck_frac,
                    rayl_lower,
                    rayl_upper);
        }
        else
        {
            Array<TF,1> solar_src(
                    coef_nc.get_variable<TF>("solar_source", {n_gpts}), {n_gpts});

            return Gas_optics<TF>(
                    gas_concs,
                    gas_names,
                    key_species,
                    band2gpt,
                    band_lims,
                    press_ref,
                    press_ref_trop,
                    temp_ref,
                    temp_ref_p,
                    temp_ref_t,
                    vmr_ref,
                    kmajor,
                    kminor_lower,
                    kminor_upper,
                    gas_minor,
                    identifier_minor,
                    minor_gases_lower,
                    minor_gases_upper,
                    minor_limits_gpt_lower,
                    minor_limits_gpt_upper,
                    minor_scales_with_density_lower,
                    minor_scales_with_density_upper,
                    scaling_gas_lower,
                    scaling_gas_upper,
                    scale_by_complement_lower,
                    scale_by_complement_upper,
                    kminor_start_lower,
                    kminor_start_upper,
                    solar_src,
                    rayl_lower,
                    rayl_upper);
        }
        // End reading of k-distribution.
    }
}

void find_location(Array<int,1>  arr,const int N, int& fst, int& lst)
{
    int i = 1;
    while (arr({i}) == 0)
        i += 1;
    fst = i;
    i = N;
    while (arr({i}) == 0)
        i -= 1;
    lst = i;
}

template<typename TF>
void load_gas_concs(Gas_concs<TF>& gas_concs, Netcdf_group& rad_nc)
{
    // This part is contained in the create
    // Netcdf_group rad_nc = input_nc.get_group("radiation");

    const int n_lay = rad_nc.get_dimension_size("lay");
    const int n_col = rad_nc.get_dimension_size("col");

    gas_concs.set_vmr("h2o",
            Array<TF,2>(rad_nc.get_variable<TF>("h2o", {n_lay,n_col}), {n_col,n_lay}));
    gas_concs.set_vmr("co2",
            rad_nc.get_variable<TF>("co2"));
    gas_concs.set_vmr("o3",
            Array<TF,2>(rad_nc.get_variable<TF>("o3", {n_lay,n_col}), {n_col,n_lay}));
    gas_concs.set_vmr("n2o",
            rad_nc.get_variable<TF>("n2o"));

    gas_concs.set_vmr("ch4",
            rad_nc.get_variable<TF>("ch4"));
    gas_concs.set_vmr("o2",
            rad_nc.get_variable<TF>("o2"));
    gas_concs.set_vmr("n2",
            rad_nc.get_variable<TF>("n2"));
    gas_concs.set_vmr("co",
            rad_nc.get_variable<TF>("co"));

    gas_concs.set_vmr("cfc11",
            rad_nc.get_variable<TF>("cfc11"));
    gas_concs.set_vmr("cf4",
            rad_nc.get_variable<TF>("cf4"));

}

template<typename TF>
void solve_radiation(Master& master)
{
    const bool do_longwave = false;
    // These are the global variables that need to be contained in a class.
    Gas_concs<TF> gas_concs;

    std::unique_ptr<Gas_optics<TF>> kdist_lw;
    std::unique_ptr<Gas_optics<TF>> kdist_sw;
    std::unique_ptr<Cloud_optics<TF>> cloud_lw;
    std::unique_ptr<Cloud_optics<TF>> cloud_sw;

    // Input and output files:.
    Netcdf_file file_nc(master, "test_rcemip_input2D.nc", Netcdf_mode::Read);
    Netcdf_file output_file(master, "Flux+OptProp.nc", Netcdf_mode::Create);

    const int n_set = file_nc.get_dimension_size("set");
    std::string sets[n_set];
    for (int iset = 0; iset < n_set; ++iset)
        sets[iset] = "year"+std::to_string(2010+iset);
    
    Netcdf_group input_nc = file_nc.get_group(sets[0]);
    load_gas_concs<TF>(gas_concs, input_nc);
    kdist_lw = std::make_unique<Gas_optics<TF>>(
            load_and_init_gas_optics(master, gas_concs, "coefficients_lw.nc"));
    kdist_sw = std::make_unique<Gas_optics<TF>>(
            load_and_init_gas_optics(master, gas_concs, "coefficients_sw.nc"));
    cloud_lw = std::make_unique<Cloud_optics<TF>>(
	load_and_init_cloud_optics(master, gas_concs, "clouds_lw.nc"));
    cloud_sw = std::make_unique<Cloud_optics<TF>>(
	load_and_init_cloud_optics(master, gas_concs, "clouds_sw.nc"));
    
    for (int iset = 0; iset < n_set; ++iset)
    {
        std::cout<<"set: "<<sets[iset]<<std::endl;
        Netcdf_group input_nc = file_nc.get_group(sets[iset]);
        const int n_lay = input_nc.get_dimension_size("lay");
        const int n_lev = input_nc.get_dimension_size("lev");
        const int n_col = input_nc.get_dimension_size("col");
        int n_cvr = 1;      
        const int abnd = 14;
        load_gas_concs<TF>(gas_concs, input_nc);

        
        // Solve the full column once.
        Array<TF,3> aod(input_nc.get_variable<TF>("aod", {abnd,n_lay, n_col}), {n_col, n_lay,abnd});
        Array<TF,3> ssa(input_nc.get_variable<TF>("ssa", {abnd,n_lay, n_col}), {n_col, n_lay,abnd});
        Array<TF,3> asy(input_nc.get_variable<TF>("asy", {abnd,n_lay, n_col}), {n_col, n_lay,abnd});
        Array<TF,2> p_lay(input_nc.get_variable<TF>("p_lay", {n_lay, n_col}), {n_col, n_lay});
        Array<TF,2> t_lay(input_nc.get_variable<TF>("t_lay", {n_lay, n_col}), {n_col, n_lay});
        Array<TF,2> p_lev(input_nc.get_variable<TF>("p_lev", {n_lev, n_col}), {n_col, n_lev});
        Array<TF,2> t_lev(input_nc.get_variable<TF>("t_lev", {n_lev, n_col}), {n_col, n_lev});
        Array<TF,1> t_sfc(input_nc.get_variable<TF>("t_sfc", {n_col}), {n_col});
        Array<TF,1> mu0(input_nc.get_variable<TF>("cosza", {n_col}), {n_col}); //cos(zenith)

        // LOAD THE LONGWAVE SPECIFIC BOUNDARY CONDITIONS.
        Array<TF,2> c_lwp(input_nc.get_variable<TF>("lwp", {n_lay, n_col}), {n_col, n_lay});
        Array<TF,2> c_iwp(input_nc.get_variable<TF>("iwp", {n_lay, n_col}), {n_col, n_lay});
        Array<TF,2> c_rel(input_nc.get_variable<TF>("rel", {n_lay, n_col}), {n_col, n_lay});
        Array<TF,2> c_rei(input_nc.get_variable<TF>("rei", {n_lay, n_col}), {n_col, n_lay});
        Array<TF,2> c_cc (input_nc.get_variable<TF>("cc", {n_lay, n_col}), {n_col, n_lay});
        Array<TF,2> lw_totflux_up  ({n_col, n_lev});
        Array<TF,2> lw_totflux_dn  ({n_col, n_lev});
        Array<TF,2> sw_totflux_up  ({n_col, n_lev});
        Array<TF,2> sw_totflux_dn  ({n_col, n_lev});
        Array<TF,2> sw_totflux_dir ({n_col, n_lev});
       
        for (int ilay=1; ilay<=n_lay; ++ilay)
            for (int icol=1; icol<=n_col; ++icol)
                if (c_cc({icol,ilay})>0)
                {
                    n_cvr = 100;
                    break;
                }

        Array<TF,2> overlap_param({n_col,n_lay});
        for (int ilay=1; ilay<=n_lay; ++ilay)
            for (int icol=1; icol<=n_col; ++icol)
            {
                const TF dp = std::abs(p_lev({icol,ilay})-p_lev({icol,ilay+1}));
                const TF dz = dp*TF(287.04)*t_lay({icol,ilay})/(TF(9.81)*p_lay({icol,ilay}));
                overlap_param({icol,ilay}) = std::exp(-dz/TF(2000.));
                
            }
        
        for (int ilev=1; ilev<=n_lev; ++ilev)
            for (int icol=1; icol<=n_col; ++icol)
            {
                lw_totflux_up({icol, ilev})  = 0.;  
                lw_totflux_dn({icol, ilev})  = 0.;  
                sw_totflux_up({icol, ilev})  = 0.;  
                sw_totflux_dn({icol, ilev})  = 0.;  
                sw_totflux_dir({icol, ilev}) = 0.;  
            }

        Array<int,2> m_lwp({n_col, n_lay});
        Array<int,2> m_iwp({n_col, n_lay}); 
        Array<int,1> cloudmask({n_lay});

        const int n_bnd = kdist_lw->get_nband();
        Array<TF,2> emis_sfc({n_bnd, n_col});

        for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
            for (int icol=1; icol<=n_col; ++icol)
                emis_sfc({ibnd, icol}) = 1.;

        const int n_ang = 1;
        const int top_at_1 = p_lay({1, 1}) < p_lay({1, n_lay});
        // LOAD THE SHORTWAVE SPECIFIC BOUNDARY CONDITIONS.
        Array<TF,2> sfc_alb_dir({n_bnd, n_col});
        Array<TF,2> sfc_alb_dif({n_bnd, n_col});
        for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
            for (int icol=1; icol<=n_col; ++icol)
            {
                sfc_alb_dir({ibnd, icol}) = 0.25;
                sfc_alb_dif({ibnd, icol}) = 0.25;
             }

        Array<TF,2> col_dry({n_col, n_lay});
        if (input_nc.variable_exists("col_dry"))
            col_dry = input_nc.get_variable<TF>("col_dry", {n_lay, n_col});
        else
        {
            kdist_lw->get_col_dry(col_dry, gas_concs.get_vmr("h2o"), p_lev);
            kdist_sw->get_col_dry(col_dry, gas_concs.get_vmr("h2o"), p_lev);
        }
        const int n_gpt_lw = kdist_lw->get_ngpt();
        const int n_gpt_sw = kdist_sw->get_ngpt();

        for (int icvr=0; icvr<n_cvr; ++icvr)
        {
            //exponential random oberlap
            for (int icol=1; icol<=n_col; ++icol)   
            {
                for (int ilay=1; ilay<=n_lay; ++ilay)
                {
                    m_lwp({icol,ilay}) = 0;
                    m_iwp({icol,ilay}) = 0;
                }
                bool clouds_present = false;
                for (int ilay=1; ilay<=n_lay; ++ilay)
                {
                   if (c_cc({icol,ilay}) > 0)
                   {
                       cloudmask({ilay}) = 1;
                       clouds_present = true;
                   } else {
                       cloudmask({ilay}) = 0;
                   }
                }
                if (clouds_present)
                {
                    int cld_fst;
                    int cld_lst;
                    find_location(cloudmask,n_lay,cld_fst,cld_lst);       
                    
                    TF rnmb = rand()/TF(RAND_MAX); 
                    if (rnmb > (1-c_cc({icol,cld_fst})))
                    {
                        m_lwp({icol,cld_fst}) = c_lwp({icol,cld_fst})>1e-20;
                        m_iwp({icol,cld_fst}) = c_iwp({icol,cld_fst})>1e-20;
                    }        
                    for (int ilay=cld_fst+1; ilay<=cld_lst; ++ilay)
                    {
                        if (cloudmask({ilay}) == 1)
                        {
                            const TF rnmb_next = rand()/TF(RAND_MAX); 
                            if (cloudmask({ilay-1}) == 1)
                            {
                                const TF crho = overlap_param({icol,ilay-1});
                                rnmb = crho*(rnmb-TF(0.5))+std::pow((TF(1)-crho*crho),TF(0.5))*
                                        (rnmb_next-TF(0.5))+TF(0.5);
                            } else {
                                rnmb = rnmb_next;
                            }
                            cloudmask({ilay}) = rnmb > (1-c_cc({icol,ilay})); 
                        }
                        if (cloudmask({ilay}) == 1)
                        {
                            m_lwp({icol,ilay}) = c_lwp({icol,ilay})>1e-20;
                            m_iwp({icol,ilay}) = c_iwp({icol,ilay})>1e-20;
                        }
                    }

                } 
            } 

            std::unique_ptr<Fluxes_broadband<TF>> fluxes =
                std::make_unique<Fluxes_broadband<TF>>(n_col, n_lev);
    
            Array<TF,2> lw_flux_up ({n_col, n_lev});
            Array<TF,2> lw_flux_dn ({n_col, n_lev});
            if (do_longwave)
            {
                std::unique_ptr<Optical_props_arry<TF>> optical_props_lw =
                        std::make_unique<Optical_props_1scl<TF>>(n_col, n_lay, *kdist_lw);
                std::unique_ptr<Optical_props_1scl<TF>> cloud_optical_props_lw =
                        std::make_unique<Optical_props_1scl<TF>>(n_col, n_lay, *cloud_lw);
                Source_func_lw<TF> sources(n_col, n_lay, *kdist_lw);
        
                kdist_lw->gas_optics(
                        p_lay,
                        p_lev,
                        t_lay,
                        t_sfc,
                        gas_concs,
                        optical_props_lw,
                        sources,
                        col_dry,
                        t_lev);

                if (n_cvr > 1)
                {
                    cloud_lw->cloud_optics(
                            m_lwp, m_iwp,
                            c_lwp, c_iwp,
                            c_rel, c_rei,
                            *cloud_optical_props_lw);
                    
                    add_to(dynamic_cast<Optical_props_1scl<TF>&>(*optical_props_lw),
                           dynamic_cast<Optical_props_1scl<TF>&>(*cloud_optical_props_lw));
    
                }
                
                Array<TF,3> lw_gpt_flux_up({n_col, n_lev, n_gpt_lw});
                Array<TF,3> lw_gpt_flux_dn({n_col, n_lev, n_gpt_lw});

                Rte_lw<TF>::rte_lw(
                        optical_props_lw,
                        top_at_1,
                        sources,
                        emis_sfc,
                        Array<TF,2>(), // Add an empty array, no inc_flux.
                        lw_gpt_flux_up,
                        lw_gpt_flux_dn,
                        n_ang);
                
                fluxes->reduce(
                        lw_gpt_flux_up, lw_gpt_flux_dn,
                        optical_props_lw, top_at_1);

                // Copy the data to the output.
                for (int ilev=1; ilev<=n_lev; ++ilev)
                    for (int icol=1; icol<=n_col; ++icol)
                    {   
                        lw_flux_up ({icol, ilev}) = fluxes->get_flux_up ()({icol, ilev});
                        lw_flux_dn ({icol, ilev}) = fluxes->get_flux_dn ()({icol, ilev});
                    }
            }


            Array<TF,2> toa_src({n_col, n_gpt_sw});
            std::unique_ptr<Optical_props_arry<TF>> optical_props_sw =
                    std::make_unique<Optical_props_2str<TF>>(n_col, n_lay, *kdist_sw);
            std::unique_ptr<Optical_props_2str<TF>> cloud_optical_props_sw =
                    std::make_unique<Optical_props_2str<TF>>(n_col, n_lay, *cloud_sw);
            
            kdist_sw->gas_optics(
                    p_lay,
                    p_lev,
                    t_lay,
                    gas_concs,
                    optical_props_sw,
                    toa_src,
                    col_dry);
            
            if (n_cvr > 1)
            {
                cloud_sw->cloud_optics(
                    m_lwp, m_iwp,
                    c_lwp, c_iwp,
                    c_rel, c_rei,
                    *cloud_optical_props_sw);
            
                add_to(dynamic_cast<Optical_props_2str<TF>&>(*optical_props_sw),
                       dynamic_cast<Optical_props_2str<TF>&>(*cloud_optical_props_sw));
            }

            Array<TF,3> sw_gpt_flux_up    ({n_col, n_lev, n_gpt_sw});
            Array<TF,3> sw_gpt_flux_dn    ({n_col, n_lev, n_gpt_sw});
            Array<TF,3> sw_gpt_flux_dn_dir({n_col, n_lev, n_gpt_sw});

            Rte_sw<TF>::rte_sw(
                    optical_props_sw,
                    top_at_1,
                    mu0,
                    toa_src,
                    sfc_alb_dir,
                    sfc_alb_dif,
                    Array<TF,2>(), // Add an empty array, no inc_flux.
                    sw_gpt_flux_up,
                    sw_gpt_flux_dn,
                    sw_gpt_flux_dn_dir);

            fluxes->reduce(
                    sw_gpt_flux_up, sw_gpt_flux_dn, sw_gpt_flux_dn_dir,
                    optical_props_sw, top_at_1);

            Array<TF,2> sw_flux_up ({n_col, n_lev});
            Array<TF,2> sw_flux_dn ({n_col, n_lev});
            Array<TF,2> sw_flux_dir({n_col, n_lev});

            // Copy the data to the output.
            for (int ilev=1; ilev<=n_lev; ++ilev)
                for (int icol=1; icol<=n_col; ++icol)
                {   
                    sw_flux_up ({icol, ilev}) = fluxes->get_flux_up ()({icol, ilev});
                    sw_flux_dn ({icol, ilev}) = fluxes->get_flux_dn ()({icol, ilev});
                    sw_flux_dir({icol, ilev}) = fluxes->get_flux_dn_dir()({icol, ilev});
                }

            const int gptpband = 16;
            for (int ilev=1; ilev<=n_lev; ++ilev)
                for (int icol=1; icol<=n_col; ++icol)
                {
                    sw_totflux_up({icol,ilev})  += sw_flux_up({icol,ilev})  * (1. / float(n_cvr));
                    sw_totflux_dn({icol,ilev})  += sw_flux_dn({icol,ilev})  * (1. / float(n_cvr));
                    sw_totflux_dir({icol,ilev}) += sw_flux_dir({icol,ilev}) * (1. / float(n_cvr));
                    if (do_longwave)
                    {
                        lw_totflux_dn({icol,ilev})  += lw_flux_dn({icol,ilev})  * (1. / float(n_cvr));
                        lw_totflux_up({icol,ilev})  += lw_flux_up({icol,ilev})  * (1. / float(n_cvr));
                    }
                }
        }
        // Compute the heating rates.
        constexpr TF g = 9.80655;
        constexpr TF cp = 1005.;

        // Store the radiation fluxes to a file
        Netcdf_group output_nc = output_file.add_group(sets[iset]);
        output_nc.add_dimension("col", n_col);
        output_nc.add_dimension("lev", n_lev);
        output_nc.add_dimension("lay", n_lay);

        auto nc_p_lev = output_nc.add_variable<TF>("plev", {"lev","col"});
        auto nc_p_lay = output_nc.add_variable<TF>("play", {"lay","col"});
        nc_p_lev.insert(p_lev.v(), {0, 0});
        nc_p_lay.insert(p_lay.v(), {0, 0});
        
        auto nc_mu0 = output_nc.add_variable<TF>("mu0", {"col"});
        nc_mu0.insert(mu0.v(), {0});

        if (do_longwave)
        {
            auto nc_lw_flux_up  = output_nc.add_variable<TF>("lw_flux_up" , {"lev", "col"});
            auto nc_lw_flux_dn  = output_nc.add_variable<TF>("lw_flux_dn" , {"lev", "col"});
            nc_lw_flux_up .insert(lw_totflux_up .v(), {0, 0});
            nc_lw_flux_dn .insert(lw_totflux_dn .v(), {0, 0});
        }

        auto nc_sw_flux_up  = output_nc.add_variable<TF>("sw_flux_up" , {"lev", "col"});
        auto nc_sw_flux_dn  = output_nc.add_variable<TF>("sw_flux_dn" , {"lev", "col"});
        auto nc_sw_flux_dir = output_nc.add_variable<TF>("sw_flux_dir", {"lev", "col"});
        nc_sw_flux_up  .insert(sw_totflux_up.v(), {0, 0});
        nc_sw_flux_dn  .insert(sw_totflux_dn.v(), {0, 0});
        nc_sw_flux_dir .insert(sw_totflux_dir.v(), {0, 0});


    }
}

int main()
{
    Master master;
    try
    {
        master.start();
        master.init();

        solve_radiation<FLOAT_TYPE>(master);
    }

    // Catch any exceptions and return 1.
    catch (const std::exception& e)
    {
        master.print_message("EXCEPTION: %s\n", e.what());
        return 1;
    }
    catch (...)
    {
        master.print_message("UNHANDLED EXCEPTION!\n");
        return 1;
    }

    // Return 0 in case of normal exit.
    return 0;
}
