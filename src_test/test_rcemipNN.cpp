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
#include "Gas_opticsNN.h"
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
     double starttime;
     double endtime;
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
    Gas_opticsNN<TF> load_and_init_gas_opticsNNSW(
            Master& master,
            const Gas_concs<TF>& gas_concs,
            const std::string& coef_file,
            const bool do_taussa)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(master, coef_file, Netcdf_mode::Read);
        int n_absorbers = coef_nc.get_dimension_size("absorber");
        int n_char = coef_nc.get_dimension_size("string_len");
        int n_bnds = coef_nc.get_dimension_size("bnd");
        int n_gpts = coef_nc.get_dimension_size("gpt");
        Array<std::string,1> gas_names(
                get_variable_string("gas_names", {n_absorbers}, coef_nc, n_char, true), {n_absorbers});
        Array<TF,2> band_lims(coef_nc.get_variable<TF>("bnd_limits_wavenumber", {n_bnds, 2}), {2, n_bnds});
        Array<int,2> band2gpt(coef_nc.get_variable<int>("bnd_limits_gpt", {n_bnds, 2}), {2, n_bnds});
        Array<std::string,1> plan_files({4});
        if (do_taussa) 
        {
            plan_files({1}) = "TRTEngineOp_1_tsw.plan";
            plan_files({2}) = "TRTEngineOp_0_tsw.plan";
            plan_files({3}) = "TRTEngineOp_1_ssa.plan";
            plan_files({4}) = "TRTEngineOp_0_ssa.plan";
        }
        else
        {
            plan_files({1}) = "TRTEngineOp_1_ray.plan";
            plan_files({2}) = "TRTEngineOp_0_ray.plan";
            plan_files({3}) = "TRTEngineOp_1_abs.plan";
            plan_files({4}) = "TRTEngineOp_0_abs.plan";
        }
        Array<TF,1> solar_src(
                coef_nc.get_variable<TF>("solar_source", {n_gpts}), {n_gpts});
        return Gas_opticsNN<TF>(
                plan_files,
                gas_names,
                band2gpt,
                band_lims,
                solar_src,
                do_taussa);
    }

    template<typename TF>
    Gas_opticsNN<TF> load_and_init_gas_opticsNNLW(
            Master& master,
            const Gas_concs<TF>& gas_concs,
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(master, coef_file, Netcdf_mode::Read);
        int n_absorbers = coef_nc.get_dimension_size("absorber");
        int n_char = coef_nc.get_dimension_size("string_len");
        int n_bnds = coef_nc.get_dimension_size("bnd");
        int n_gpts = coef_nc.get_dimension_size("gpt");
        Array<std::string,1> gas_names(
                get_variable_string("gas_names", {n_absorbers}, coef_nc, n_char, true), {n_absorbers});
        Array<TF,2> band_lims(coef_nc.get_variable<TF>("bnd_limits_wavenumber", {n_bnds, 2}), {2, n_bnds});
        Array<int,2> band2gpt(coef_nc.get_variable<int>("bnd_limits_gpt", {n_bnds, 2}), {2, n_bnds});
        Array<std::string,1> plan_files({4});

        plan_files({1}) = "TRTEngineOp_1_tlw.plan";
        plan_files({2}) = "TRTEngineOp_0_tlw.plan";
        plan_files({3}) = "TRTEngineOp_1_plk.plan";
        plan_files({4}) = "TRTEngineOp_0_plk.plan";
        return Gas_opticsNN<TF>(
                plan_files,
                gas_names,
                band2gpt,
                band_lims);
    }

    std::vector<float> load_weights(Netcdf_group group)
    {
        return group.get_variable<float>("wgth1",{32,4});
    }
}

template<typename TF>
void load_gas_concs(Gas_concs<TF>& gas_concs, Netcdf_file& input_nc)
{
    // This part is contained in the create
    Netcdf_group rad_nc = input_nc.get_group("radiation");

    const int n_lay = rad_nc.get_dimension_size("lay");

    gas_concs.set_vmr("h2o",
            Array<TF,1>(rad_nc.get_variable<TF>("h2o", {n_lay}), {n_lay}));
    gas_concs.set_vmr("o3",
            Array<TF,1>(rad_nc.get_variable<TF>("o3", {n_lay}), {n_lay}));
}

template<typename TF>
void solve_radiation(Master& master)
{
    // We are doing a single column run.
    const int n_col = 1;
    constexpr int N_layI=4;
    constexpr int N_layIp=6;
    constexpr int N_lay1=64;
    constexpr int N_lay2=64;
    constexpr int N_layOlw=256;
    constexpr int N_layOlwp=768;
    constexpr int N_layOsw=224;
    Netcdf_file NcFile(master, "wgth.nc", Netcdf_mode::Read);


    // These are the global variables that need to be contained in a class.
    Gas_concs<TF> gas_concs;

    std::unique_ptr<Gas_opticsNN<TF>> kdist_lw;
    std::unique_ptr<Gas_opticsNN<TF>> kdist_sw;

    // This is the part that is done in the initialization.
    Netcdf_file file_nc(master, "test_rcemip_input.nc", Netcdf_mode::Read);

    load_gas_concs<TF>(gas_concs, file_nc);
    kdist_lw = std::make_unique<Gas_opticsNN<TF>>(
            load_and_init_gas_opticsNNLW(master, gas_concs, "coefficients_lw.nc"));
    kdist_sw = std::make_unique<Gas_opticsNN<TF>>(
            load_and_init_gas_opticsNNSW(master, gas_concs, "coefficients_sw.nc",true)); //true: predict SW with tau_ray and tau_abs
    // LOAD THE LONGWAVE SPECIFIC BOUNDARY CONDITIONS.
    // Set the surface temperature and emissivity.
    Array<TF,1> t_sfc({1});
    t_sfc({1}) = 300.;

    const int n_bnd = kdist_lw->get_nband();

    Array<TF,2> emis_sfc({n_bnd, 1});
    for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
        emis_sfc({ibnd, 1}) = 1.;

    const int n_ang = 1;

    // LOAD THE SHORTWAVE SPECIFIC BOUNDARY CONDITIONS.
    Array<TF,1> sza({n_col});
    Array<TF,2> sfc_alb_dir({n_bnd, n_col});
    Array<TF,2> sfc_alb_dif({n_bnd, n_col});

    sza({1}) = 0.7339109504636155;

    for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
    {
        sfc_alb_dir({ibnd, 1}) = 0.07;
        sfc_alb_dif({ibnd, 1}) = 0.07;
    }

    Array<TF,1> mu0({n_col});
    mu0({1}) = std::cos(sza({1}));

    // Solve the full column once.
    Netcdf_group input_nc = file_nc.get_group("radiation");

    const int n_lay = input_nc.get_dimension_size("lay");
    const int n_lev = input_nc.get_dimension_size("lev");

    Array<TF,2> p_lay(input_nc.get_variable<TF>("p_lay", {n_lay, n_col}), {n_col, n_lay});
    Array<TF,2> t_lay(input_nc.get_variable<TF>("t_lay", {n_lay, n_col}), {n_col, n_lay});
    Array<TF,2> p_lev(input_nc.get_variable<TF>("p_lev", {n_lev, n_col}), {n_col, n_lev});
    Array<TF,2> t_lev(input_nc.get_variable<TF>("t_lev", {n_lev, n_col}), {n_col, n_lev});
    //find index that defines border between upper and lower atmosphere
    int idx_tropo = 0;
    for (int i = 1; i <= n_lay; i++)
    {
       if (p_lay({1,i}) > 9948.4315)
       {
           idx_tropo += 1;
       }
    }
    int idxupper = n_lay-idx_tropo;
    Netcdf_group tlwnc = NcFile.get_group("TauLW");
    Network TLW(idx_tropo,idxupper,
                tlwnc.get_variable<float>("bias1_lower",{N_lay1}),
                tlwnc.get_variable<float>("bias2_lower",{N_lay2}),
                tlwnc.get_variable<float>("bias3_lower",{N_layOlw}),
                tlwnc.get_variable<float>("wgth1_lower",{N_lay1,  N_layI}),
                tlwnc.get_variable<float>("wgth2_lower",{N_lay2,  N_lay1}),
                tlwnc.get_variable<float>("wgth3_lower",{N_layOlw,N_lay2}),
                tlwnc.get_variable<float>("Fmean_lower",{N_layI}),
                tlwnc.get_variable<float>("Fstdv_lower",{N_layI}),
                tlwnc.get_variable<float>("Lmean_lower",{N_layOlw}),
                tlwnc.get_variable<float>("Lstdv_lower",{N_layOlw}),
                tlwnc.get_variable<float>("bias1_upper",{N_lay1}),
                tlwnc.get_variable<float>("bias2_upper",{N_lay2}),
                tlwnc.get_variable<float>("bias3_upper",{N_layOlw}),
                tlwnc.get_variable<float>("wgth1_upper",{N_lay1,  N_layI}),
                tlwnc.get_variable<float>("wgth2_upper",{N_lay2,  N_lay1}),
                tlwnc.get_variable<float>("wgth3_upper",{N_layOlw,N_lay2}),
                tlwnc.get_variable<float>("Fmean_upper",{N_layI}),
                tlwnc.get_variable<float>("Fstdv_upper",{N_layI}),
                tlwnc.get_variable<float>("Lmean_upper",{N_layOlw}),
                tlwnc.get_variable<float>("Lstdv_upper",{N_layOlw}),
                N_layOlw,N_layI);

    Netcdf_group plnck = NcFile.get_group("Planck3");
    Network PLK(idx_tropo,idxupper,
                plnck.get_variable<float>("bias1_lower",{N_lay1}),
                plnck.get_variable<float>("bias2_lower",{N_lay2}),
                plnck.get_variable<float>("bias3_lower",{N_layOlwp}),
                plnck.get_variable<float>("wgth1_lower",{N_lay1,  N_layIp}),
                plnck.get_variable<float>("wgth2_lower",{N_lay2,  N_lay1}),
                plnck.get_variable<float>("wgth3_lower",{N_layOlwp,N_lay2}),
                plnck.get_variable<float>("Fmean_lower",{N_layIp}),
                plnck.get_variable<float>("Fstdv_lower",{N_layIp}),
                plnck.get_variable<float>("Lmean_lower",{N_layOlwp}),
                plnck.get_variable<float>("Lstdv_lower",{N_layOlwp}),
                plnck.get_variable<float>("bias1_upper",{N_lay1}),
                plnck.get_variable<float>("bias2_upper",{N_lay2}),
                plnck.get_variable<float>("bias3_upper",{N_layOlwp}),
                plnck.get_variable<float>("wgth1_upper",{N_lay1,  N_layIp}),
                plnck.get_variable<float>("wgth2_upper",{N_lay2,  N_lay1}),
                plnck.get_variable<float>("wgth3_upper",{N_layOlwp,N_lay2}),
                plnck.get_variable<float>("Fmean_upper",{N_layIp}),
                plnck.get_variable<float>("Fstdv_upper",{N_layIp}),
                plnck.get_variable<float>("Lmean_upper",{N_layOlwp}),
                plnck.get_variable<float>("Lstdv_upper",{N_layOlwp}),
                N_layOlwp,N_layIp);

    // Solve the longwave first.
    std::unique_ptr<Optical_props_arry<TF>> optical_props_lw =
            std::make_unique<Optical_props_1scl<TF>>(n_col, n_lay, *kdist_lw);

    Source_func_lw<TF> sources(n_col, n_lay, *kdist_lw);
    starttime=get_wall_time();
    kdist_lw->gas_optics(TLW,PLK,
            p_lay,
            p_lev,
            t_lay,
            t_sfc,
            gas_concs,
            optical_props_lw,
            sources,
            t_lev);
    endtime = get_wall_time();
    std::cout<<"LONGWAVE "<<endtime-starttime<<std::endl;
    starttime=get_wall_time();
    kdist_lw->gas_optics(TLW,PLK,
            p_lay,
            p_lev,
            t_lay,
            t_sfc,
            gas_concs,
            optical_props_lw,
            sources,
            t_lev);
    endtime = get_wall_time();
    std::cout<<"LONGWAVE "<<endtime-starttime<<std::endl;

    std::unique_ptr<Fluxes_broadband<TF>> fluxes =
            std::make_unique<Fluxes_broadband<TF>>(n_col, n_lev);

    const int top_at_1 = p_lay({1, 1}) < p_lay({1, n_lay});

    const int n_gpt_lw = optical_props_lw->get_ngpt();
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

    Array<TF,2> lw_flux_up ({n_col, n_lev});
    Array<TF,2> lw_flux_dn ({n_col, n_lev});
    Array<TF,2> lw_flux_net({n_col, n_lev});
    Array<TF,2> lw_heating ({n_col, n_lay});

    // Copy the data to the output.
    for (int ilev=1; ilev<=n_lev; ++ilev)
    {
        lw_flux_up ({1, ilev}) = fluxes->get_flux_up ()({1, ilev});
        lw_flux_dn ({1, ilev}) = fluxes->get_flux_dn ()({1, ilev});
        lw_flux_net({1, ilev}) = fluxes->get_flux_net()({1, ilev});
    }
    //Short waveeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
    const int n_gpt_sw = kdist_sw->get_ngpt();
    Array<TF,2> toa_src({n_col, n_gpt_sw});
    Network SSA(idx_tropo,idxupper,
                tlwnc.get_variable<float>("bias1_lower",{N_lay1}),
                tlwnc.get_variable<float>("bias2_lower",{N_lay2}),
                tlwnc.get_variable<float>("bias3_lower",{N_layOlw}),
                tlwnc.get_variable<float>("wgth1_lower",{N_lay1,  N_layI}),
                tlwnc.get_variable<float>("wgth2_lower",{N_lay2,  N_lay1}),
                tlwnc.get_variable<float>("wgth3_lower",{N_layOlw,N_lay2}),
                tlwnc.get_variable<float>("Fmean_lower",{N_layI}),
                tlwnc.get_variable<float>("Fstdv_lower",{N_layI}),
                tlwnc.get_variable<float>("Lmean_lower",{N_layOlw}),
                tlwnc.get_variable<float>("Lstdv_lower",{N_layOlw}),
                tlwnc.get_variable<float>("bias1_upper",{N_lay1}),
                tlwnc.get_variable<float>("bias2_upper",{N_lay2}),
                tlwnc.get_variable<float>("bias3_upper",{N_layOlw}),
                tlwnc.get_variable<float>("wgth1_upper",{N_lay1,  N_layI}),
                tlwnc.get_variable<float>("wgth2_upper",{N_lay2,  N_lay1}),
                tlwnc.get_variable<float>("wgth3_upper",{N_layOlw,N_lay2}),
                tlwnc.get_variable<float>("Fmean_upper",{N_layI}),
                tlwnc.get_variable<float>("Fstdv_upper",{N_layI}),
                tlwnc.get_variable<float>("Lmean_upper",{N_layOlw}),
                tlwnc.get_variable<float>("Lstdv_upper",{N_layOlw}),
                N_layOlw,N_layI);

    Network TSW(idx_tropo,idxupper,
                tlwnc.get_variable<float>("bias1_lower",{N_lay1}),
                tlwnc.get_variable<float>("bias2_lower",{N_lay2}),
                tlwnc.get_variable<float>("bias3_lower",{N_layOlw}),
                tlwnc.get_variable<float>("wgth1_lower",{N_lay1,  N_layI}),
                tlwnc.get_variable<float>("wgth2_lower",{N_lay2,  N_lay1}),
                tlwnc.get_variable<float>("wgth3_lower",{N_layOlw,N_lay2}),
                tlwnc.get_variable<float>("Fmean_lower",{N_layI}),
                tlwnc.get_variable<float>("Fstdv_lower",{N_layI}),
                tlwnc.get_variable<float>("Lmean_lower",{N_layOlw}),
                tlwnc.get_variable<float>("Lstdv_lower",{N_layOlw}),
                tlwnc.get_variable<float>("bias1_upper",{N_lay1}),
                tlwnc.get_variable<float>("bias2_upper",{N_lay2}),
                tlwnc.get_variable<float>("bias3_upper",{N_layOlw}),
                tlwnc.get_variable<float>("wgth1_upper",{N_lay1,  N_layI}),
                tlwnc.get_variable<float>("wgth2_upper",{N_lay2,  N_lay1}),
                tlwnc.get_variable<float>("wgth3_upper",{N_layOlw,N_lay2}),
                tlwnc.get_variable<float>("Fmean_upper",{N_layI}),
                tlwnc.get_variable<float>("Fstdv_upper",{N_layI}),
                tlwnc.get_variable<float>("Lmean_upper",{N_layOlw}),
                tlwnc.get_variable<float>("Lstdv_upper",{N_layOlw}),
                N_layOlw,N_layI);

    std::unique_ptr<Optical_props_arry<TF>> optical_props_sw =
            std::make_unique<Optical_props_2str<TF>>(n_col, n_lay, *kdist_sw);
    starttime = get_wall_time();
    kdist_sw->gas_optics(
            p_lay,
            p_lev,
            t_lay,
            gas_concs,
            optical_props_sw,
            toa_src,
            SSA,
            TSW);
    endtime = get_wall_time();
    std::cout<<"shortwave: "<<endtime-starttime<<std::endl;


    const TF tsi_scaling = 0.4053176301654965;
    for (int igpt=1; igpt<=n_gpt_sw; ++igpt)
        toa_src({1, igpt}) *= tsi_scaling;

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
    Array<TF,2> sw_flux_net({n_col, n_lev});
    Array<TF,2> sw_heating ({n_col, n_lay});

    // Copy the data to the output.
    for (int ilev=1; ilev<=n_lev; ++ilev)
    {
        sw_flux_up ({1, ilev}) = fluxes->get_flux_up ()({1, ilev});
        sw_flux_dn ({1, ilev}) = fluxes->get_flux_dn ()({1, ilev});
        sw_flux_net({1, ilev}) = fluxes->get_flux_net()({1, ilev});
    }

    // Compute the heating rates.
    constexpr TF g = 9.80655;
    constexpr TF cp = 1005.;

    Array<TF,2> heating ({n_col, n_lay});

    for (int ilay=1; ilay<=n_lay; ++ilay)
    {
        lw_heating({1, ilay}) =
                ( lw_flux_up({1, ilay+1}) - lw_flux_up({1, ilay})
                - lw_flux_dn({1, ilay+1}) + lw_flux_dn({1, ilay}) )
                * g / ( cp * (p_lev({1, ilay+1}) - p_lev({1, ilay})) ) * 86400.;

        sw_heating({1, ilay}) =
                ( sw_flux_up({1, ilay+1}) - sw_flux_up({1, ilay})
                - sw_flux_dn({1, ilay+1}) + sw_flux_dn({1, ilay}) )
                * g / ( cp * (p_lev({1, ilay+1}) - p_lev({1, ilay})) ) * 86400.;

        heating({1, ilay}) = lw_heating({1, ilay}) + sw_heating({1, ilay});
    }

    // Store the radiation fluxes to a file
    Netcdf_file output_nc(master, "test_rcemip_output.nc", Netcdf_mode::Create);
    output_nc.add_dimension("col", n_col);
    output_nc.add_dimension("lev", n_lev);
    output_nc.add_dimension("lay", n_lay);
    output_nc.add_dimension("gpt", n_gpt_lw);

    auto nc_p_lev = output_nc.add_variable<TF>("lev", {"lev"});
    auto nc_p_lay = output_nc.add_variable<TF>("lay", {"lay"});
    nc_p_lev.insert(p_lev.v(), {0});
    nc_p_lay.insert(p_lay.v(), {0});

    auto nc_lw_flux_up  = output_nc.add_variable<TF>("lw_flux_up" , {"lev", "col"});
    auto nc_lw_flux_dn  = output_nc.add_variable<TF>("lw_flux_dn" , {"lev", "col"});
    auto nc_lw_flux_net = output_nc.add_variable<TF>("lw_flux_net", {"lev", "col"});
    auto nc_lw_heating  = output_nc.add_variable<TF>("lw_heating" , {"lay", "col"});

    nc_lw_flux_up .insert(lw_flux_up .v(), {0, 0});
    nc_lw_flux_dn .insert(lw_flux_dn .v(), {0, 0});
    nc_lw_flux_net.insert(lw_flux_net.v(), {0, 0});
    nc_lw_heating .insert(lw_heating .v(), {0, 0});

    auto nc_sw_flux_up  = output_nc.add_variable<TF>("sw_flux_up" , {"lev", "col"});
    auto nc_sw_flux_dn  = output_nc.add_variable<TF>("sw_flux_dn" , {"lev", "col"});
    auto nc_sw_flux_net = output_nc.add_variable<TF>("sw_flux_net", {"lev", "col"});
    auto nc_sw_heating  = output_nc.add_variable<TF>("sw_heating" , {"lay", "col"});

    nc_sw_flux_up .insert(sw_flux_up .v(), {0, 0});
    nc_sw_flux_dn .insert(sw_flux_dn .v(), {0, 0});
    nc_sw_flux_net.insert(sw_flux_net.v(), {0, 0});
    nc_sw_heating .insert(sw_heating .v(), {0, 0});

    auto nc_heating = output_nc.add_variable<TF>("heating", {"lay", "col"});
    nc_heating.insert(heating.v(), {0, 0});
    auto nc_lw_gpt_flux_dn = output_nc.add_variable<TF>("lw_gpt_flux_dn" , {"gpt","lev", "col"});
    nc_lw_gpt_flux_dn.insert(lw_gpt_flux_dn.v(),{0,0,0});
    auto nc_lw_gpt_flux_up = output_nc.add_variable<TF>("lw_gpt_flux_up" , {"gpt","lev", "col"});
    nc_lw_gpt_flux_up.insert(lw_gpt_flux_up.v(),{0,0,0});
    
    
    Array<TF,3> myplk = sources.get_lay_source();
    auto planckout = output_nc.add_variable<TF>("planck" , {"gpt","lay", "col"});
    planckout.insert(myplk.v(),{0,0,0});
    Array<TF,3> myplkI= sources.get_lev_source_inc();
    auto planckoutI = output_nc.add_variable<TF>("planckI" , {"gpt","lev", "col"});
    planckoutI.insert(myplkI.v(),{0,0,0});

    Array<TF,3> myplkD = sources.get_lev_source_dec();
    auto planckoutD = output_nc.add_variable<TF>("planckD" , {"gpt","lev", "col"});
    planckoutD.insert(myplkD.v(),{0,0,0});


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
