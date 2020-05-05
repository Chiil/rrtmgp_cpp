#ifndef GAS_OPTICSNN_H
#define GAS_OPTICSNN_H

#include <string>
#include "Array.h"
#include <iostream>
//#include <argsParser.h>
//#include <buffers.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <Network.h>

#define restrict __restrict__
// Forward declarations.
template<typename TF> class Optical_props;
template<typename TF> class Optical_props_arry;
template<typename TF> class Gas_concs;
template<typename TF> class Source_func_lw;

//template<typename TF>
template<typename TF,int Nlayer,int N_gas,int N_lay1,int N_lay2,int N_lay3>
class Gas_opticsNN : public Optical_props<TF>
{
    public:
        // Constructor for longwave variant.
        Gas_opticsNN(
                const Array<std::string,1>& gas_names,
                const Array<int,2>& band2gpt,
                const Array<TF,2>& band_lims_wavenum);

        // Constructor for shortwave variant.
        Gas_opticsNN(
                const Array<std::string,1>& gas_names,
                const Array<int,2>& band2gpt,
                const Array<TF,2>& band_lims_wavenum,
                const Array<TF,1>& solar_src,
                const bool do_taussa);

        // Longwave variant.
        void gas_optics(
                Network<Nlayer,N_lay1,N_lay2,N_lay3>& TLW,
                Network<Nlayer,N_lay1,N_lay2,N_lay3>& PLK,
                const Array<TF,2>& play,
                const Array<TF,2>& plev,
                const Array<TF,2>& tlay,
                const Array<TF,1>& tsfc,
                const Gas_concs<TF>& gas_desc,
                std::unique_ptr<Optical_props_arry<TF>>& optical_props,
                Source_func_lw<TF>& sources,
                const Array<TF,2>& tlev,
                const int idx_tropo,
                const bool lower_atm, const bool upper_atm) const;

        // Shortwave variant.
        void gas_optics(
                Network<Nlayer,N_lay1,N_lay2,N_lay3>& SSA,
                Network<Nlayer,N_lay1,N_lay2,N_lay3>& TSW,
                const Array<TF,2>& play,
                const Array<TF,2>& plev,
                const Array<TF,2>& tlay,
                const Gas_concs<TF>& gas_desc,
                std::unique_ptr<Optical_props_arry<TF>>& optical_props,
                Array<TF,2>& toa_src,
                const int idx_tropo,
                const bool lower_atm, const bool upper_atm) const;

    private:
        const TF press_ref_trop = 9948.431564193395; //network is trained on this, so might as well hardcode it
        bool is_longwave;   
        bool do_taussa;
        Array<std::string,1> gas_names;
        Array<TF,1> solar_src;
        void compute_tau_ssa_NN(
                Network<Nlayer,N_lay1,N_lay2,N_lay3>& SSA,
                Network<Nlayer,N_lay1,N_lay2,N_lay3>& TSW,
                const int ncol, const int nlay, const int ngpt, const int nband, const int idx_tropo,
                const double* restrict const play,
                const double* restrict const plev,
                const double* restrict const tlay,
                const Gas_concs<TF>& gas_desc,
                std::unique_ptr<Optical_props_arry<TF>>& optical_props,
                const bool lower_atm, const bool upper_atm) const;

        void compute_tau_sources_NN(
                Network<Nlayer,N_lay1,N_lay2,N_lay3>& TLW,
                Network<Nlayer,N_lay1,N_lay2,N_lay3>& PLK,
                const int ncol, const int nlay, const int ngpt, const int nband, const int idx_tropo,
                const double* restrict const play, 
                const double* restrict const plev,
                const double* restrict const tlay, 
                const double* restrict const tlev,
                const Gas_concs<TF>& gas_desc,
                Source_func_lw<TF>& sources,
                std::unique_ptr<Optical_props_arry<TF>>& optical_props,
                const bool lower_atm, const bool upper_atm) const;

        void lay2sfc_factor(
                const Array<TF,2>& tlay,
                const Array<TF,1>& tsfc,
                Source_func_lw<TF>& sources,
                const int& ncol,
                const int& nlay,
                const int& nband) const;

};
#endif
