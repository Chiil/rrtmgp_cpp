#include <cmath>
#include <numeric>
#include <boost/algorithm/string.hpp>
#include "Gas_concs.h"
#include "Gas_opticsNN.h"
#include "Array.h"
#include "Optical_props.h"
#include "Source_functions.h"
#include "rrtmgp_kernels.h"
#include <time.h>
#include <sys/time.h>

#define restrict __restrict__

static const int Ninput = 4;     // number of input columns (water vapour, ozone, pressure, temperature)
static const int Ninput2 = Ninput + 2;

    double get_wall_time2()
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
    double starttimeX,endtimeX;
    inline float mylog(float x)
    {
        x = sqrt(x);x = sqrt(x);
        x = sqrt(x);x = sqrt(x);
        x = (x-1.0f) * 16.0f;
        return x;
    }
 
    void copy_arrays_tau(
                 const float* restrict const data_in,
                 const float* restrict const data_dp,
                 double* restrict const data_out,
                 const int N1,  const int N2a, 
                 const int N2b, const int N3,
                 const int nlay)
    {
        const float* dp_temp = &data_dp[N1*N2a];
        const int Nup = N2b-N2a;
        for (int i = 0; i < N3; ++i)
        {
            const int outidx = i*nlay*N1+N2a*N1;
            const float* in_temp = &data_in[i*Nup*N1];
            double* out_temp = &data_out[outidx];
            #pragma ivdep            
            for (int j = 0; j < N1*Nup; ++j)
            {
                out_temp[j] = in_temp[j] * dp_temp[j]; 
            }             
        }
    }

    void copy_arrays_ssa(
                 const float* restrict const data_in,
                 double* restrict const data_out,
                 const int N1,  const int N2a, 
                 const int N2b, const int N3,
                 const int nlay)
    {
        const int Nup = N2b-N2a;
        for (int i = 0; i < N3; ++i)
        {
            const int outidx = i*nlay*N1+N2a*N1;
            const float* in_temp = &data_in[i*Nup*N1];
            double* out_temp = &data_out[outidx];
            #pragma ivdep            
            for (int j = 0; j < N1*Nup; ++j)
            {
                out_temp[j] = in_temp[j]; 
            }             
        }
    }

    void copy_arrays_plk(
                 const float* restrict const data_in,
                 double* restrict const data_out1,
                 double* restrict const data_out2,
                 double* restrict const data_out3,
                 const int N1,  const int N2a, 
                 const int N2b, const int N3,
                 const int nlay)
    {
        const int Nup = N2b-N2a;
        for (int i = 0; i < N3; ++i)
        {
            const int outidx = i*nlay*N1+N2a*N1;
            const float* in_temp1 = &data_in[i*Nup*N1];
            const float* in_temp2 = &data_in[(i+N3)*Nup*N1];
            const float* in_temp3 = &data_in[(i+N3+N3)*Nup*N1];
            double* out_temp1= &data_out1[outidx];
            double* out_temp2= &data_out2[outidx];
            double* out_temp3= &data_out3[outidx];
            #pragma ivdep           
            for (int j = 0; j < N1*Nup; ++j)
            {
                out_temp1[j] = in_temp1[j];
                out_temp2[j] = in_temp2[j];
                out_temp3[j] = in_temp3[j];
            
            }
        }
    }

}
       
//     // Constructor of longwave variant.
template<typename TF>
Gas_opticsNN<TF>::Gas_opticsNN(
        const Array<std::string,1>& plan_files, //lower_tau, upper_tau, lower_planck, upper_planck
        const Array<std::string,1>& gas_names,
        const Array<int,2>& band2gpt,
        const Array<TF,2>& band_lims_wavenum):
            Optical_props<TF>(band_lims_wavenum, band2gpt)
{
    this->is_longwave = true;
    this->gas_names = gas_names;
}

// Constructor of the shortwave variant.
template<typename TF>
Gas_opticsNN<TF>::Gas_opticsNN(
        const Array<std::string,1>& plan_files, //lower_tau, upper_tau, lower_ssa, upper_ssa
        const Array<std::string,1>& gas_names,
        const Array<int,2>& band2gpt,
        const Array<TF,2>& band_lims_wavenum,
        const Array<TF,1>& solar_src,
        const bool do_taussa):
            Optical_props<TF>(band_lims_wavenum, band2gpt),
            solar_src(solar_src)
{ 
   
    this->is_longwave = false;   
    this->do_taussa = do_taussa;
    this->gas_names = gas_names;
}

// Gas optics solver longwave variant.
template<typename TF>
void Gas_opticsNN<TF>::gas_optics(Network& TLW,Network& PLK,
        const Array<TF,2>& play,
        const Array<TF,2>& plev,
        const Array<TF,2>& tlay,
        const Array<TF,1>& tsfc,
        const Gas_concs<TF>& gas_desc,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props,
        Source_func_lw<TF>& sources,
        const Array<TF,2>& tlev) const
{
    const int ncol = play.dim(1);
    const int nlay = play.dim(2);
    const int ngpt = this->get_ngpt();
    const int nband = this->get_nband();

    compute_tau_sources_NN(TLW, PLK,
            ncol, nlay, ngpt, nband,
            play.ptr(), plev.ptr(), 
            tlay.ptr(), tlev.ptr(),
            gas_desc, sources, 
            optical_props);   

    //fill surface sources  
    lay2sfc_factor(tlay,tsfc,sources,ncol,nlay,nband);   
}

// Gas optics solver shortwave variant.
template<typename TF>
void Gas_opticsNN<TF>::gas_optics(
        const Array<TF,2>& play,
        const Array<TF,2>& plev,
        const Array<TF,2>& tlay,
        const Gas_concs<TF>& gas_desc,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props,
        Array<TF,2>& toa_src,
        Network& SSA,
        Network& TSW) const

{   
    const int ncol = play.dim(1);
    const int nlay = play.dim(2);
    const int ngpt = this->get_ngpt();
    const int nband = this->get_nband();
    compute_tau_ssa_NN(
            SSA,TSW,
            ncol, nlay, ngpt, nband,
            play.ptr(), plev.ptr(), tlay.ptr(), 
            gas_desc, optical_props);

    // External source function is constant.
    for (int igpt=1; igpt<=ngpt; ++igpt)
        for (int icol=1; icol<=ncol; ++icol)
            toa_src({icol, igpt}) = this->solar_src({igpt});
}

template<typename TF>
void Gas_opticsNN<TF>::lay2sfc_factor(
        const Array<TF,2>& tlay,
        const Array<TF,1>& tsfc,
        Source_func_lw<TF>& sources,
        const int& ncol,
        const int& nlay,
        const int& nband) const
{
    Array<TF,1> sfc_factor({nband});
    Array<TF,3>& src_layer = sources.get_lay_source();
    Array<TF,2>& src_sfc   = sources.get_sfc_source();
    for (int icol=1; icol<=ncol; ++icol)
    {
        const float tempfrac = tsfc({icol})/tlay({icol,1});
        sfc_factor({1}) = 0.184351f*pow(tempfrac,3.0f) + 0.080502f*pow(tempfrac,2.0f) + 0.779973f*tempfrac - 0.044828f;
        sfc_factor({2}) = 1.456914f*pow(tempfrac,3.0f) - 2.434327f*pow(tempfrac,2.0f) + 2.690731f*tempfrac - 0.713335f;
        sfc_factor({3}) = 4.931766f*pow(tempfrac,3.0f) - 11.21031f*pow(tempfrac,2.0f) + 10.51688f*tempfrac - 3.238430f;
        sfc_factor({4}) = 8.450806f*pow(tempfrac,3.0f) - 20.69442f*pow(tempfrac,2.0f) + 19.35547f*tempfrac - 6.112041f;
        sfc_factor({5}) = 13.12718f*pow(tempfrac,3.0f) - 33.65488f*pow(tempfrac,2.0f) + 31.67371f*tempfrac - 10.14633f;
        sfc_factor({6}) = 23.45908f*pow(tempfrac,3.0f) - 63.12546f*pow(tempfrac,2.0f) + 60.28334f*tempfrac - 19.61765f;
        sfc_factor({7}) = exp(-4.896122f+4.896520f*tempfrac);
        sfc_factor({8}) = exp(-5.361318f+5.361619f*tempfrac);
        sfc_factor({9}) = exp(-6.048732f+6.049125f*tempfrac);
        sfc_factor({10})= exp(-6.796339f+6.796480f*tempfrac);
        sfc_factor({11})= exp(-7.640071f+7.640656f*tempfrac);
        sfc_factor({12})= exp(-9.083692f+9.084135f*tempfrac);
        sfc_factor({13})= exp(-10.21916f+10.21936f*tempfrac);
        sfc_factor({14})= exp(-10.96706f+10.96721f*tempfrac);
        sfc_factor({15})= exp(-11.88411f+11.88459f*tempfrac);
        sfc_factor({16})= exp(-13.55701f+13.55843f*tempfrac);
        for (int iband=1; iband<=nband; ++iband)
            for (int igpt=1; igpt<=16; ++igpt)
            {
                const int idxgpt = igpt + 16 * (iband-1);
                src_sfc({icol,idxgpt}) = sfc_factor({iband}) * src_layer({icol,1,idxgpt});
            }       
    }               
}
 
//Neural Network optical property function for shortwave
//Currently only implemented for atmospheric profilfes ordered bottom-first
template<typename TF>
void Gas_opticsNN<TF>::compute_tau_ssa_NN(
        Network& SSA, Network& TSW,
        const int ncol, const int nlay, const int ngpt, const int nband,
        const double* restrict const play,
        const double* restrict const plev,
        const double* restrict const tlay,
        const Gas_concs<TF>& gas_desc,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props) const
{
    double* tau = optical_props->get_tau().ptr();
    double* ssa = optical_props->get_ssa().ptr();
    
    const int batchSize=ncol*nlay;
    int idx_tropo = 0;
    float nul = 0.;
    float een = 1.;

    //find index that defines border between upper and lower atmosphere
    for (int i = 0; i < nlay; i++)
    {
       if (play[i] > this->press_ref_trop) {idx_tropo += 1;}
    }

    std::cout<<"idxtropo: "<<idx_tropo<<std::endl;

    float dp[ncol * nlay];
    for (int ilay=0; ilay<nlay; ++ilay)
        for (int icol=0; icol<ncol; ++icol)
        {
            const int dpidx = icol*nlay + ilay;
            const int plidx = icol*(nlay+1) + ilay;
            dp[dpidx] = abs(plev[plidx]-plev[plidx+1]);
        }

    //get gas concentrations
    const double* h2o = gas_desc.get_vmr(this->gas_names({1})).ptr();
    const double* o3  = gas_desc.get_vmr(this->gas_names({3})).ptr();

    //// Lower atmosphere:
    //fill input array  
    float input_lower[idx_tropo*Ninput];
    float output_lower_tau[idx_tropo*ngpt];
    float output_lower_ssa[idx_tropo*ngpt];
    starttime = get_wall_time2();
    for (int i = 0; i < idx_tropo; ++i)
    {
        const float val = mylog(h2o[i]);
        input_lower[i] = val;
    }

    int startidx = idx_tropo * 1;
    for (int i = 0; i < idx_tropo; ++i)
    {
        const float val = mylog(o3[i]);
        const int idx   = startidx + i;
        input_lower[idx] = val;
    }

    startidx = idx_tropo * 2;
    for (int i = 0; i < idx_tropo; ++i)
    {
        const float val = mylog(play[i]);
        const int idx   = startidx + i;
        input_lower[idx] = val;
    }

    startidx = idx_tropo * 3;
    for (int i = 0; i < idx_tropo; ++i)
    {
        const float val = tlay[i];
        const int idx   = startidx + i;
        input_lower[idx] = val;
    }


    endtime = get_wall_time2();
    std::cout<<"SW input time: "<<endtime-starttime<<std::endl;

    TSW.Inference(input_lower, output_lower_tau, 1,1,1); //lower atmosphere, exp(output), normalize input
    SSA.Inference(input_lower, output_lower_ssa, 1,0,0); //lower atmosphere, output, input already normalized);
   
    copy_arrays_ssa(output_lower_ssa,ssa,ncol,0,idx_tropo,ngpt,nlay);    
    copy_arrays_tau(output_lower_tau,dp,tau,ncol,0,idx_tropo,ngpt,nlay); 


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++     
    //// Upper atmosphere:
    //fill input array
    const int Nlayupper = batchSize - idx_tropo;
    float input_upper[(batchSize - idx_tropo)*Ninput];
    float output_upper_tau[(batchSize - idx_tropo)*ngpt];
    float output_upper_ssa[(batchSize - idx_tropo)*ngpt];

    starttime = get_wall_time2();
    for (int i = idx_tropo; i < batchSize; ++i)
    {
        const int idx = i - idx_tropo;
        const float val = mylog(h2o[i]);
        input_upper[idx] = val;
    }

    startidx = Nlayupper * 1;
    for (int i = idx_tropo; i < batchSize; ++i)
    {
        const float val = mylog(o3[i]);
        const int idx   = i - idx_tropo + startidx;
        input_upper[idx] = val;
    }

    startidx = Nlayupper * 2;
    for (int i = idx_tropo; i < batchSize; ++i)
    {
        const float val = mylog(play[i]);
        const int idx   = i - idx_tropo + startidx;
        input_upper[idx] = val;
    }

    startidx = Nlayupper * 3;
    for (int i = idx_tropo; i < batchSize; ++i)
    {
        const float val = tlay[i];
        const int idx   = i - idx_tropo + startidx;
        input_upper[idx] = val;
    }
    endtime = get_wall_time2();
    std::cout<<"SW input time: "<<endtime-starttime<<std::endl;

    TSW.Inference(input_upper, output_upper_tau, 0,1,1); //upper atmosphere, exp(output), normalize input
    SSA.Inference(input_upper, output_upper_ssa, 0,0,0); //upper atmosphere, output, input already normalized 
    
    copy_arrays_ssa(output_upper_ssa,ssa,ncol,idx_tropo,batchSize,ngpt,nlay);
    copy_arrays_tau(output_upper_tau,dp,tau,ncol,idx_tropo,batchSize,ngpt,nlay);

}


//Neural Network optical property function for longwave
//Currently only implemented for atmospheric profilfes ordered bottom-first
template<typename TF>
void Gas_opticsNN<TF>::compute_tau_sources_NN(Network& TLW,Network& PLK,
        const int ncol, const int nlay, const int ngpt, const int nband,
        const double* restrict const play,
        const double* restrict const plev,
        const double* restrict const tlay,
        const double* restrict const tlev,
        const Gas_concs<TF>& gas_desc,
        Source_func_lw<TF>& sources,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props) const
{
    double* tau = optical_props->get_tau().ptr();
    double* src_layer = sources.get_lay_source().ptr();
    double* src_lvinc = sources.get_lev_source_inc().ptr();
    double* src_lvdec = sources.get_lev_source_dec().ptr();

    const int batchSize=ncol*nlay;
    int idx_tropo = 0;
    float nul = 0.;
    float een = 1.;

    //find index that defines border between upper and lower atmosphere
    for (int i = 0; i < nlay; i++)
    {
       if (play[i] > this->press_ref_trop) {idx_tropo += 1;}
    }

    std::cout<<"idx_tropo: "<<idx_tropo<<std::endl;
    
    float dp[ncol * nlay];
    for (int ilay=0; ilay<nlay; ++ilay)
        for (int icol=0; icol<ncol; ++icol)
        {
            const int dpidx = icol*nlay + ilay;
            const int plidx = icol*(nlay+1) + ilay;
            dp[dpidx] = abs(plev[plidx]-plev[plidx+1]);
        }
    
    //get gas concentrations
    const double* h2o = gas_desc.get_vmr(this->gas_names({1})).ptr();
    const double* o3  = gas_desc.get_vmr(this->gas_names({3})).ptr();

    //// Lower atmosphere:
    //fill input array  
    float input_lower_tau[idx_tropo*Ninput];
    float input_lower_plk[idx_tropo*(Ninput+2)];
    float output_lower_tau[idx_tropo*ngpt];
    float output_lower_plk[idx_tropo*ngpt*3];

    for (int i = 0; i < idx_tropo; ++i)
    {
        const float val = mylog(h2o[i]);
        input_lower_tau[i] = val;
        input_lower_plk[i] = val;
    }
    int startidx = idx_tropo * 1;
    for (int i = 0; i < idx_tropo; ++i)
    {
        const float val = mylog(o3[i]);
        const int idx   = startidx + i;
        input_lower_tau[idx] = val;
        input_lower_plk[idx] = val;
    }

    startidx = idx_tropo * 2;
    for (int i = 0; i < idx_tropo; ++i)
    {
        const float val = mylog(play[i]);
        const int idx   = startidx + i;
        input_lower_tau[idx] = val;
        input_lower_plk[idx] = val;
    }

    startidx = idx_tropo * 3;
    for (int i = 0; i < idx_tropo; ++i)
    {
        const float val = tlay[i];
        const int idx   = startidx + i;
        input_lower_tau[idx] = val;
        input_lower_plk[idx] = val;
    }

    startidx = idx_tropo * 4;
    input_lower_plk[startidx] = tlev[0];
    for (int i = 0; i < idx_tropo-1; ++i)
    {
        const float val = tlev[i+1];
        const int idx1 = startidx+idx_tropo+i;
        const int idx2 = startidx+i+1;
        input_lower_plk[idx1] = val;
        input_lower_plk[idx2] = val;
    }
    input_lower_plk[idx_tropo * 6 - 1] = tlev[idx_tropo];

    TLW.Inference(input_lower_tau, output_lower_tau, 1,1,1); //lower atmosphere, exp(output), normalize input
    PLK.Inference(input_lower_plk, output_lower_plk, 1,1,1); //lower atmosphere, exp(output), normalize input
 
    copy_arrays_tau(output_lower_tau,dp,tau,ncol,0,idx_tropo,ngpt,nlay);
    copy_arrays_plk(output_lower_plk,src_layer,src_lvinc,src_lvdec,ncol,0,idx_tropo,ngpt,nlay);

    //// Upper atmosphere:
    //fill input array
    const int Nlayupper = batchSize - idx_tropo;
    float input_upper_tau[(Nlayupper)*Ninput];
    float input_upper_plk[(Nlayupper)*(Ninput+2)];
    float output_upper_tau[(Nlayupper)*ngpt];
    float output_upper_plk[(Nlayupper)*ngpt*3];   

    //do inference Optical Depth
    for (int i = idx_tropo; i < batchSize; ++i)
    {
        const int idx = i - idx_tropo;
        const float val = mylog(h2o[i]);
        input_upper_tau[idx] = val;
        input_upper_plk[idx] = val;
    }

    startidx = Nlayupper * 1;
    for (int i = idx_tropo; i < batchSize; ++i)
    {
        const int idx = i - idx_tropo + startidx;
        const float val = mylog(o3[i]);
        input_upper_tau[idx] = val;
        input_upper_plk[idx] = val;
    }

    startidx = Nlayupper * 2;
    for (int i = idx_tropo; i < batchSize; ++i)
    {
        const float val = mylog(play[i]);
        const int idx   = i - idx_tropo + startidx;
        input_upper_tau[idx] = val;
        input_upper_plk[idx] = val;
    }

    startidx = Nlayupper * 3;
    for (int i = idx_tropo; i < batchSize; ++i)
    {
        const float val = tlay[i];
        const int idx   = i - idx_tropo + startidx;
        input_upper_tau[idx] = val;
        input_upper_plk[idx] = val;
    }

    startidx = Nlayupper * 4;
    input_upper_plk[startidx] = tlev[idx_tropo];
    for (int i = idx_tropo; i < batchSize-1; ++i)
    {
        const float val = tlev[i+1];
        const int idx1 = startidx+Nlayupper+(i-idx_tropo);
        const int idx2 = startidx+1+(i-idx_tropo);
        input_upper_plk[idx1] = val;
        input_upper_plk[idx2] = val;
    }
    input_upper_plk[Nlayupper * 6 - 1] = tlev[batchSize];

    TLW.Inference(input_upper_tau, output_upper_tau, 0,1,1); //upper atmosphere, exp(output), normalize input
    PLK.Inference(input_upper_plk, output_upper_plk, 0,1,1); //upper atmosphere, exp(output), normalize input
 
    copy_arrays_tau(output_upper_tau,dp,tau,ncol,idx_tropo,batchSize,ngpt,nlay); 
    copy_arrays_plk(output_upper_plk,src_layer,src_lvinc,src_lvdec,ncol,idx_tropo,batchSize,ngpt,nlay); 
}




#ifdef FLOAT_SINGLE_RRTMGP
template class Gas_opticsNN<float>;
#else
template class Gas_opticsNN<double>;
#endif

