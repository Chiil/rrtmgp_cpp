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

//#define restrict __restrict__

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
            play, plev, tlay, tlev,
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
            play, plev, tlay, gas_desc,
            optical_props);

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
        const Array<TF,2>& play,
        const Array<TF,2>& plev,
        const Array<TF,2>& tlay,
        const Gas_concs<TF>& gas_desc,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props) const
{
    Array<TF,3>& tau = optical_props->get_tau();
    Array<TF,3>& ssa = optical_props->get_ssa();
    const int batchSize=ncol*nlay;
    int idx_tropo = 0;
    float nul = 0.;
    float een = 1.;

    //find index that defines border between upper and lower atmosphere
    for (int i = 1; i <= nlay; i++)
    {
       if (play({1,i}) > this->press_ref_trop)
       {
           idx_tropo += 1;
       }
    }

    float dp[ncol][nlay];
    for (int ilay=1; ilay<=nlay; ++ilay)
        for (int icol=1; icol<=ncol; ++icol)
        {
            dp[icol-1][ilay-1] = abs(plev({icol,ilay})-plev({icol,ilay+1}));
        }

    //get gas concentrations
    const Array<TF,2>& h2o = gas_desc.get_vmr(this->gas_names({1}));
    const Array<TF,2>& o3  = gas_desc.get_vmr(this->gas_names({3}));

    //// Lower atmosphere: 
    //fill input array
    float input_lower[idx_tropo*Ninput];
    for (int i = 0; i < idx_tropo; i++)
    {
        input_lower[i*Ninput+0] = log(h2o({1,i+1}));
        input_lower[i*Ninput+1] = log(o3({1,i+1}));
        input_lower[i*Ninput+2] = log(play({1,i+1}));
        input_lower[i*Ninput+3] = tlay({1,i+1});
    }
    
    //do inference Optical Depth
    if (this->do_taussa)
    {
        float output_lower_tau[idx_tropo*ngpt];
        float output_lower_ssa[idx_tropo*ngpt];
        //TSW_lower.Inference(input_lower,output_lower_tau);
        //SSA_lower.Inference(input_lower,output_lower_ssa);
        for (int icol=1; icol<=ncol; ++icol)
             for (int ilay=1; ilay<=idx_tropo; ++ilay)
                {
                    const float delta_p = dp[icol-1][ilay-1];
                    for (int igpt=1; igpt<=ngpt; ++igpt)
                    {
                        const int idxlay = (igpt-1)+(ilay-1)*ngpt+(icol-1)*idx_tropo*ngpt;
                        tau({icol, ilay, igpt}) = output_lower_tau[idxlay]*delta_p;
                        ssa({icol, ilay, igpt}) = std::min(std::max(output_lower_ssa[idxlay],nul),een);
                    }   
                }
    } 
    else 
    {
        float output_lower_abs[idx_tropo*ngpt];
        float output_lower_ray[idx_tropo*ngpt];       
        for (int icol=1; icol<=ncol; ++icol)
             for (int ilay=1; ilay<=idx_tropo; ++ilay)
                {
                    const float delta_p = dp[icol-1][ilay-1];
                    for (int igpt=1; igpt<=ngpt; ++igpt)
                    {
                        const int idxlay = (igpt-1)+(ilay-1)*ngpt+(icol-1)*idx_tropo*ngpt;
                        const float tau_tot = output_lower_abs[idxlay] + output_lower_ray[idxlay];
                        tau({icol, ilay, igpt}) = tau_tot * delta_p;
                        ssa({icol, ilay, igpt}) = output_lower_ray[idxlay] / tau_tot;
                    }
                }
    }
    //// Upper atmosphere:
    //fill input array
    float input_upper[(batchSize - idx_tropo)*Ninput];
    for (int i = idx_tropo; i < batchSize; i++)
    {
        input_upper[Ninput*(i-idx_tropo)+0] = mylog(h2o({1,i+1}));
        input_upper[Ninput*(i-idx_tropo)+1] = mylog(o3({1,i+1}));
        input_upper[Ninput*(i-idx_tropo)+2] = mylog(play({1,i+1}));
        input_upper[Ninput*(i-idx_tropo)+3] = tlay({1,i+1});
    }

    if (do_taussa)
    {
    //do inference Optical Depth
        float output_upper_tau[(batchSize - idx_tropo)*ngpt];
        float output_upper_ssa[(batchSize - idx_tropo)*ngpt];
        //TSW_upper.Inference(input_upper,output_upper_tau);
        //SSA_upper.Inference(input_upper,output_upper_ssa);
        for (int icol=1; icol<=ncol; ++icol)
            for (int ilay=idx_tropo+1; ilay<=batchSize; ++ilay)
                {
                const float delta_p = dp[icol-1][ilay-1];
                for (int igpt=1; igpt<=ngpt; ++igpt)
                    {
                        const int idxlay = (igpt-1)+(ilay-1-idx_tropo)*ngpt+(icol-1)*(batchSize-idx_tropo)*ngpt;
                        tau({icol, ilay, igpt}) = output_upper_tau[idxlay]*delta_p;
                        ssa({icol, ilay, igpt}) = std::min(std::max(output_upper_ssa[idxlay],nul),een);
                    }
                }
    }
    else
    {
        //do inference Optical Depth
        float output_upper_abs[(batchSize - idx_tropo)*ngpt];
        float output_upper_ray[(batchSize - idx_tropo)*ngpt];   
        for (int icol=1; icol<=ncol; ++icol)
            for (int ilay=idx_tropo+1; ilay<=batchSize; ++ilay)
                {
                const float delta_p = dp[icol-1][ilay-1];
                for (int igpt=1; igpt<=ngpt; ++igpt)
                    {
                        const int idxlay = (igpt-1)+(ilay-1-idx_tropo)*ngpt+(icol-1)*(batchSize-idx_tropo)*ngpt;
                        const float tau_tot = output_upper_abs[idxlay] + output_upper_ray[idxlay];
                        tau({icol, ilay, igpt}) = tau_tot*delta_p;
                        ssa({icol, ilay, igpt}) = output_upper_ray[idxlay] / tau_tot;
                    }
                }
    }

}


//Neural Network optical property function for longwave
//Currently only implemented for atmospheric profilfes ordered bottom-first
template<typename TF>
void Gas_opticsNN<TF>::compute_tau_sources_NN(Network& TLW,Network& PLK,
        const int ncol, const int nlay, const int ngpt, const int nband,
        const Array<TF,2>& play,
        const Array<TF,2>& plev,
        const Array<TF,2>& tlay,
        const Array<TF,2>& tlev,
        const Gas_concs<TF>& gas_desc,
        Source_func_lw<TF>& sources,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props) const
{
    double* TAU = optical_props->get_tau().ptr();
    double* src_LAYER = sources.get_lay_source().ptr();
    double* src_LVINC = sources.get_lev_source_inc().ptr();
    double* src_LVDEC = sources.get_lev_source_dec().ptr();
    Array<TF,3>& tau = optical_props->get_tau();
    Array<TF,3>& src_layer = sources.get_lay_source();
    Array<TF,3>& src_lvinc = sources.get_lev_source_inc();
    Array<TF,3>& src_lvdec = sources.get_lev_source_dec();

    const int batchSize=ncol*nlay;
    int idx_tropo = 0;
    float nul = 0.;
    float een = 1.;
    //find index that defines border between upper and lower atmosphere
    for (int i = 1; i <= nlay; i++)
    {
       if (play({1,i}) > this->press_ref_trop)
       {
           idx_tropo += 1;
       }
    }
    //get gas concentrations
    const Array<TF,2>& h2o = gas_desc.get_vmr(this->gas_names({1}));
    const Array<TF,2>& o3  = gas_desc.get_vmr(this->gas_names({3}));

    //// Lower atmosphere:
    //fill input array  
    starttime = get_wall_time2();
    float input_lower_tau[idx_tropo*Ninput];
    float input_lower_plk[idx_tropo*(Ninput+2)];
    float output_lower_tau[idx_tropo*ngpt];
    float output_lower_plk[idx_tropo*ngpt*3];

    for (int i = 0; i < idx_tropo; ++i)
    {
        const float val = mylog(h2o({1,i+1}));
        input_lower_tau[i] = val;
        input_lower_plk[i] = val;
    }
    int startidx = idx_tropo * 1;
    for (int i = 0; i < idx_tropo; ++i)
    {
        const float val = mylog(o3({1,i+1}));
        const int idx   = startidx + i;
        input_lower_tau[idx] = val;
        input_lower_plk[idx] = val;
    }

    startidx = idx_tropo * 2;
    for (int i = 0; i < idx_tropo; ++i)
    {
        const float val = mylog(play({1,i+1}));
        const int idx   = startidx + i;
        input_lower_tau[idx] = val;
        input_lower_plk[idx] = val;
    }

    startidx = idx_tropo * 3;
    for (int i = 0; i < idx_tropo; ++i)
    {
        const float val = tlay({1,i+1});
        const int idx   = startidx + i;
        input_lower_tau[idx] = val;
        input_lower_plk[idx] = val;
    }

    startidx = idx_tropo * 4;
    input_lower_plk[startidx] = tlev({1,1});
    for (int i = 0; i < idx_tropo-1; ++i)
    {
        const float val = tlev({1,i+2});
        const int idx = startidx+2*i;
        input_lower_plk[idx + 1] = val;
        input_lower_plk[idx + 2] = val;
    }
    input_lower_plk[idx_tropo * 6 - 1] = tlev({1,idx_tropo+1});


    float dp[ncol* nlay];
    for (int ilay=1; ilay<=nlay; ++ilay)
        for (int icol=1; icol<=ncol; ++icol)
        {
           const int dpidx = (icol-1)*nlay + (ilay-1);
           dp[dpidx] = abs(plev({icol,ilay})-plev({icol,ilay+1}));
        }
    endtime = get_wall_time2();
    std::cout<<"input time: "<<endtime-starttime<<" "<<output_lower_tau[5]<<std::endl;
    
    starttime = get_wall_time2();
    TLW.Inference(input_lower_tau, output_lower_tau, 1);
    endtime = get_wall_time2();
    std::cout<<"NNsolver time: "<<endtime-starttime<<" "<<output_lower_tau[5]<<std::endl;

    starttime = get_wall_time2();
    PLK.Inference(input_lower_plk, output_lower_plk, 1);
    endtime = get_wall_time2();
    std::cout<<"NNsolver time: "<<endtime-starttime<<" "<<output_lower_tau[5]<<std::endl;

    starttime=get_wall_time2()                                                                             ;
    for (int igpt=1; igpt<=ngpt; ++igpt)
        for (int ilay=1; ilay<=idx_tropo; ++ilay)
            for (int icol=1; icol<=ncol; ++icol)
            {
                const int idxlay = (icol-1) + (ilay-1)*ncol + (igpt-1)*idx_tropo*ncol;
                const int dpidx = (icol-1)*nlay + (ilay-1);
                tau({icol, ilay, igpt}) = output_lower_tau[idxlay] * dp[dpidx];
            }
    std::cout<<"yeah "<<tau({1,1,1})<<" & "<<tau({1,3,6})<<std::endl;
    for (int igpt=1; igpt<=ngpt; ++igpt)
        for (int ilay=1; ilay<=idx_tropo; ++ilay)
            for (int icol=1; icol<=ncol; ++icol)
            {
                const int idxlay  = (icol-1) + (ilay-1) * ncol; 
                const int idxlay1 = idxlay + (igpt-1)          *idx_tropo * ncol;
                const int idxlay2 = idxlay + (igpt-1+ngpt)     *idx_tropo * ncol;
                const int idxlay3 = idxlay + (igpt-1+ngpt+ngpt)*idx_tropo * ncol;
                src_layer({icol, ilay, igpt}) = output_lower_plk[idxlay1];
                src_lvinc({icol, ilay, igpt}) = output_lower_plk[idxlay2];
                src_lvdec({icol, ilay, igpt}) = output_lower_plk[idxlay3];
            }
            

    endtime = get_wall_time2();
    std::cout<<"store time: "<<endtime-starttime<<" "<<tau({2,2,2})<<std::endl;
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
        const int ix = i - idx_tropo;
        const float val = mylog(h2o({1,i+1}));
        input_upper_tau[ix] = val;
        input_upper_plk[ix] = val;
    }

    startidx = Nlayupper * 1;
    for (int i = idx_tropo; i < batchSize; ++i)
    {
        const int idx = i - idx_tropo + startidx;
        const float val = mylog(o3({1,i+1}));
        input_upper_tau[i] = val;
        input_upper_plk[i] = val;
    }

    startidx = Nlayupper * 2;
    for (int i = idx_tropo; i < batchSize; ++i)
    {
        const float val = mylog(play({1,i+1}));
        const int idx   = i - idx_tropo + startidx;
        input_upper_tau[idx] = val;
        input_upper_plk[idx] = val;
    }

    startidx = Nlayupper * 3;
    for (int i = idx_tropo; i < batchSize; ++i)
    {
        const float val = tlay({1,i+1});
        const int idx   = i - idx_tropo + startidx;
        input_upper_tau[idx] = val;
        input_upper_plk[idx] = val;
    }

    startidx = Nlayupper * 4;
    input_upper_plk[startidx] = tlev({1,1+idx_tropo});
    for (int i = idx_tropo; i < batchSize-1; ++i)
    {
        const float val = tlev({1,i+2});
        const int idx = startidx+2*(i-idx_tropo);
        input_upper_plk[idx + 1] = val;
        input_upper_plk[idx + 2] = val;
    }
    input_upper_plk[Nlayupper * 6 - 1] = tlev({1,batchSize+1});

    starttime = get_wall_time2();
    TLW.Inference(input_upper_tau, output_upper_tau, 0);
    endtime = get_wall_time2();
    std::cout<<"Up_NNsolver time: "<<endtime-starttime<<" "<<output_lower_tau[5]<<std::endl;

    starttime = get_wall_time2();
    PLK.Inference(input_upper_plk, output_upper_plk, 0);
    endtime = get_wall_time2();
    std::cout<<"Up_NNsolver time: "<<endtime-starttime<<" "<<output_lower_tau[5]<<std::endl;

    starttime = get_wall_time2();
    for (int igpt=1; igpt<=ngpt; ++igpt)
       for (int ilay=idx_tropo + 1; ilay<=batchSize; ++ilay)
           for (int icol=1; icol<=ncol; ++icol)
          {
                const int idxlay = (icol-1) + (ilay-1-idx_tropo)*ncol + (igpt-1)*Nlayupper*ncol;
                const int idxtau = (icol-1) + (ilay-1)*ncol + (igpt-1)*batchSize*ncol;
                const int dpidx = (icol-1)*nlay + (ilay-1);
                TAU[idxtau] = output_upper_tau[idxlay] * dp[dpidx]*2;
            }
    endtime = get_wall_time2();
    std::cout<<"Up_NNoutputY time: "<<endtime-starttime<<" "<<tau({6,5000,1})<<" "<<TAU[5]<<std::endl;


    starttime = get_wall_time2();
    for (int igpt=1; igpt<=ngpt; ++igpt)
        for (int ilay=idx_tropo+1;ilay<=batchSize; ++ilay)
            for (int icol=1; icol<=ncol; ++icol)
            {
                const int idxplk = (icol-1) + (ilay-1)*ncol + (igpt-1)*batchSize*ncol;
                const int idxlay  = (icol-1) + (ilay-1-idx_tropo) * ncol; 
                const int idxlay1 = idxlay + (igpt-1)          *Nlayupper * ncol;
                const int idxlay2 = idxlay + (igpt-1+ngpt)     *Nlayupper * ncol;
                const int idxlay3 = idxlay + (igpt-1+ngpt+ngpt)*Nlayupper * ncol;
                src_LAYER[idxplk] = output_upper_plk[idxlay1];
                src_LVINC[idxplk] = output_upper_plk[idxlay2];
                src_LVDEC[idxplk] = output_upper_plk[idxlay3];
            }
    endtime = get_wall_time2();
    std::cout<<"Up_NNoutput timeP2: "<<endtime-starttime<<" "<<src_layer({1,5000,1})<<std::endl;


}





#ifdef FLOAT_SINGLE_RRTMGP
template class Gas_opticsNN<float>;
#else
template class Gas_opticsNN<double>;
#endif

