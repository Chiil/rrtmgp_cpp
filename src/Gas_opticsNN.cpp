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

static Logger gLogger;
static const int Ninput = 5;     // number of input columns (water vapour, ozone, pressure, temperature)
static const int Noutput_SW = 224;  // number of output columns (shortwave optical depths)

namespace
{
    double starttime,endtime;
    double get_wall_time(){
        struct timeval time;
        if (gettimeofday(&time,NULL)){
            //  Handle error
            return 0;
            }
        return (double)time.tv_sec + (double)time.tv_usec * .000001;
    }

    ICudaEngine* makecudaengine(std::string eng_name){
        //Read Engine from file
        std::stringstream ModelStream;
        ModelStream.seekg(0, ModelStream.beg);
        std::ifstream cache(eng_name);
        ModelStream << cache.rdbuf();
        cache.close();
        //Obtain size and memory usage of engine
        ModelStream.seekg(0, std::ios::end);
        const int modelSize = ModelStream.tellg();
        ModelStream.seekg(0, std::ios::beg);
        void* modelMem = malloc(modelSize);
        ModelStream.read((char*)modelMem, modelSize);
        //Deserialize engine
        IRuntime* runtime = createInferRuntime(gLogger);
        //Return deserialized engine
        return runtime->deserializeCudaEngine(modelMem,modelSize,NULL);
    }
}

//     // IMPLEMENTATION OF CLASS FUNCTIONS.
//     // TensorRT inference function
template<typename TF>
void Gas_opticsNN<TF>::inference(IExecutionContext& context, 
                            float * input, 
                            float * output, 
                            const int & batchSize)const
{
        const ICudaEngine& engine = context.getEngine();
        void* buffers[2];
        int inputIndex = engine.getBindingIndex("TensorRTInputPH_0");
        int outputIndex = engine.getBindingIndex("TensorRTOutputPH_0");
        cudaMalloc(&buffers[inputIndex], batchSize * Ninput * sizeof(float));   // data
        cudaMalloc(&buffers[outputIndex], batchSize * Noutput_SW * sizeof(float));   // data
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(buffers[inputIndex], input, batchSize * Ninput * sizeof(float), cudaMemcpyHostToDevice, stream);
        context.enqueue(batchSize, buffers, stream, nullptr);
        cudaMemcpyAsync(output, buffers[outputIndex], batchSize * Noutput_SW * sizeof(float), cudaMemcpyDeviceToHost, stream);
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
    init_TRT_engines(
            plan_files);

}

// Constructor of the shortwave variant.
template<typename TF>
Gas_opticsNN<TF>::Gas_opticsNN(
        const Array<std::string,1>& plan_files, //lower_tau, upper_tau, lower_ssa, upper_ssa
        const Array<std::string,1>& gas_names,
        const Array<int,2>& band2gpt,
        const Array<TF,2>& band_lims_wavenum,
        const Array<TF,1>& solar_src):
            Optical_props<TF>(band_lims_wavenum, band2gpt),
            solar_src(solar_src)
{ 
    this->is_longwave = false;   
    this->gas_names = gas_names;
    init_TRT_engines(
            plan_files);
            
}

template<typename TF>
void Gas_opticsNN<TF>::init_TRT_engines(
        const Array<std::string,1> & plan_files)
{
        ICudaEngine* engine_lower_tau = makecudaengine(plan_files({1}));
        this->context_lower_tau = engine_lower_tau->createExecutionContext();

        ICudaEngine* engine_upper_tau = makecudaengine(plan_files({2}));
        this->context_upper_tau = engine_upper_tau->createExecutionContext();
         
        if (this->is_longwave){
            ICudaEngine* engine_lower_planck = makecudaengine(plan_files({3}));
            this->context_lower_plk = engine_lower_planck->createExecutionContext();

            ICudaEngine* engine_upper_planck = makecudaengine(plan_files({4}));
            this->context_upper_plk = engine_upper_planck->createExecutionContext();
         } else{
            ICudaEngine* engine_lower_ssa = makecudaengine(plan_files({3}));
            this->context_lower_ssa = engine_lower_ssa->createExecutionContext();

            ICudaEngine* engine_upper_ssa = makecudaengine(plan_files({4}));
            this->context_upper_ssa = engine_upper_ssa->createExecutionContext();
         }        
        
}

// Gas optics solver longwave variant.
template<typename TF>
void Gas_opticsNN<TF>::gas_optics(
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
    Array<TF,2> sfc_factor({ncol,nband});

    compute_tau_sources_NN(
            ncol, nlay, nband, ngpt,
            play, plev, tlay, tlev,
            gas_desc, sources, 
            optical_props);   
    //fill surface sources  
    lay2sfc_factor(tlay,tsfc,sfc_factor,sources,ncol,nlay,nband);   
}

// Gas optics solver shortwave variant.
template<typename TF>
void Gas_opticsNN<TF>::gas_optics(
        const Array<TF,2>& play,
        const Array<TF,2>& plev,
        const Array<TF,2>& tlay,
        const Gas_concs<TF>& gas_desc,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props,
        Array<TF,2>& toa_src) const

{   
    const int ncol = play.dim(1);
    const int nlay = play.dim(2);
    const int ngpt = this->get_ngpt();
    const int nband = this->get_nband();

    compute_tau_ssa_NN(
          ncol, nlay, ngpt, nband,
          play, plev, tlay, gas_desc,
          optical_props);

    // External source function is constant.
    for (int igpt=1; igpt<=ngpt; ++igpt){
        for (int icol=1; icol<=ncol; ++icol){
            toa_src({icol, igpt}) = this->solar_src({igpt});
        }
    }
}

template<typename TF>
void Gas_opticsNN<TF>::lay2sfc_factor(
        const Array<TF,2>& tlay,
        const Array<TF,1>& tsfc,
        Array<TF,2>& sfc_factor,
        Source_func_lw<TF>& sources,
        const int& ncol,
        const int& nlay,
        const int& nband) const
{
    Array<TF,3>& src_layer = sources.get_lay_source();
    Array<TF,2>& src_sfc   = sources.get_sfc_source();

    for (int icol=1; icol<=ncol; ++icol){
        const float tempfrac = tsfc({icol})/tlay({nlay,icol});
        sfc_factor({1}) = 0.184351*pow(tempfrac,3.) + 0.080502*pow(tempfrac,2.) + 0.779973*tempfrac - 0.044828;
        sfc_factor({2}) = 1.456914*pow(tempfrac,3.) - 2.434327*pow(tempfrac,2.) + 2.690731*tempfrac - 0.713335;
        sfc_factor({3}) = 4.931766*pow(tempfrac,3.) - 11.21031*pow(tempfrac,2.) + 10.51688*tempfrac - 3.238430;
        sfc_factor({4}) = 8.450806*pow(tempfrac,3.) - 20.69442*pow(tempfrac,2.) + 19.35547*tempfrac - 6.112041;
        sfc_factor({5}) = 13.12718*pow(tempfrac,3.) - 33.65488*pow(tempfrac,2.) + 31.67371*tempfrac - 10.14633;
        sfc_factor({6}) = 23.45908*pow(tempfrac,3.) - 63.12546*pow(tempfrac,2.) + 60.28334*tempfrac - 19.61765;
        sfc_factor({7}) = exp(-4.896122+4.896520*tempfrac);
        sfc_factor({8}) = exp(-5.361318+5.361619*tempfrac);
        sfc_factor({9}) = exp(-6.048732+6.049125*tempfrac);
        sfc_factor({10})= exp(-6.796339+6.796480*tempfrac);
        sfc_factor({11})= exp(-7.640071+7.640656*tempfrac);
        sfc_factor({12})= exp(-9.083692+9.084135*tempfrac);
        sfc_factor({13})= exp(-10.21916+10.21936*tempfrac);
        sfc_factor({14})= exp(-10.96706+10.96721*tempfrac);
        sfc_factor({15})= exp(-11.88411+11.88459*tempfrac);
        sfc_factor({16})= exp(-13.55701+13.55843*tempfrac);
        for (int iband=1; iband<=nband; ++iband){
            for (int igpt=1; igpt<=16; ++igpt){
                src_sfc({icol,igpt}) = sfc_factor({iband}) * src_layer({icol,1,igpt});
            }
        } 
    }               
}
 
//Neural Network optical property function for shortwave
//Currently only implemented for atmospheric profilfes ordered bottom-first
template<typename TF>
void Gas_opticsNN<TF>::compute_tau_ssa_NN(
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
    for (int i = 1; i <= nlay; i++){
       if (play({1,i}) > this->press_ref_trop){
           idx_tropo += 1;
       }
    }

    //get gas concentrations
    const Array<TF,2>& h2o = gas_desc.get_vmr(this->gas_names({1}));
    const Array<TF,2>& co2 = gas_desc.get_vmr(this->gas_names({2}));
    const Array<TF,2>& o3  = gas_desc.get_vmr(this->gas_names({3}));

    //// Lower atmosphere: 
    //fill input array
    float input_lower[idx_tropo*Ninput];
    float output_lower_tau[idx_tropo*ngpt];
    float output_lower_ssa[idx_tropo*ngpt];
    for (int i = 0; i < idx_tropo; i++){
        input_lower[i*Ninput+0] = log(h2o({1,i+1}));
        input_lower[i*Ninput+1] = co2({1,1});
        input_lower[i*Ninput+2] = log(o3({1,i+1}));
        input_lower[i*Ninput+3] = log(play({1,i+1}));
        input_lower[i*Ninput+4] = tlay({1,i+1});
    }

    float dp[ncol][nlay];
    for (int ilay=1; ilay<=nlay; ++ilay){
        for (int icol=1; icol<=ncol; ++icol){
           dp[icol-1][ilay-1] = abs(plev({icol,ilay})-plev({icol,ilay+1}));
        }
    }
    //do inference Optical Depth
    inference(*this->context_lower_tau, input_lower, output_lower_tau, idx_tropo);
    inference(*this->context_lower_ssa, input_lower, output_lower_ssa, idx_tropo);
    for (int icol=1; icol<=ncol; ++icol){
         for (int ilay=1; ilay<=idx_tropo; ++ilay){
            for (int igpt=1; igpt<=ngpt; ++igpt){
                const int idxlay = (igpt-1)+(ilay-1)*ngpt+(icol-1)*idx_tropo*ngpt;
                tau({icol, ilay, igpt}) = output_lower_tau[idxlay] * dp[icol-1][ilay-1];
                if (not this->is_longwave) {ssa({icol, ilay, igpt}) = std::min(std::max(output_lower_ssa[idxlay],nul),een);}
            }
        }
    }

    //// Upper atmosphere:
    //fill input array
    float input_upper[(batchSize - idx_tropo)*Ninput];
    float output_upper_tau[(batchSize - idx_tropo)*ngpt];
    float output_upper_ssa[(batchSize - idx_tropo)*ngpt];
    for (int i = idx_tropo; i < batchSize; i++){
        input_upper[Ninput*(i-idx_tropo)+0] = log(h2o({1,i+1}));
        input_upper[Ninput*(i-idx_tropo)+1] = co2({1,1});
        input_upper[Ninput*(i-idx_tropo)+2] = log(o3({1,i+1}));
        input_upper[Ninput*(i-idx_tropo)+3] = log(play({1,i+1}));
        input_upper[Ninput*(i-idx_tropo)+4] = tlay({1,i+1});
    }

    //do inference Optical Depth
    inference(*this->context_upper_tau, input_upper, output_upper_tau,batchSize -  idx_tropo);
    inference(*this->context_upper_ssa, input_upper, output_upper_ssa,batchSize -  idx_tropo);
    for (int icol=1; icol<=ncol; ++icol){
        for (int ilay=idx_tropo+1; ilay<=batchSize; ++ilay){
            for (int igpt=1; igpt<=ngpt; ++igpt){
                const int idxlay = (igpt-1)+(ilay-1-idx_tropo)*ngpt+(icol-1)*(batchSize-idx_tropo)*ngpt;
                tau({icol, ilay, igpt}) = output_upper_tau[idxlay] * dp[icol-1][ilay-1];
                ssa({icol, ilay, igpt}) = std::min(std::max(output_upper_ssa[idxlay],nul),een);
            }
        }
    }

}



//Neural Network optical property function for longwave
//Currently only implemented for atmospheric profilfes ordered bottom-first
template<typename TF>
void Gas_opticsNN<TF>::compute_tau_sources_NN(
        const int ncol, const int nlay, const int ngpt, const int nband,
        const Array<TF,2>& play,
        const Array<TF,2>& plev,
        const Array<TF,2>& tlay,
        const Array<TF,2>& tlev,
        const Gas_concs<TF>& gas_desc,
        Source_func_lw<TF>& sources,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props) const
{
    Array<TF,3>& tau = optical_props->get_tau();
    Array<TF,3>& src_layer = sources.get_lay_source();
    Array<TF,3>& src_lvinc = sources.get_lev_source_inc();
    Array<TF,3>& src_lvdec = sources.get_lev_source_dec();

    const int batchSize=ncol*nlay;
    int idx_tropo = 0;
    float nul = 0.;
    float een = 1.;
    //find index that defines border between upper and lower atmosphere
    for (int i = 1; i <= nlay; i++){
       if (play({1,i}) > this->press_ref_trop){
           idx_tropo += 1;
       }
    }

    //get gas concentrations
    const Array<TF,2>& h2o = gas_desc.get_vmr(this->gas_names({1}));
    const Array<TF,2>& o3  = gas_desc.get_vmr(this->gas_names({3}));

    //// Lower atmosphere:
    //fill input array
    float input_lower_tau[idx_tropo*Ninput];
    float input_lower_plk[idx_tropo*Ninput];
    float output_lower_tau[idx_tropo*ngpt];
    float output_lower_plk[idx_tropo*ngpt*3];
    for (int i = 0; i < idx_tropo; i++){
        input_lower_tau[i*Ninput+0] = log(h2o({1,i+1}));
        input_lower_tau[i*Ninput+1] = log(o3({1,i+1}));
        input_lower_tau[i*Ninput+2] = log(play({1,i+1}));
        input_lower_tau[i*Ninput+3] = tlay({1,i+1});
    }
    for (int i = 0; i < idx_tropo; i++){
        input_lower_plk[i*Ninput+0] = log(h2o({1,i+1}));
        input_lower_plk[i*Ninput+1] = log(o3({1,i+1}));
        input_lower_plk[i*Ninput+2] = log(play({1,i+1}));
        input_lower_plk[i*Ninput+3] = tlay({1,i+1});
        input_lower_plk[i*Ninput+4] = tlev({1,i+1});
        input_lower_plk[i*Ninput+5] = tlev({1,i+2});

    }

    float dp[ncol][nlay];
    for (int ilay=1; ilay<=nlay; ++ilay){
        for (int icol=1; icol<=ncol; ++icol){
           dp[icol-1][ilay-1] = abs(plev({icol,ilay})-plev({icol,ilay+1}));
        }
    }
    //do inference Optical Depth
    inference(*this->context_lower_tau, input_lower_tau, output_lower_tau, idx_tropo);
    inference(*this->context_lower_plk, input_lower_plk, output_lower_plk, idx_tropo);
    for (int icol=1; icol<=ncol; ++icol){
         for (int ilay=1; ilay<=idx_tropo; ++ilay){
            for (int igpt=1; igpt<=ngpt; ++igpt){
                const int idxlay1 = (igpt-1)+(ilay-1)*ngpt+(icol-1)*idx_tropo*ngpt;
                const int idxlay2 = (igpt-1)+(ilay-1)*ngpt*3+(icol-1)*idx_tropo*ngpt*3;
                const int idxlay3 = (igpt-1+ngpt)+(ilay-1)*ngpt*3+(icol-1)*idx_tropo*ngpt*3;
                const int idxlay4 = (igpt-1+ngpt*2)+(ilay-1)*ngpt*3+(icol-1)*idx_tropo*ngpt*3;
                tau({icol, ilay, igpt}) = output_lower_tau[idxlay1] * dp[icol-1][ilay-1];
                src_layer({icol, ilay, igpt}) = output_lower_plk[idxlay2];
                src_lvinc({icol, ilay, igpt}) = output_lower_plk[idxlay3];
                src_lvdec({icol, ilay, igpt}) = output_lower_plk[idxlay4];
            }
        }
    }

    //// Upper atmosphere:
    //fill input array
    float input_upper_tau[(batchSize - idx_tropo)*Ninput];
    float input_upper_plk[(batchSize - idx_tropo)*Ninput];
    float output_upper_tau[(batchSize - idx_tropo)*ngpt];
    float output_upper_plk[(batchSize - idx_tropo)*ngpt];
    for (int i = idx_tropo; i < batchSize; i++){
        input_upper_tau[Ninput*(i-idx_tropo)+0] = log(h2o({1,i+1}));
        input_upper_tau[Ninput*(i-idx_tropo)+1] = log(o3({1,i+1}));
        input_upper_tau[Ninput*(i-idx_tropo)+2] = log(play({1,i+1}));
        input_upper_tau[Ninput*(i-idx_tropo)+3] = tlay({1,i+1});
    }
    for (int i = idx_tropo; i < batchSize; i++){
        input_upper_plk[Ninput*(i-idx_tropo)+0] = log(h2o({1,i+1}));
        input_upper_plk[Ninput*(i-idx_tropo)+1] = log(o3({1,i+1}));
        input_upper_plk[Ninput*(i-idx_tropo)+2] = log(play({1,i+1}));
        input_upper_plk[Ninput*(i-idx_tropo)+3] = tlay({1,i+1});
        input_upper_plk[Ninput*(i-idx_tropo)+4] = tlev({1,i+1});
        input_upper_plk[Ninput*(i-idx_tropo)+5] = tlev({1,i+2});

    }


    //do inference Optical Depth
    inference(*this->context_upper_tau, input_upper_tau, output_upper_tau,batchSize -  idx_tropo);
    inference(*this->context_upper_plk, input_upper_plk, output_upper_plk,batchSize -  idx_tropo);
    for (int icol=1; icol<=ncol; ++icol){
        for (int ilay=idx_tropo+1; ilay<=batchSize; ++ilay){
            for (int igpt=1; igpt<=ngpt; ++igpt){
                const int idxlay1 = (igpt-1)+(ilay-1-idx_tropo)*ngpt+(icol-1)*(batchSize-idx_tropo)*ngpt;
                const int idxlay2 = (igpt-1)+(ilay-1-idx_tropo)*ngpt*3+(icol-1)*(batchSize-idx_tropo)*ngpt*3;
                const int idxlay3 = (igpt-1+ngpt)+(ilay-1-idx_tropo)*ngpt*3+(icol-1)*(batchSize-idx_tropo)*ngpt*3;
                const int idxlay4 = (igpt-1+ngpt*2)+(ilay-1-idx_tropo)*ngpt*3+(icol-1)*(batchSize-idx_tropo)*ngpt*3;
                tau({icol, ilay, igpt}) = output_upper_tau[idxlay1] * dp[icol-1][ilay-1];
                src_layer({icol, ilay, igpt}) = output_lower_plk[idxlay2];
                src_lvinc({icol, ilay, igpt}) = output_lower_plk[idxlay3];
                src_lvdec({icol, ilay, igpt}) = output_lower_plk[idxlay4];
            }
        }
    }

}





#ifdef FLOAT_SINGLE
template class Gas_opticsNN<float>;
#else
template class Gas_opticsNN<double>;
#endif

