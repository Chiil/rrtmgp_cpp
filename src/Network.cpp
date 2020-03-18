//#include <math.h>

#include <cmath>
#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include "Netcdf_interface.h"
#include "Network.h"
#include <mkl.h>
#include <time.h>
#include <sys/time.h>
#define restrict __restrict__
namespace
{
    double get_wall_time3()
    {
        struct timeval time;
        if (gettimeofday(&time,NULL))
        {
            //  Handle error
            return 0;
        }
        return (double)time.tv_sec + (double)time.tv_usec * .000001;
    }
    double mystart,myend;

    //extern "C" void cblas_sgemm(
    //const  CBLAS_ORDER, const  CBLAS_TRANSPOSE, const  CBLAS_TRANSPOSE, const int, const int, const int,
    //const float, const float*, const int, const float*, const int, const float, const float*, const int);

    inline float leaky_relu(const float a) {return std::max(0.2f*a,a); }

    inline void bias_and_activate(float* restrict output, const float* restrict bias, const int Nout, const int Nbatch)
    {
        for (int i = 0; i < Nout  ; ++i)
            {
                #pragma ivdep
                for (int j = 0; j < Nbatch; ++j)
                    output[j+i*Nbatch] = leaky_relu(output[j+i*Nbatch] + bias[i]);
            }
    }

    void matmul_bias_act_blas(
            const int Nbatch,
            const int Nrow, 
            const int Ncol, 
            const float* restrict weights,
            const float* restrict bias,
            float* restrict const layer_in,
            float* restrict const layer_out,
            const int trans)
    {
//        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Nrow,Nbatch,Ncol,1.,weights,Ncol,layer_in,Ncol ,0.,layer_out,Nbatch);       
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Nrow,Nbatch,Ncol,1.,weights,Ncol,layer_in,Nbatch ,0.,layer_out,Nbatch);       
        bias_and_activate(layer_out,bias,Nrow,Nbatch);
    }        		

    inline float faster_but_inaccurate_exp(float x) 
    {
        x = 1.0f + x / 16.0f;
        x *= x; x *= x; x *= x; x *= x;
        return x;
    }

    template<int Nlayer,int N_layI,int N_lay1,int N_lay2,int N_lay3>
    void normalize_input(
            float* restrict const input, 
            const float* restrict const input_mean,
            const float* restrict const input_stdev,
            const int Nbatch)
    {
        for (int k=0; k<N_layI; ++k)
        {
            #pragma ivdep
            for (int i=0; i<Nbatch; ++i)
            {
                const int idxin = i + k * Nbatch;
                input[idxin]=(input[idxin] - input_mean[k]) / input_stdev[k];
            }   
        }   
    }

    template<int Nlayer,int N_layI,int N_lay1,int N_lay2,int N_lay3>
    void Feedforward(
            float* restrict const input, 
            float* restrict const output,
            const float* restrict const layer1_wgth,
            const float* restrict const layer2_wgth,
            const float* restrict const layer3_wgth,
            const float* restrict const output_wgth,
            const float* restrict const layer1_bias,
            const float* restrict const layer2_bias,
            const float* restrict const layer3_bias,
            const float* restrict const output_bias,
            const float* restrict const input_mean,
            const float* restrict const input_stdev,
            const float* restrict const output_mean,
            const float* restrict const output_stdev,
            float* restrict const layer1,
            float* restrict const layer2,
            float* restrict const layer3,
            const int Nbatch,
            const int N_layO,
            const int do_exp,
            const int do_inpnorm)
    
    {  
        if (do_inpnorm) {normalize_input<Nlayer,N_layI,N_lay1,N_lay2,N_lay3>(input,input_mean,input_stdev,Nbatch);}

        if constexpr (Nlayer==0)
        {   
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N_layO,Nbatch,N_layI,1.,output_wgth,N_layI,input,Nbatch,0.,output,Nbatch);
        }
        if constexpr (Nlayer==1)
        {   
            matmul_bias_act_blas(Nbatch,N_lay1,N_layI,layer1_wgth,layer1_bias,input,layer1,0);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N_layO,Nbatch,N_lay1,1.,output_wgth,N_lay1,layer1,Nbatch,0.,output,Nbatch);
        }
        if constexpr (Nlayer==2)
        {   
            matmul_bias_act_blas(Nbatch,N_lay1,N_layI,layer1_wgth,layer1_bias,input,layer1,0);
            matmul_bias_act_blas(Nbatch,N_lay2,N_lay1,layer2_wgth,layer2_bias,layer1,layer2,0);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N_layO,Nbatch,N_lay2,1.,output_wgth,N_lay2,layer2,Nbatch,0.,output,Nbatch);
        }
        if constexpr (Nlayer==3)
        {   
            matmul_bias_act_blas(Nbatch,N_lay1,N_layI,layer1_wgth,layer1_bias,input,layer1,0);
            matmul_bias_act_blas(Nbatch,N_lay2,N_lay1,layer2_wgth,layer2_bias,layer1,layer2,0);
            matmul_bias_act_blas(Nbatch,N_lay3,N_lay2,layer3_wgth,layer3_bias,layer2,layer3,0);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N_layO,Nbatch,N_lay3,1.,output_wgth,N_lay3,layer3,Nbatch,0.,output,Nbatch);
        }

        //output layer and denormalize
        if (do_exp == 1)
        {
            for (int j=0; j<N_layO; ++j)
            {
                #pragma ivdep
                for (int i=0; i<Nbatch; ++i)
                {
                    const int layidx = i + j * Nbatch;
                    output[layidx] = faster_but_inaccurate_exp(((output[layidx] +  output_bias[j]) * output_stdev[j]) + output_mean[j]) ;
                }
            }
        } else {
            for (int j=0; j<N_layO; ++j)
            {
                #pragma ivdep
                for (int i=0; i<Nbatch; ++i)
                {
                    const int layidx = i + j * Nbatch;
                    output[layidx] = ((output[layidx] +  output_bias[j]) * output_stdev[j]) + output_mean[j] ;
                }
            }
        }
    }
}

template<int Nlayer,int N_layI,int N_lay1,int N_lay2,int N_lay3>
void Network<Nlayer,N_layI,N_lay1,N_lay2,N_lay3>::Inference(
        float* inputs,
        float* outputs,
        const int lower_atmos,
        const int do_exp,
        const int do_inpnorm)
{
    if (lower_atmos == 1)
    {
        if constexpr (Nlayer>0) this->layer1.resize(N_lay1*this->Nbatch_lower);
        if constexpr (Nlayer>1) this->layer2.resize(N_lay2*this->Nbatch_lower);
        if constexpr (Nlayer>2) this->layer3.resize(N_lay3*this->Nbatch_lower);
        Feedforward<Nlayer,N_layI,N_lay1,N_lay2,N_lay3>(
            inputs,
            outputs,
            this->layer1_wgth_lower.data(),
            this->layer2_wgth_lower.data(),
            this->layer3_wgth_lower.data(),
            this->output_wgth_lower.data(),
            this->layer1_bias_lower.data(),
            this->layer2_bias_lower.data(),
            this->layer3_bias_lower.data(),
            this->output_bias_lower.data(),
            this->mean_input_lower.data(),
            this->stdev_input_lower.data(),
            this->mean_output_lower.data(),
            this->stdev_output_lower.data(),
            this->layer1.data(),
            this->layer2.data(),
            this->layer3.data(),
            this->Nbatch_lower,
            this->N_layO,
            do_exp,
            do_inpnorm);
    } else {
        if constexpr (Nlayer>0) this->layer1.resize(N_lay1*this->Nbatch_upper);
        if constexpr (Nlayer>1) this->layer2.resize(N_lay2*this->Nbatch_upper);
        if constexpr (Nlayer>2) this->layer3.resize(N_lay3*this->Nbatch_upper);
        Feedforward<Nlayer,N_layI,N_lay1,N_lay2,N_lay3>(
            inputs,
            outputs,
            this->layer1_wgth_upper.data(),
            this->layer2_wgth_upper.data(),
            this->layer3_wgth_upper.data(),
            this->output_wgth_upper.data(),
            this->layer1_bias_upper.data(),
            this->layer2_bias_upper.data(),
            this->layer3_bias_upper.data(),
            this->output_bias_upper.data(),
            this->mean_input_upper.data(),
            this->stdev_input_upper.data(),
            this->mean_output_upper.data(),
            this->stdev_output_upper.data(),
            this->layer1.data(),
            this->layer2.data(),
            this->layer3.data(),
            this->Nbatch_upper,
            this->N_layO,
            do_exp,
            do_inpnorm);
    }

}

template<int Nlayer,int N_layI,int N_lay1,int N_lay2,int N_lay3>
Network<Nlayer,N_layI,N_lay1,N_lay2,N_lay3>::Network(const int Nbatch_lower,const int Nbatch_upper,
                 Netcdf_group& grp,const int N_layO)
{
    this->Nbatch_lower = Nbatch_lower;
    this->Nbatch_upper = Nbatch_upper;
    this->N_layO = N_layO;

    if constexpr (Nlayer == 0)
    {
        this->output_bias_lower = grp.get_variable<float>("bias1_lower",{N_layO});
        this->output_wgth_lower = grp.get_variable<float>("wgth1_lower",{N_layO,N_layI});
        this->output_bias_upper = grp.get_variable<float>("bias1_upper",{N_layO});
        this->output_wgth_upper = grp.get_variable<float>("wgth1_upper",{N_layO,N_layI});
    }
    else if constexpr (Nlayer == 1)
    {
        this->layer1_bias_lower = grp.get_variable<float>("bias1_lower",{N_lay1});
        this->output_bias_lower = grp.get_variable<float>("bias2_lower",{N_layO});
        this->layer1_wgth_lower = grp.get_variable<float>("wgth1_lower",{N_lay1,N_layI});
        this->output_wgth_lower = grp.get_variable<float>("wgth2_lower",{N_layO,N_lay1});
        this->layer1_bias_upper = grp.get_variable<float>("bias1_upper",{N_lay1});
        this->output_bias_upper = grp.get_variable<float>("bias2_upper",{N_layO});
        this->layer1_wgth_upper = grp.get_variable<float>("wgth1_upper",{N_lay1,N_layI});
        this->output_wgth_upper = grp.get_variable<float>("wgth2_upper",{N_layO,N_lay1});
    }
    else if constexpr (Nlayer == 2)
    {
        this->layer1_bias_lower = grp.get_variable<float>("bias1_lower",{N_lay1});
        this->layer2_bias_lower = grp.get_variable<float>("bias2_lower",{N_lay2});
        this->output_bias_lower = grp.get_variable<float>("bias3_lower",{N_layO});
        this->layer1_wgth_lower = grp.get_variable<float>("wgth1_lower",{N_lay1,N_layI});
        this->layer2_wgth_lower = grp.get_variable<float>("wgth2_lower",{N_lay2,N_lay1});
        this->output_wgth_lower = grp.get_variable<float>("wgth3_lower",{N_layO,N_lay2});
        this->layer1_bias_upper = grp.get_variable<float>("bias1_upper",{N_lay1});
        this->layer2_bias_upper = grp.get_variable<float>("bias2_upper",{N_lay2});
        this->output_bias_upper = grp.get_variable<float>("bias3_upper",{N_layO});
        this->layer1_wgth_upper = grp.get_variable<float>("wgth1_upper",{N_lay1,N_layI});
        this->layer2_wgth_upper = grp.get_variable<float>("wgth2_upper",{N_lay2,N_lay1});
        this->output_wgth_upper = grp.get_variable<float>("wgth3_upper",{N_layO,N_lay2});
    }
    else if constexpr (Nlayer == 3)
    {
        this->layer1_bias_lower = grp.get_variable<float>("bias1_lower",{N_lay1});
        this->layer2_bias_lower = grp.get_variable<float>("bias2_lower",{N_lay2});
        this->layer3_bias_lower = grp.get_variable<float>("bias3_lower",{N_lay3});
        this->output_bias_lower = grp.get_variable<float>("bias4_lower",{N_layO});
        this->layer1_wgth_lower = grp.get_variable<float>("wgth1_lower",{N_lay1,N_layI});
        this->layer2_wgth_lower = grp.get_variable<float>("wgth2_lower",{N_lay2,N_lay1});
        this->layer3_wgth_lower = grp.get_variable<float>("wgth3_lower",{N_lay3,N_lay2});
        this->output_wgth_lower = grp.get_variable<float>("wgth4_lower",{N_layO,N_lay3});
        this->layer1_bias_upper = grp.get_variable<float>("bias1_upper",{N_lay1});
        this->layer2_bias_upper = grp.get_variable<float>("bias2_upper",{N_lay2});
        this->layer3_bias_upper = grp.get_variable<float>("bias3_upper",{N_lay3});
        this->output_bias_upper = grp.get_variable<float>("bias4_upper",{N_layO});
        this->layer1_wgth_upper = grp.get_variable<float>("wgth1_upper",{N_lay1,N_layI});
        this->layer2_wgth_upper = grp.get_variable<float>("wgth2_upper",{N_lay2,N_lay1});
        this->layer3_wgth_upper = grp.get_variable<float>("wgth3_upper",{N_lay3,N_lay2});
        this->output_wgth_upper = grp.get_variable<float>("wgth4_upper",{N_layO,N_lay3});
    }
    this->mean_input_lower   = grp.get_variable<float>("Fmean_lower",{N_layI});
    this->stdev_input_lower  = grp.get_variable<float>("Fstdv_lower",{N_layI});
    this->mean_output_lower  = grp.get_variable<float>("Lmean_lower",{N_layO});
    this->stdev_output_lower = grp.get_variable<float>("Lstdv_lower",{N_layO});
    
    this->mean_input_upper   = grp.get_variable<float>("Fmean_upper",{N_layI});
    this->stdev_input_upper  = grp.get_variable<float>("Fstdv_upper",{N_layI});
    this->mean_output_upper  = grp.get_variable<float>("Lmean_upper",{N_layO});
    this->stdev_output_upper = grp.get_variable<float>("Lstdv_upper",{N_layO});

}

template class Network<Nlayer,NlayI,Nlay1,Nlay2,Nlay3>;



