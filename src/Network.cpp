//#include <math.h>

#include <cmath>
#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <Network.h>
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
    constexpr int N_layI=4;
    constexpr int N_lay1=64;
    constexpr int N_lay2=64;

    extern "C" void cblas_sgemm(
    const  CBLAS_ORDER, const  CBLAS_TRANSPOSE, const  CBLAS_TRANSPOSE, const int, const int, const int,
    const float, const float*, const int, const float*, const int, const float, const float*, const int);

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

    void transpose(float* restrict matrix, const int Ncol, const int Nrow)
    {
        float tmp[Ncol*Nrow];
        for (int i = 0; i < Ncol; ++i)
            for (int j = 0; j < Nrow; ++j)
                tmp[j+i*Nrow] = matrix[i+j*Ncol];
            
        for (int i = 0; i < Ncol; ++i)
            for (int j = 0; j < Nrow; ++j)
                matrix[j+i*Nrow] = tmp[j+i*Nrow];
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
        if (trans==1)
        {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Nrow,Nbatch,Ncol,1.,weights,Ncol,layer_in,Ncol ,0.,layer_out,Nbatch);       
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Nrow,Nbatch,Ncol,1.,weights,Ncol,layer_in,Nbatch ,0.,layer_out,Nbatch);       
        }
        bias_and_activate(layer_out,bias,Nrow,Nbatch);
    }        		


    void matmul_leakyrelu(
            const int Nbatch,
            const int Nrow, 
            const int Ncol, 
            const float* restrict weights,
            const float* restrict bias,
            float* restrict const layer_in,
            float* restrict const layer_out)
    {
        for (int i=0; i<Nbatch; ++i)
        {
            for (int j=0; j<Nrow; ++j)
            {
                const int layidx=j + i * Nrow;
                layer_out[layidx] = 0.;
                for (int k=0; k<Ncol; ++k)
                {
                    const int wgtidx = k + j * Ncol;
                    const int inpidx = k + i * Ncol;
                    layer_out[layidx] += weights[wgtidx] * layer_in[inpidx];
                }
                layer_out[layidx] += bias[j];
                layer_out[layidx] = std::max(0.2f * layer_out[layidx], layer_out[layidx]);
            }
        }
    }        		

    inline float faster_but_inaccurate_exp(float x) 
    {
        x = 1.0f + x / 16.0f;
        x *= x; x *= x; x *= x; x *= x;
        return x;
    }

    void normalize_input(
            float* restrict const input, 
            const float* restrict const input_mean,
            const float* restrict const input_stdev,
            const int Nbatch,
            const int N_layI)
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

    void Feedforward(
            float* restrict const input, 
            float* restrict const output,
            const float* restrict const layer1_wgth,
            const float* restrict const layer2_wgth,
            const float* restrict const output_wgth,
            const float* restrict const layer1_bias,
            const float* restrict const layer2_bias,
            const float* restrict const output_bias,
            const float* restrict const input_mean,
            const float* restrict const input_stdev,
            const float* restrict const output_mean,
            const float* restrict const output_stdev,
            float* restrict const layer1,
            float* restrict const layer2,
            const int Nbatch,
            const int N_layO,
            const int N_layI,
            const int do_exp,
            const int do_inpnorm)
    
    {  
        if (do_inpnorm) {normalize_input(input,input_mean,input_stdev,Nbatch,N_layI);}

        matmul_bias_act_blas(Nbatch,N_lay1,N_layI,layer1_wgth,layer1_bias,input,layer1,0);
        matmul_bias_act_blas(Nbatch,N_lay2,N_lay1,layer2_wgth,layer2_bias,layer1,layer2,0);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N_layO,Nbatch,N_lay2,1.,output_wgth,N_lay2,layer2,Nbatch,0.,output,Nbatch);

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



void Network::file_reader(
        float* weights,
        const std::string& filename,
        const int N)
{
    std::ifstream file (filename.c_str());
    file.is_open();
    for ( int i=0; i<N;++i)
        file>> weights[i];
    file.close();
}

void Network::Inference(
        float* inputs,
        float* outputs,
        const int lower_atmos,
        const int do_exp,
        const int do_inpnorm)
{
    if (lower_atmos == 1)
    {
        this->layer1.resize(N_lay1*this->Nbatch_lower);
        this->layer2.resize(N_lay2*this->Nbatch_lower);
        Feedforward(
            inputs,
            outputs,
            this->layer1_wgth_lower.data(),
            this->layer2_wgth_lower.data(),
            this->output_wgth_lower.data(),
            this->layer1_bias_lower.data(),
            this->layer2_bias_lower.data(),
            this->output_bias_lower.data(),
            this->mean_input_lower.data(),
            this->stdev_input_lower.data(),
            this->mean_output_lower.data(),
            this->stdev_output_lower.data(),
            this->layer1.data(),
            this->layer2.data(),
            this->Nbatch_lower,
            this->N_layO,
            this->N_layI,
            do_exp,
            do_inpnorm);
    } else {
        this->layer1.resize(N_lay1*this->Nbatch_upper);
        this->layer2.resize(N_lay2*this->Nbatch_upper);
        Feedforward(
            inputs,
            outputs,
            this->layer1_wgth_upper.data(),
            this->layer2_wgth_upper.data(),
            this->output_wgth_upper.data(),
            this->layer1_bias_upper.data(),
            this->layer2_bias_upper.data(),
            this->output_bias_upper.data(),
            this->mean_input_upper.data(),
            this->stdev_input_upper.data(),
            this->mean_output_upper.data(),
            this->stdev_output_upper.data(),
            this->layer1.data(),
            this->layer2.data(),
            this->Nbatch_upper,
            this->N_layO,
            this->N_layI,
            do_exp,
            do_inpnorm);
    }

}


Network::Network(const int Nbatch_lower,const int Nbatch_upper,
                 const std::vector<float>& bias1_lower,
                 const std::vector<float>& bias2_lower,
                 const std::vector<float>& bias3_lower,
                 const std::vector<float>& wgth1_lower,
                 const std::vector<float>& wgth2_lower,
                 const std::vector<float>& wgth3_lower,
                 const std::vector<float>& Fmean_lower,
                 const std::vector<float>& Fstdv_lower,
                 const std::vector<float>& Lmean_lower,
                 const std::vector<float>& Lstdv_lower,
                 const std::vector<float>& bias1_upper,
                 const std::vector<float>& bias2_upper,
                 const std::vector<float>& bias3_upper,
                 const std::vector<float>& wgth1_upper,
                 const std::vector<float>& wgth2_upper,
                 const std::vector<float>& wgth3_upper,
                 const std::vector<float>& Fmean_upper,
                 const std::vector<float>& Fstdv_upper,
                 const std::vector<float>& Lmean_upper,
                 const std::vector<float>& Lstdv_upper,
                 const int N_layO,const int N_layI)
{
    this->Nbatch_lower = Nbatch_lower;
    this->Nbatch_upper = Nbatch_upper;
    this->N_layO = N_layO;
    this->N_layI = N_layI;

    this->layer1_wgth_lower = wgth1_lower;
    this->layer2_wgth_lower = wgth2_lower;
    this->output_wgth_lower = wgth3_lower;
    this->layer1_bias_lower = bias1_lower;
    this->layer2_bias_lower = bias2_lower;
    this->output_bias_lower = bias3_lower;
    this->mean_input_lower = Fmean_lower;
    this->stdev_input_lower = Fstdv_lower;
    this->mean_output_lower = Lmean_lower;
    this->stdev_output_lower = Lstdv_lower;

    this->layer1_wgth_upper = wgth1_upper;
    this->layer2_wgth_upper = wgth2_upper;
    this->output_wgth_upper = wgth3_upper;
    this->layer1_bias_upper = bias1_upper;
    this->layer2_bias_upper = bias2_upper;
    this->output_bias_upper = bias3_upper;
    this->mean_input_upper = Fmean_upper;
    this->stdev_input_upper = Fstdv_upper;
    this->mean_output_upper = Lmean_upper;
    this->stdev_output_upper = Lstdv_upper;
}





