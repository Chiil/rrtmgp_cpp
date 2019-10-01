//#include <math.h>
#include <cmath>
#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <Network.h>
#include <mkl_cblas.h>
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
    constexpr int N_lay1=32;
    constexpr int N_lay2=64;
    constexpr int N_lay3=128;

    extern "C" void cblas_sgemm(
    const  CBLAS_ORDER, const  CBLAS_TRANSPOSE, const  CBLAS_TRANSPOSE, const int, const int, const int,
    const float, const float*, const int, const float*, const int, const float, const float*, const int);

    inline float leaky_relu(const float a) {return std::max(0.2f*a,a); }

    inline void bias_and_activate(float* __restrict__ output, const float* __restrict__ bias, const int Nout, const int Nbatch)
    {
        for (int i = 0; i < Nout  ; ++i)
            {
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

    void matmul_blas(
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
            const int N_layO)
    
    {  
      std::cout<<"batch size: "<<Nbatch<<std::endl; 
      //normalization and manual solver     
////  for (int j=0; j<Nbatch; ++j)
////  {
////      for (int k=0; k<N_layI; ++k)
////      {
////          const int idxin = k + j * N_layI;
////          input[idxin]=(input[idxin] - input_mean[k]) / input_stdev[k];
////      }   
////  }
////      matmul_leakyrelu(Nbatch,N_lay1,N_layI,layer1_wgth,layer1_bias,input,layer1);
////      matmul_leakyrelu(Nbatch,N_lay2,N_lay1,layer2_wgth,layer2_bias,layer1,layer2);
////          mystart = get_wall_time3();
////      matmul_leakyrelu(Nbatch,N_lay3,N_lay2,layer3_wgth,layer3_bias,layer2,layer3);
////          myend = get_wall_time3();
////          std::cout<<"time: "<<myend-mystart<<std::endl;
////      for (int i=0; i<Nbatch; ++i)
////      {
////          for (int j=0; j<N_layO; ++j)
////          {
////              const int layidx=i + j * Nbatch;
////              output[layidx] = 0.;
////              for (int k=0; k<N_lay3; ++k)
////              {
////                  const int wgtidx = k + j * N_lay3;
////                  const int inpidx = k + i * N_lay3;
////                  output[layidx] += output_wgth[wgtidx] * layer3[inpidx];
////              }
////          }
////      }
////


          //normalization and blas solver
          for (int k=0; k<N_layI; ++k)
          {
              for (int i=0; i<Nbatch; ++i)
              {
                  const int idxin = i + k * Nbatch;
                  input[idxin]=(input[idxin] - input_mean[k]) / input_stdev[k];
              }   
          }
         matmul_blas(Nbatch,N_lay1,N_layI,layer1_wgth,layer1_bias,input,layer1,0);
         matmul_blas(Nbatch,N_lay2,N_lay1,layer2_wgth,layer2_bias,layer1,layer2,0);
             mystart = get_wall_time3();
         matmul_blas(Nbatch,N_lay3,N_lay2,layer3_wgth,layer3_bias,layer2,layer3,0);
            myend = get_wall_time3();
            std::cout<<"time: "<<myend-mystart<<std::endl;
         cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N_layO,Nbatch,N_lay3,1.,output_wgth,N_lay3,layer3,Nbatch,0.,output,Nbatch);
    
    
         //output layer and denormalize
         for (int j=0; j<N_layO; ++j)
         {
             for (int i=0; i<Nbatch; ++i)
             {
                 const int layidx = i + j * Nbatch;
                 output[layidx] = faster_but_inaccurate_exp(((output[layidx] +  output_bias[j]) * output_stdev[j]) + output_mean[j]) ;
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
        float* outputs)
{
    Feedforward(
        inputs,
        outputs,
        this->layer1_wgth.data(),
        this->layer2_wgth.data(),
        this->layer3_wgth.data(),
        this->output_wgth.data(),
        this->layer1_bias.data(),
        this->layer2_bias.data(),
        this->layer3_bias.data(),
        this->output_bias.data(),
        this->mean_input.data(),
        this->stdev_input.data(),
        this->mean_output.data(),
        this->stdev_output.data(),
        this->layer1.data(),
        this->layer2.data(),
        this->layer3.data(),
        this->Nbatch,
        this->N_layO);
}


Network::Network(const int Nbatch,
                 const char bias1[20],
                 const char bias2[20],
                 const char bias3[20],
                 const char bias4[20],
                 const char wgth1[20],
                 const char wgth2[20],
                 const char wgth3[20],
                 const char wgth4[20],
                 const char Min[25],
                 const char Sin[25],
                 const char Mout[25],
                 const char Sout[25],
                 const int N_layO)
{
    this->Nbatch = Nbatch;
    this->N_layO = N_layO;

    float lay1_bias[N_lay1];
    float lay2_bias[N_lay2];
    float lay3_bias[N_lay3];
    float lay4_bias[N_layO];
    file_reader(lay1_bias,bias1,N_lay1);
    file_reader(lay2_bias,bias2,N_lay2);
    file_reader(lay3_bias,bias3,N_lay3);
    file_reader(lay4_bias,bias4,N_layO);

    float lay1_wgth[N_lay1*N_layI];
    float lay2_wgth[N_lay2*N_lay1];
    float lay3_wgth[N_lay3*N_lay2];
    float lay4_wgth[N_layO*N_lay3];

    file_reader(lay1_wgth,wgth1,N_lay1*N_layI);
    file_reader(lay2_wgth,wgth2,N_lay2*N_lay1);
    file_reader(lay3_wgth,wgth3,N_lay3*N_lay2);
    file_reader(lay4_wgth,wgth4,N_layO*N_lay3);

    float mean_in[N_layI];
    float stdv_in[N_layI];
    float mean_out[N_layO];
    float stdv_out[N_layO];

    file_reader(mean_in, Min,   N_layI);
    file_reader(stdv_in, Sin,   N_layI);
    file_reader(mean_out,Mout,  N_layO);
    file_reader(stdv_out,Sout,  N_layO);

    for (int i=0; i<N_layI; ++i)
    {
        this->mean_input.push_back(mean_in[i]);
        this->stdev_input.push_back(stdv_in[i]);
    } 

    for (int i=0; i<N_lay1; ++i)
        this->layer1_bias.push_back(lay1_bias[i]);

    for (int i=0; i<N_lay2; ++i)
        this->layer2_bias.push_back(lay2_bias[i]);

    for (int i=0; i<N_lay3; ++i)
        this->layer3_bias.push_back(lay3_bias[i]);

    for (int i=0; i<N_layO; ++i)
    {
        this->output_bias.push_back(lay4_bias[i]);
        this->mean_output.push_back(mean_out[i]);
        this->stdev_output.push_back(stdv_out[i]);
    }
    for (int i=0; i<N_lay1; ++i)
        for (int j=0; j<N_layI; ++j)
            {
                const int idx = j + i*N_layI;
                this->layer1_wgth.push_back(lay1_wgth[idx]);
            } 
    for (int i=0; i<N_lay2; ++i)
        for (int j=0; j<N_lay1; ++j)
            {
                const int idx = j + i*N_lay1;
                this->layer2_wgth.push_back(lay2_wgth[idx]);
            }
    for (int i=0; i<N_lay3; ++i)
        for (int j=0; j<N_lay2; ++j)
            {
                const int idx = j + i*N_lay2;
                this->layer3_wgth.push_back(lay3_wgth[idx]);
            }
    for (int i=0; i<N_layO; ++i)
        for (int j=0; j<N_lay3; ++j)
            { 
                const int idx = j + i*N_lay3;
                this->output_wgth.push_back(lay4_wgth[idx]);
            }
    this->layer1.resize(N_lay1*Nbatch);
    this->layer2.resize(N_lay2*Nbatch);
    this->layer3.resize(N_lay3*Nbatch);

}





