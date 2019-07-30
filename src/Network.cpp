#include <math.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <Network.h>

void Network::file_reader(
    float* weights,
    std::string filename,
    int N)
{
    std::ifstream file (filename.c_str());
    file.is_open();
    for ( int i = 0; i<N;++i)
        file>> weights[i];                          
    file.close();   
}   

void Network::Inference(
    float *Inputs,
    float *Outputs,
    const int Nbatch)
{
    
        Feedforward(
            Inputs,
            Outputs,
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
            Nbatch);  
}


void Network::Feedforward(
    float *input, 
    float *output,
    float *layer1_wgth,
    float *layer2_wgth,
    float *layer3_wgth,
    float *output_wgth,
    float *layer1_bias,
    float *layer2_bias,
    float *layer3_bias,
    float *output_bias,
    float *input_mean,
    float *input_stdev,
    float *output_mean,
    float *output_stdev,
    const int Nbatch)

{
    float layer1[N_lay1*Nbatch];
    float layer2[N_lay2*Nbatch];
    float layer3[N_lay3*Nbatch];

    //normalize with mean and st. dev.
    for (int j = 0; j < Nbatch; ++j)
        {
            for (int k = 0; k < N_layI; ++k)
            {
                const int idxin = k + j * N_layI;
                input[idxin] = (input[idxin] - input_mean[k]) / input_stdev[k];
            }   
        }
    //first layer
    for (int i = 0; i < Nbatch; ++i)
    {
        for (int j = 0; j < N_lay1; ++j)
        {
            const int layidx = j + i * N_lay1;
            layer1[layidx] = 0.;
            for (int k = 0; k < N_layI; ++k)
            {
                const int wgtidx = k + j * N_layI;
                const int inpidx = k + i * N_layI;
                layer1[layidx] += layer1_wgth[wgtidx] * input[inpidx];
    
            }
            layer1[layidx] += layer1_bias[j];
            layer1[layidx] = std::max(0.2f * layer1[layidx], layer1[layidx]);
        }
    }

    //second layer
    for (int i = 0; i < Nbatch; ++i)
    {
        for (int j = 0; j < N_lay2; ++j)
        {
            const int layidx = j + i * N_lay2;
            layer2[layidx] = 0.;
            for (int k = 0; k < N_lay1; ++k)
            {
                const int wgtidx = k + j * N_lay1;
                const int inpidx = k + i * N_lay1;
                layer2[layidx] += layer2_wgth[wgtidx] * layer1[inpidx];

            }
            layer2[layidx] += layer2_bias[j];
            layer2[layidx] = std::max(0.2f * layer2[layidx], layer2[layidx]);
        }
    }

    //third layer
    for (int i = 0; i < Nbatch; ++i)
    {
        for (int j = 0; j < N_lay3; ++j)
        {
            const int layidx = j + i * N_lay3;
            layer3[layidx] = 0.;
            for (int k = 0; k < N_lay2; ++k)
            {
                const int wgtidx = k + j * N_lay2;
                const int inpidx = k + i * N_lay2;
                layer3[layidx] += layer3_wgth[wgtidx] * layer2[inpidx];

            }
            layer3[layidx] += layer3_bias[j];
            layer3[layidx] = std::max(0.2f * layer3[layidx], layer3[layidx]);
        }
    }


    //output layer and denormalize
   for (int i = 0; i < Nbatch; ++i)
    {
        for (int j = 0; j < N_layO; ++j)
        {
            const int layidx = j + i * N_layO;
            output[layidx] = 0.;
            for (int k = 0; k < N_lay3; ++k)
            {
                const int wgtidx = k + j * N_lay3;
                const int inpidx = k + i * N_lay3;
                output[layidx] += output_wgth[wgtidx] * layer3[inpidx];

            }
            output[layidx] = ((output[layidx] +  output_bias[j]) * output_mean[j]) + output_stdev[j] ;
 
        }
    }
std::cout<<"Network2 "<<output[2]<<" "<<output[30]<<std::endl;

}


Network::Network(){
    float lay1_bias[N_lay1];
    float lay2_bias[N_lay2];
    float lay3_bias[N_lay3];
    float lay4_bias[N_layO];

    file_reader(lay1_bias,"bias1.txt",N_lay1);
    file_reader(lay2_bias,"bias2.txt",N_lay2);
    file_reader(lay3_bias,"bias3.txt",N_lay3);
    file_reader(lay4_bias,"bias4.txt",N_layO);

    float lay1_wgth[N_lay1*N_layI];
    float lay2_wgth[N_lay2*N_lay1];
    float lay3_wgth[N_lay3*N_lay2];
    float lay4_wgth[N_layO*N_lay3];

    file_reader(lay1_wgth,"wgth1.txt",N_lay1*N_layI);
    file_reader(lay2_wgth,"wgth2.txt",N_lay2*N_lay1);
    file_reader(lay3_wgth,"wgth3.txt",N_lay3*N_lay2);
    file_reader(lay4_wgth,"wgth4.txt",N_layO*N_lay3);

    float mean_in[N_layI];
    float stdv_in[N_layI];
    float mean_out[N_layO];
    float stdv_out[N_layO];

    file_reader(mean_in,"mean_in.txt",  N_layI);
    file_reader(stdv_in,"std_in.txt",   N_layI);
    file_reader(mean_out,"mean_out.txt",N_layO);
    file_reader(stdv_out,"std_out.txt", N_layO);

    for (int i = 0; i < N_layI; ++i)
        {
            this->mean_input.push_back(  mean_in[i]);
            this->stdev_input.push_back( stdv_in[i]);
        } 

    for (int i = 0; i < N_lay1; ++i)
            this->layer1_bias.push_back( lay1_bias[i]);

    for (int i = 0; i < N_lay2; ++i)
        this->layer2_bias.push_back(lay2_bias[i]);

    for (int i = 0; i < N_lay3; ++i)
        this->layer3_bias.push_back(lay3_bias[i]);

    for (int i = 0; i < N_layO; ++i)
        {
            this->output_bias.push_back(  lay4_bias[i]);
            this->mean_output.push_back(  mean_out[i]);
            this->stdev_output.push_back( stdv_out[i]);
        }
    for (int i = 0; i < N_lay1; ++i)
        for (int j = 0; j < N_layI; ++j)
            {
                const int idx = j + i*N_layI;
                this->layer1_wgth.push_back(lay1_wgth[idx]);
            } 
    for (int i = 0; i < N_lay2; ++i)
        for (int j = 0; j < N_lay1; ++j)
            {
                const int idx = j + i*N_lay1;
                this->layer2_wgth.push_back(lay2_wgth[idx]);
            }
    for (int i = 0; i < N_lay3; ++i)
        for (int j = 0; j < N_lay2; ++j)
            {
                const int idx = j + i*N_lay2;
                this->layer3_wgth.push_back(lay3_wgth[idx]);
            }
    for (int i = 0; i < N_layO; ++i)
        for (int j = 0; j < N_lay3; ++j)
            { 
                const int idx = j + i*N_lay3;
                this->output_wgth.push_back(lay4_wgth[idx]);
            }
}





