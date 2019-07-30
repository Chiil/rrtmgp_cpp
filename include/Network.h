#include <math.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
const int N_layI=4;
const int N_lay1=32;
const int N_lay2=64;
const int N_lay3=128;
const int N_layO=256;

class Network
{
    private:
        void file_reader(
            float *weights,
            std::string filename,
            int N);

        void Feedforward(
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
            const int offset);

        std::vector<float> layer1_wgth;//{N_lay1 * N_layO};
        std::vector<float> layer2_wgth;//{N_lay2 * N_lay1};
        std::vector<float> layer3_wgth;//{N_lay3 * N_lay2};
        std::vector<float> output_wgth;//{N_layO * N_lay3};
        std::vector<float> layer1_bias;//{N_lay1};
        std::vector<float> layer2_bias;//{N_lay2};
        std::vector<float> layer3_bias;//{N_lay3};
        std::vector<float> output_bias;//{N_layO}; 

        std::vector<float> mean_input;//{N_layO};
        std::vector<float> stdev_input;//{N_layO};
        std::vector<float> mean_output;//{N_layO};
        std::vector<float> stdev_output;//{N_layO};



    public:
        void Inference(
            float *Inputs,
            float *Outputs,
            const int Nbatch);

    Network();

};
