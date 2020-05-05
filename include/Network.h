#ifndef NETWORK_H
#define NETWORK_H

#include <math.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>

constexpr int layer_count  = 0;  //number of layers used
constexpr int input_gases  = 0;  //number of extra gases
constexpr int layer1_nodes = 64; //nodes in first layer (if used)
constexpr int layer2_nodes = 64; //nodes in second layer (if used)
constexpr int layer3_nodes = 0;  //nodes in third layer (if used)


#ifdef EXTRA_GASES_IN
    constexpr int Ngas   = EXTRA_GASES_IN;
#else
    constexpr int Ngas   = input_gases;
#endif

#ifdef NODES_LAYER_1
    constexpr int Nlay1  = NODES_LAYER_1;
#else
    constexpr int Nlay1  = layer1_nodes;
#endif

#ifdef NODES_LAYER_2
    constexpr int Nlay2  = NODES_LAYER_2;
#else
    constexpr int Nlay2  = layer2_nodes;
#endif

#ifdef NODES_LAYER_3
    constexpr int Nlay3  = NODES_LAYER_3;
#else
    constexpr int Nlay3  = layer3_nodes;
#endif

#ifdef NUMBER_LAYERS
    constexpr int Nlayer = NUMBER_LAYERS;
#else
    constexpr int Nlayer = layer_count;
#endif


template <int Nlayer,int N_lay1,int N_lay2,int N_lay3>
class Network
{
    public:
        void Inference(
            float* inputs,
            float* outputs,
            const int lower_atmos,
            const int do_exp,
            const int do_inpnorm);

        Network(const int Nbatch_lower,
                const int Nbatch_upper,
                Netcdf_group& grp,
                const int N_layO,
                const int N_layI);

    private:
        int Nbatch_lower;
        int Nbatch_upper;
        int N_layO;
        int N_layI;
        std::vector<float> output_wgth_lower;
        std::vector<float> output_wgth_upper;
        std::vector<float> output_bias_lower;
        std::vector<float> output_bias_upper;
        
        std::vector<float> layer1_wgth_lower;
        std::vector<float> layer1_bias_lower;
        std::vector<float> layer1_wgth_upper;
        std::vector<float> layer1_bias_upper;
        std::vector<float> layer1;
        
        std::vector<float> layer2_wgth_lower;
        std::vector<float> layer2_bias_lower;
        std::vector<float> layer2_wgth_upper;
        std::vector<float> layer2_bias_upper;
        std::vector<float> layer2;
        
        std::vector<float> layer3_wgth_lower;
        std::vector<float> layer3_bias_lower;
        std::vector<float> layer3_wgth_upper;
        std::vector<float> layer3_bias_upper;
        std::vector<float> layer3;
   
        std::vector<float> mean_input_lower;
        std::vector<float> stdev_input_lower;
        std::vector<float> mean_output_lower;
        std::vector<float> stdev_output_lower;

        std::vector<float> mean_input_upper;
        std::vector<float> stdev_input_upper;
        std::vector<float> mean_output_upper;
        std::vector<float> stdev_output_upper;

};
#endif
