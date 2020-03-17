#ifndef NETWORK_H
#define NETWORK_H

#include <math.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
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
                Netcdf_group grp,
                const int N_layO, const int N_layI);

    private:
        int Nbatch_lower;
        int Nbatch_upper;
        int N_layO;
        int N_layI;
        std::vector<float> output_wgth_lower;
        std::vector<float> output_wgth_upper;
        std::vector<float> output_bias_lower;
        std::vector<float> output_bias_upper;
        //if constexpr (Nlayer > 0)
        //{
            std::vector<float> layer1_wgth_lower;
            std::vector<float> layer1_bias_lower;
            std::vector<float> layer1_wgth_upper;
            std::vector<float> layer1_bias_upper;
            std::vector<float> layer1;
        //}
        //if constexpr (Nlayer > 1)
        //{
        std::vector<float> layer2_wgth_lower;
        std::vector<float> layer2_bias_lower;
        std::vector<float> layer2_wgth_upper;
        std::vector<float> layer2_bias_upper;
        std::vector<float> layer2;
        //}
        //if constexpr (Nlayer > 2)
        //{
        std::vector<float> layer3_wgth_lower;
        std::vector<float> layer3_bias_lower;
        std::vector<float> layer3_wgth_upper;
        std::vector<float> layer3_bias_upper;
        std::vector<float> layer3;
        //}
};
#endif
