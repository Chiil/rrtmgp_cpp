#include <math.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
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
                const int N_layO, const int N_layI);

    private:
        void file_reader(
            float* weights,
            const std::string& filename,
            const int N);
        int Nbatch_lower;
        int Nbatch_upper;
        int N_layO;
        int N_layI;
        std::vector<float> layer1_wgth_lower;
        std::vector<float> layer2_wgth_lower;
        std::vector<float> output_wgth_lower;
        std::vector<float> layer1_bias_lower;
        std::vector<float> layer2_bias_lower;
        std::vector<float> output_bias_lower;
        std::vector<float> mean_input_lower;
        std::vector<float> stdev_input_lower;
        std::vector<float> mean_output_lower;
        std::vector<float> stdev_output_lower;
        std::vector<float> layer1_wgth_upper;
        std::vector<float> layer2_wgth_upper;
        std::vector<float> output_wgth_upper;
        std::vector<float> layer1_bias_upper;
        std::vector<float> layer2_bias_upper;
        std::vector<float> output_bias_upper;
        std::vector<float> mean_input_upper;
        std::vector<float> stdev_input_upper;
        std::vector<float> mean_output_upper;
        std::vector<float> stdev_output_upper;
        std::vector<float> layer1;
        std::vector<float> layer2;
};
