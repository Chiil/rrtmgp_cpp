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
            float* outputs);

        Network(const int Nbatch,
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
                const int N_layO);

    private:
        void file_reader(
            float* weights,
            const std::string& filename,
            const int N);

        int Nbatch;
        int N_layO;
        std::vector<float> layer1_wgth;
        std::vector<float> layer2_wgth;
        std::vector<float> layer3_wgth;
        std::vector<float> output_wgth;
        std::vector<float> layer1_bias;
        std::vector<float> layer2_bias;
        std::vector<float> layer3_bias;
        std::vector<float> output_bias;
        std::vector<float> mean_input;
        std::vector<float> stdev_input;
        std::vector<float> mean_output;
        std::vector<float> stdev_output;
        std::vector<float> layer1;
        std::vector<float> layer2;
        std::vector<float> layer3;        
};
