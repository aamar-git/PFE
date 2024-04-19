#include <cstdlib>

int main() {
    // Run the xxd command to convert the TFLite model to C code
    std::system("xxd -i C:\\Users\\Aamar\\Documents\\MATLAB\\Examples\\R2023b\\deeplearning_shared\\KeywordSpottingInNoiseUsingMFCCAndLSTMNetworksExample\\CNN\\smaller_optimized_model.tflite > model_data.cc");

    return 0;
}
