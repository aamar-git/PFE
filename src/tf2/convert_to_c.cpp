#include <cstdlib>

int main() {
    // Run the xxd command to convert the TFLite model to C code
    std::system("xxd -i C:\\Users\\Aamar\\Desktop\\Project\\src\\tf2\\optimized_model.tflite\ > model_data.cc");

    return 0;
}
