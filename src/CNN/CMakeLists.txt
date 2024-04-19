cmake_minimum_required(VERSION 3.25)
project(CNN)

set(CMAKE_CXX_STANDARD 14)

add_executable(CNN
        convert_to_c.cpp
        optimized_model.tflite
        smaller_optimized_model.tflite)
