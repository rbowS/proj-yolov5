#ifndef DBINTG_CUDA_TOOLS_DATAAUG
#define DBINTG_CUDA_TOOLS_DATAAUG

#include "mix-memory.cuh"
#include "cuda-tools.cuh"
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace IMGPRrocess{
    
    __host__ __device__ void invertAffineTransform(const float* src, float* dst);

    __forceinline__ __device__ float bilinearInterpolation(const unsigned char* src, int src_width, 
                                           int src_height, float x, float y, int channel);

    __global__ void warpAffine(unsigned char* src_data, float *output_image, float *d2i,
                               const int input_height, const int input_width, const int src_width, 
                               const int src_height);

    __global__ void decode_kernel(float *in, float *out, float *d2i, const int numbox, 
                const int numprob, const int num_classes, const float confidence_threshold);

    void encode_kernel_invoker(std::vector<std::string> filePaths, float *d_output_image, 
                               std::vector<cv::Mat> &imgMats, float *d2is, const int input_batch, 
                               const int input_width, const int input_height);

    std::vector<std::vector<float>> decode_kernel_invoker(float *d_in, float *d2i, 
                    const int numbox, const int numprob, const int num_classes, 
                    const float confidence_threshold);

}

#endif