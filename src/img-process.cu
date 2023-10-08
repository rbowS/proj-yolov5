#include "simple-logger.cuh"
#include "img-process.cuh"

namespace IMGPRrocess{
    

    __host__ __device__ void invertAffineTransform(const float* src, float* dst) {
        float determinant = src[0] * src[4] - src[1] * src[3];
        if (determinant != 0.0) {
            float inv_det = 1.0 / determinant;
            dst[0] = src[4] * inv_det;
            dst[1] = -src[1] * inv_det;
            dst[2] = (src[1] * src[5] - src[2] * src[4]) * inv_det;
            dst[3] = -src[3] * inv_det;
            dst[4] = src[0] * inv_det;
            dst[5] = (src[2] * src[3] - src[0] * src[5]) * inv_det;
        } else {
            INFOF("Matrix is not invertible.");
        }
    }


    __forceinline__ __device__ float bilinearInterpolation(const unsigned char* src, int src_width, 
                                           int src_height, float x, float y, int channel) 
    {
        int x0 = static_cast<int>(floor(x));
        int x1 = static_cast<int>(ceil(x));
        int y0 = static_cast<int>(floor(y));
        int y1 = static_cast<int>(ceil(y));

        float dx = x - x0;
        float dy = y - y0;

        float interpolated_value = (1 - dx) * (1 - dy) * src[(y0 * src_width + x0) * 3 + channel] +
                                dx * (1 - dy) * src[(y0 * src_width + x1) * 3 + channel] +
                                (1 - dx) * dy * src[(y1 * src_width + x0) * 3 + channel] +
                                dx * dy * src[(y1 * src_width + x1) * 3 + channel];

        return interpolated_value;
    }

    __global__ void warpAffine(unsigned char* src_data, float *output_image, float *d2i,
                               const int input_height, const int input_width, const int src_width, 
                               const int src_height)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(idx < input_height*input_width)
        {
            int y = idx / input_width;
            int x = idx % input_width;
            float src_x, src_y;
            src_x = d2i[0] * x + d2i[1] * y + d2i[2];
            src_y = d2i[3] * x + d2i[4] * y + d2i[5];
            for (int c = 0; c < 3; c++) { 
                if (src_x >= 0 && src_x < src_width && src_y >= 0 && src_y < src_height) {
                    output_image[(3-c-1)*input_height*input_width + y*input_width+x] =
                        bilinearInterpolation(src_data, src_width, src_height, src_x, src_y, c);
                    output_image[(3-c-1)*input_height*input_width + y*input_width+x] /= 255.0;
                } else {
                    output_image[(3-c-1)*input_height*input_width + y*input_width+x] = 114;
                    output_image[(3-c-1)*input_height*input_width + y*input_width+x] /= 255.0;
                }
            }
        }   
        
    }


    __global__ void decode_kernel(float *in, float *out, float *d2i, const int numbox, 
                const int numprob, const int num_classes, const float confidence_threshold)
    {
        const int idx = threadIdx.x + blockDim.x*blockIdx.x;
        const int tid = threadIdx.x;
        __shared__ float3 smem;
        if(tid == 0)
        {
            smem.x = d2i[0];
            smem.y = d2i[2];
            smem.z = d2i[5];
        }
        __syncthreads();
        if(idx < numbox)
        {
            float* ptr = in + idx * numprob;
            float objness = ptr[4];
            //bad writting, may cause warp diverge
            if(objness > confidence_threshold)
            {
                float* pclass = ptr + 5;
                float prob = pclass[0];
                float label = 0;
                for (int i = 1; i < num_classes; i++)
                {
                    if(prob < pclass[i])
                    {
                        prob = pclass[i];
                        label = i;
                    }
                }
                
                float confidence = prob * objness;
                if(confidence > confidence_threshold)
                {
                    // 中心点、宽、高
                    float cx     = ptr[0];
                    float cy     = ptr[1];
                    float width  = ptr[2];
                    float height = ptr[3];

                    // 预测框
                    float left   = cx - width * 0.5;
                    float top    = cy - height * 0.5;
                    float right  = cx + width * 0.5;
                    float bottom = cy + height * 0.5;

                    // 对应图上的位置
                    float image_base_left   = smem.x * left   + smem.y;
                    float image_base_right  = smem.x * right  + smem.y;
                    float image_base_top    = smem.x * top    + smem.z;
                    float image_base_bottom = smem.x * bottom + smem.z;

                    out[idx*6]   = image_base_left;
                    out[idx*6+1] = image_base_top;
                    out[idx*6+2] = image_base_right;
                    out[idx*6+3] = image_base_bottom;
                    out[idx*6+4] = label;
                    out[idx*6+5] = confidence;
                }
            }
        }
    }


    void encode_kernel_invoker(std::vector<std::string> filePaths, float *d_output_image, 
                               std::vector<cv::Mat> &imgMats, float *d2is, const int input_batch, 
                               const int input_width, const int input_height)
    {
        
        cudaStream_t streams[input_batch];
        for (int i = 0; i < input_batch; i++) {
            cudaStreamCreate(&streams[i]);
        }
        
        ///////////////////////////////////////////////////
        // letter box
        const int blockSize = 256;
        const int gridSize = (input_height*input_width + blockSize -1)/blockSize;
        for (int i = 0; i < input_batch; i++)
        {
            imgMats[i] = cv::imread(filePaths[i]);
            int src_width = imgMats[i].cols;
            int src_height = imgMats[i].rows;
            
            unsigned char* h_src_data = imgMats[i].data;
            // 通过双线性插值对图像进行resize
            float scale_x = input_width / (float)imgMats[i].cols;
            float scale_y = input_height / (float)imgMats[i].rows;
            float scale = std::min(scale_x, scale_y);
            float i2d[6];
            // resize图像，源图像和目标图像几何中心的对齐
            i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * imgMats[i].cols + input_width + scale  - 1) * 0.5;
            i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * imgMats[i].rows + input_height + scale - 1) * 0.5;
            IMGPRrocess::invertAffineTransform(i2d, d2is+i*6);

            float *d_d2i;
            unsigned char* d_src_data;
            checkRuntime( cudaMalloc((void**)&d_d2i, 6*sizeof(float)) );
            checkRuntime( cudaMalloc((void**)&d_src_data, 3*src_width*src_height*sizeof(unsigned char)) );
            checkRuntime( cudaMemcpyAsync(d_d2i, d2is+i*6, sizeof(float)*6, cudaMemcpyHostToDevice, streams[i]) );
            checkRuntime( cudaMemcpyAsync(d_src_data, h_src_data, 3*src_width*src_height*sizeof(unsigned char), cudaMemcpyHostToDevice, streams[i]) );

            
            warpAffine<<<gridSize, blockSize, 0, streams[i]>>>
                      (d_src_data, d_output_image+i*3*input_width*input_height, 
                       d_d2i, input_height, input_width, src_width, src_height);
            cudaFree(d_d2i);
            cudaFree(d_src_data);
        }
        
        for (int i = 0; i < input_batch; i++) {
            cudaStreamDestroy(streams[i]);
        }
        
    }


    std::vector<std::vector<float>> decode_kernel_invoker(float *d_in, float *d2i, 
                    const int numbox, const int numprob, const int num_classes, 
                    const float confidence_threshold){
        
        std::vector<std::vector<float>> bboxes;
        const int blockSize = 256;
        const int gridSize = (numbox + blockSize -1)/blockSize;
        float *d_out;
        float *d_d2i;
        float *h_out = new float[numbox*6];
        checkRuntime( cudaMalloc((void**)&d_out, numbox*6*sizeof(float)) );
        checkRuntime( cudaMemset(d_out, 0, numbox*6*sizeof(float)) );
        checkRuntime( cudaMalloc((void**)&d_d2i, 6*sizeof(float)) );
        checkRuntime( cudaMemcpy(d_d2i, d2i, sizeof(float)*6, cudaMemcpyHostToDevice) );
        decode_kernel<<<gridSize, blockSize>>>(d_in, d_out, d_d2i, numbox, 
                        numprob, num_classes, confidence_threshold);
        cudaDeviceSynchronize();
        checkRuntime( cudaMemcpy(h_out, d_out, numbox*6*sizeof(float), cudaMemcpyDeviceToHost) );
        for (int i = 0; i < numbox; i++)
        {
            if(h_out[i*6+5] != 0.0f)
            {
                bboxes.push_back({h_out[i*6], h_out[i*6+1], h_out[i*6+2], 
                                  h_out[i*6+3],h_out[i*6+4], h_out[i*6+5]});
            }
        }
        delete[] h_out;
        cudaFree(d_out);
        cudaFree(d_d2i);
        return bboxes;
    }

    

}