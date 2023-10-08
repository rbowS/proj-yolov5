#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

#include <cuda_runtime.h>

#include <stdio.h>
#include <math.h>

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <functional>
#include <memory>
#include <assert.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "cuda-tools.cuh"
#include "trt_builder.cuh"
#include "simple-logger.cuh"
#include "mix-memory.cuh"
#include "img-process.cuh"
#include "timer.cuh"

using namespace std;


inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

// coco数据集的labels，关于coco：https://cocodataset.org/#home
static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

// hsv转bgr
static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v){
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f*s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
    case 0:r = v; g = t; b = p;break;
    case 1:r = q; g = v; b = p;break;
    case 2:r = p; g = v; b = t;break;
    case 3:r = p; g = q; b = v;break;
    case 4:r = t; g = p; b = v;break;
    case 5:r = v; g = p; b = q;break;
    default:r = 1; g = 1; b = 1;break;}
    return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id){
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kWARNING){
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
            if(severity == Severity::kWARNING){
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if(severity <= Severity::kERROR){
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else{
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
} logger;


// 通过智能指针管理nv返回的指针参数
// 内存自动释放，避免泄漏
template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

bool exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}


vector<unsigned char> load_file(const string& file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}


bool build_model(const char *path)
{   
    if(exists("workspace/engine.trtmodel")){
        printf("Engine.trtmodel has exists.\n");
        return true;
    }

    //SimpleLogger::set_log_level(SimpleLogger::LogLevel::Verbose);
    TRT::compile(
        TRT::Mode::FP32,
        512,
        path,
        "workspace/engine.trtmodel",
        1 << 28
    );
    INFO("Done.");
    return true;
}

void copy_toGPU(float* input_data_device_start, vector<vector<cv::Mat>> &imgMats, float *d2is_start, 
                vector<string> filePaths, const int buffer_id, const int batch_size)
{
    int real_batch_size   = filePaths.size();
    int input_channel = 3;
    int input_height  = 640;
    int input_width   = 640;
    int input_numel   = batch_size * input_channel * input_height * input_width;
    IMGPRrocess::encode_kernel_invoker(filePaths, input_data_device_start+buffer_id*input_numel, 
                                       imgMats[buffer_id], 
                                       d2is_start+6*batch_size*buffer_id, real_batch_size, 
                                       input_width, input_height);

}

void inference(float* input_data_device, vector<unsigned char> &engine_data,
               vector<cv::Mat> &imgMats, float *d2is, 
               const int input_batch){

    TRTLogger logger;
    auto runtime   = make_nvshared( nvinfer1::createInferRuntime(logger) );
    auto engine = make_nvshared( runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()) );
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        return;
    }

    if(engine->getNbBindings() != 2){
        printf("onnx导出有问题, 必须是1个输入和1个输出, 你有：%d个输出.\n", engine->getNbBindings() - 1);
        return;
    }

    int input_channel = 3;
    int input_height  = 640;
    int input_width   = 640;
    int input_numel   = input_batch * input_channel * input_height * input_width;

    
    auto execution_context = make_nvshared( engine->createExecutionContext() );
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    // 设置输出 3x3输入，对应3x3输出
    auto output_dims = engine->getBindingDimensions(1);

    int output_numbox = output_dims.d[1];
    int output_numprob = output_dims.d[2];
    int num_classes = output_numprob - 5;
    int output_numel = input_batch * output_numbox * output_numprob;

    MixMemory output_data;
    //float* output_data_host   = output_data.cpu<float>(output_numel);
    float* output_data_device = output_data.gpu<float>(output_numel);
    
    // 明确当前推理时，使用的数据输入大小
    auto input_dims = engine->getBindingDimensions(0);
    input_dims.d[0] = input_batch;

    execution_context->setBindingDimensions(0, input_dims);
    float* bindings[] = {input_data_device, output_data_device};
    bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
    checkRuntime(cudaStreamSynchronize(stream));
    checkRuntime(cudaStreamDestroy(stream));
    
    // decode box：从不同尺度下的预测还原到原输入图上(包括:预测框，类被概率，置信度
    
    float confidence_threshold = 0.25;
    float nms_threshold = 0.5;
    #if DEBUG
        std::cout<<"output_numbox------------------>"<<output_numbox<<std::endl;
    #endif
    for (int kk = 0; kk < input_batch; kk++)
    {
        vector<vector<float>> bboxes = 
                IMGPRrocess::decode_kernel_invoker(output_data_device+kk*output_numbox*output_numprob, 
                                                   d2is+6*kk, output_numbox, output_numprob, num_classes, 
                                                   confidence_threshold);
        #if DEBUG    
            printf("decoded bboxes.size = %d\n", bboxes.size());
        #endif
        // nms非极大抑制
        std::sort(bboxes.begin(), bboxes.end(), [](vector<float>& a, vector<float>& b){return a[5] > b[5];});
        std::vector<bool> remove_flags(bboxes.size());
        std::vector<vector<float>> box_result;
        box_result.reserve(bboxes.size());

        auto iou = [](const vector<float>& a, const vector<float>& b){
            float cross_left   = std::max(a[0], b[0]);
            float cross_top    = std::max(a[1], b[1]);
            float cross_right  = std::min(a[2], b[2]);
            float cross_bottom = std::min(a[3], b[3]);

            float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
            float union_area = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]) 
                            + std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]) - cross_area;
            if(cross_area == 0 || union_area == 0) return 0.0f;
            return cross_area / union_area;
        };

        for(int i = 0; i < bboxes.size(); ++i){
            if(remove_flags[i]) continue;

            auto& ibox = bboxes[i];
            box_result.emplace_back(ibox);
            for(int j = i + 1; j < bboxes.size(); ++j){
                if(remove_flags[j]) continue;

                auto& jbox = bboxes[j];
                if(ibox[4] == jbox[4]){
                    // class matched
                    if(iou(ibox, jbox) >= nms_threshold)
                        remove_flags[j] = true;
                }
            }
        }
        #if DEBUG  
            printf("box_result.size = %d\n", box_result.size());
        #endif
        for(int i = 0; i < box_result.size(); ++i){
            auto& ibox = box_result[i];
            float left = ibox[0];
            float top = ibox[1];
            float right = ibox[2];
            float bottom = ibox[3];
            int class_label = ibox[4];
            float confidence = ibox[5];
            cv::Scalar color;
            tie(color[0], color[1], color[2]) = random_color(class_label);
            cv::rectangle(imgMats[kk], cv::Point(left, top), cv::Point(right, bottom), color, 3);

            auto name      = cocolabels[class_label];
            auto caption   = cv::format("%s %.2f", name, confidence);
            int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(imgMats[kk], cv::Point(left-3, top-33), cv::Point(left + text_width, top), color, -1);
            cv::putText(imgMats[kk], caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        }
        string wt_path = "output/image-draw_"+to_string(input_batch)+"_"+to_string(kk)+".jpg";
        cv::imwrite(wt_path, imgMats[kk]);
    }
    
}


vector<string> get_imgFiles(string folderPath)
{
    std::vector<std::string> fileNames;

    DIR *dir;
    struct dirent *entry;

    if ((dir = opendir(folderPath.c_str())) != NULL) {
        while ((entry = readdir(dir)) != NULL) {
            // 忽略 "." 和 ".." 目录
            if (std::string(entry->d_name) != "." && std::string(entry->d_name) != "..") {
                std::string filePath = folderPath + "/" + entry->d_name;
                fileNames.push_back(filePath);
            }
        }
        closedir(dir);
    } else {
        INFOV("无法打开目录");
    }

    return fileNames;
}

// 缓冲区的最大容量
const int BUFFER_SIZE = 2;

// 缓冲区的头部和尾部指针
int head = 0;
int tail = 0;

// 互斥锁，用于保护对缓冲区的访问
std::mutex mtx;

// 条件变量，用于通知生产者和消费者
std::condition_variable cvar;


// 判断缓冲区是否为空的函数
bool is_buffer_empty() {
    return head == tail;
}

// 判断缓冲区是否为满的函数
bool is_buffer_full() {
    return (tail + 1) % BUFFER_SIZE == head;
}


// 生产者线程的函数
void producer(float* buffer, vector<vector<cv::Mat>> &imgMats, float *d2is_start, 
              vector<string> img_paths, vector<int> &buffer_batchSize, 
              const int batchSize, const int counts) 
{
    int count = 0;
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cvar.wait(lock, []{return !is_buffer_full();});
        vector<string> filePaths;
        cout<<"produce"<<endl;
        for (int i = count; i < count + batchSize && i < img_paths.size(); ++i) {
            filePaths.push_back(img_paths[i]);
        }
        int real_batchSize = filePaths.size();
        buffer_batchSize[tail] = real_batchSize;
        copy_toGPU(buffer, imgMats, d2is_start, filePaths, tail, batchSize);

        tail = (tail + 1) % BUFFER_SIZE;
        lock.unlock();
        cvar.notify_one();
        count += real_batchSize;
        if (count >= counts) {
            break;
        }
    }
}

// 消费者线程的函数
void consumer(float* input_data_device_start, vector<unsigned char> &engine_data,
              vector<vector<cv::Mat>> &imgMats, float *d2is_start, vector<int> &buffer_batchSize,
              const int batch_d2i_size, const int batch_img_size, const int counts) {
    int count = 0;
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cvar.wait(lock, []{return !is_buffer_empty();});
        cout<<"consum"<<endl;
        const int input_batch = buffer_batchSize[head];
        inference(input_data_device_start+head*batch_img_size, engine_data, imgMats[head], 
                  d2is_start+head*batch_d2i_size, input_batch);
        head = (head + 1) % BUFFER_SIZE;
        lock.unlock();
        cvar.notify_one();
        count += input_batch;
        if (count >= counts) {
            break;
        }
    }
}




int main(){
    const char *path = "/home/srb/trtLearning/yolo-dbintegrate/workspace/replaced.onnx";
    const char *engine_path = "/home/srb/trtLearning/yolo-dbintegrate/workspace/engine.trtmodel";
    string dir_Path = "/home/srb/trtLearning/images/VOC2005_1/PNGImages/TUGraz_cars";
    vector<string> img_paths = get_imgFiles(dir_Path);
    if (!access(engine_path, F_OK) == 0)
    {
        if(!build_model(path)){
            return -1;
        }
    }
    
    vector<unsigned char> engine_data = load_file(engine_path);

    int batchSize = 256; 
    int input_channel = 3;
    int input_height  = 640;
    int input_width   = 640;
    int input_numel   = batchSize * input_channel * input_height * input_width;
    const int batch_d2i_size = batchSize*6;
    vector<int> buffer_batchSize(BUFFER_SIZE);

    // 共享的数组作为缓冲区
    float* input_data_device_start;
    float *d2is_start = new float[BUFFER_SIZE*batchSize*6];
    checkRuntime( cudaMalloc((void**)&input_data_device_start, BUFFER_SIZE*input_numel*sizeof(float)) );

    
    GpuTimer timer;
    float time_cost;
    timer.Start();

    vector<vector<cv::Mat>> imgMats(BUFFER_SIZE, vector<cv::Mat>(batchSize));
    const int counts = img_paths.size();
    std::thread t1(producer, input_data_device_start, std::ref(imgMats), d2is_start, 
                   img_paths, std::ref(buffer_batchSize),  batchSize, counts);
    std::thread t2(consumer, input_data_device_start, std::ref(engine_data),
                   std::ref(imgMats), d2is_start, std::ref(buffer_batchSize), batch_d2i_size, 
                   input_numel, counts);
    t1.join();
    t2.join();

    timer.Stop();
    time_cost = timer.Elapsed();
    printf("yolo-integrate: %f msecs.\n", time_cost);

    cudaFree(input_data_device_start);
    delete[] d2is_start;
    return 0;
}
