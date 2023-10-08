#include "MYLNODE2.cuh"
#include "NvInfer.h"

#include <cassert>
#include <cstring>
#include <vector>

using std::vector;
using std::string;
using namespace nvinfer1;

void MYLNODE2_inference(const float *x, const float *addNum, const float *mutNum, float *output, const int batchSize, int n, cudaStream_t stream);

// MYLNODE2 plugin的特定常量
namespace{

    const char *MYLNODE2_PLUGIN_VERSION{"1"};// 采用的名称要对应上onnx-tensorrt-release-8.0/builtin_op_importers.cpp:5094行定义的名称
    const char *MYLNODE2_PLUGIN_NAME{"MYLNODE2"};
}

// 静态类字段的初始化
PluginFieldCollection MYLNODE2PluginCreator::mFC{};// FieldCollection 字段收集

vector<PluginField> MYLNODE2PluginCreator::mPluginAttributes;

// 实际注册时，注册的是创建器，交给tensorRT管理
REGISTER_TENSORRT_PLUGIN(MYLNODE2PluginCreator);

// 用于序列化插件的Helper function
template <typename T>
void writeToBuffer(char *&buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// 用于反序列化插件的Helper function
template <typename T>
T readFromBuffer(const char *&buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

MYLNODE2Plugin::MYLNODE2Plugin(const string name)
:mLayerName(name)
{
    #if DEBUG
        std::cout<<"create MYLNODE2Plugin"<<std::endl;
    #endif
}


const char *MYLNODE2Plugin::getPluginType() const noexcept
{
    return MYLNODE2_PLUGIN_NAME;
}

const char *MYLNODE2Plugin::getPluginVersion() const noexcept
{
    return MYLNODE2_PLUGIN_VERSION;
}

int MYLNODE2Plugin::getNbOutputs() const noexcept
{
    return 1;
}

// 获取该层的输出维度是多少
nvinfer1::DimsExprs MYLNODE2Plugin::getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs *inputs, 
    int32_t nbInputs, nvinfer1::IExprBuilder &exprBuilder
) noexcept
{
    // MYLNODE2ping不改变输入尺寸，所以输出尺寸将与输入尺寸相同
    return *inputs;
}

int MYLNODE2Plugin::initialize() noexcept
{
    return 0;
}

// 行性能测试
int MYLNODE2Plugin::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
    const void* const *inputs, void* const *outputs, void *worksapce, cudaStream_t stream
) noexcept
{
    void *output = outputs[0];
    size_t volume = 1;
    for (int i = 1; i < inputDesc->dims.nbDims; i++)
    {
        volume *= inputDesc->dims.d[i];
    }
    mInputVolume = volume;
    MYLNODE2_inference(
        static_cast<const float*>(inputs[0]),
        static_cast<const float*>(inputs[1]),
        static_cast<const float*>(inputs[2]),
        static_cast<float*>(output),
        inputDesc->dims.d[0],
        mInputVolume,
        stream
    );

    return 0;
}

size_t MYLNODE2Plugin::getSerializationSize() const noexcept
{
    return 0;
}

// 该层的参数序列化储存为trtmodel文件
void MYLNODE2Plugin::serialize(void *buffer) const noexcept
{
    char *d = static_cast<char*>(buffer);
    const char *a = d;
    
    assert(d == a+getSerializationSize());
}

// 判断该插件所支持的数据格式和类型
bool MYLNODE2Plugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) 
noexcept
{
    auto type = inOut[pos].type;
    auto format = inOut[pos].format;
    if(type == DataType::kFLOAT && format == PluginFormat::kLINEAR)
        return true;
    return false;
}

void MYLNODE2Plugin::terminate() noexcept {}


// 配置插件格式:告诉你目前这个层所采用的数据格式和类型
void MYLNODE2Plugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int32_t nbInputs,
    const DynamicPluginTensorDesc* out, int32_t nbOutputs
) noexcept
{
    auto type = in->desc.type;
    auto format = in->desc.format;
    assert(nbOutputs == 1);
    assert(type == DataType::kFLOAT);
    assert(format == PluginFormat::kLINEAR);
}

void MYLNODE2Plugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

// 克隆插件
IPluginV2DynamicExt* MYLNODE2Plugin::clone() const noexcept
{
    #if DEBUG
        printf("===================克隆插件MYLNODE2=================\n");
    #endif
    auto plugin = new MYLNODE2Plugin(mLayerName);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void MYLNODE2Plugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* MYLNODE2Plugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// 插件创建器
MYLNODE2PluginCreator::MYLNODE2PluginCreator()
{
    // 收集PluginField的参数
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* MYLNODE2PluginCreator::getPluginName() const noexcept
{
    return MYLNODE2_PLUGIN_NAME;
}

const char* MYLNODE2PluginCreator::getPluginVersion() const noexcept
{
    return MYLNODE2_PLUGIN_VERSION;
}

const PluginFieldCollection* MYLNODE2PluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

// 创建plugin
IPluginV2* MYLNODE2PluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    return new MYLNODE2Plugin(name);
}

// 反序列化插件参数进行创建
IPluginV2* MYLNODE2PluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call MYLNODE1Plugin::destroy()
    return new MYLNODE2Plugin(name);
}

void MYLNODE2PluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* MYLNODE2PluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}