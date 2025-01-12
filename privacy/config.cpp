#include "config.h"

const std::map<std::string, int> str2backend{
    {"cann", cv::dnn::DNN_BACKEND_CANN},
    {"opencv", cv::dnn::DNN_BACKEND_OPENCV},
    {"cuda", cv::dnn::DNN_BACKEND_CUDA},
    {"timvx", cv::dnn::DNN_BACKEND_TIMVX},
};

const std::map<std::string, int> str2target{
    {"cpu", cv::dnn::DNN_TARGET_CPU},
    {"cuda", cv::dnn::DNN_TARGET_CUDA},
    {"npu", cv::dnn::DNN_TARGET_NPU},
    {"cuda_fp16", cv::dnn::DNN_TARGET_CUDA_FP16}
};

