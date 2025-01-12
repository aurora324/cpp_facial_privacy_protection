#include "YuNet.h"

YuNet::YuNet(const std::string& model_path,
    const cv::Size& input_size,
    float conf_threshold,
    float nms_threshold,
    int top_k,
    int backend_id,
    int target_id)
    : model_path_(model_path), input_size_(input_size),
    conf_threshold_(conf_threshold), nms_threshold_(nms_threshold),
    top_k_(top_k), backend_id_(backend_id), target_id_(target_id)
{
    model = cv::FaceDetectorYN::create(model_path_, "", input_size_, conf_threshold_, nms_threshold_, top_k_, backend_id_, target_id_);
}

void YuNet::setInputSize(const cv::Size& input_size)
{
    input_size_ = input_size;
    model->setInputSize(input_size_);
}

cv::Mat YuNet::infer(const cv::Mat& image)
{
    cv::Mat res;
    model->detect(image, res);
    return res;
}