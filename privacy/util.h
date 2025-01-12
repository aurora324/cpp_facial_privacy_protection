#ifndef UTIL_H
#define UTIL_H

#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <vector>

cv::Mat load_mask_image(const std::string& mask_path);
cv::Mat visualize(const cv::Mat& image, const cv::Mat& faces, float fps);
cv::Mat visualize_gaussian(const cv::Mat& image, const cv::Mat& faces, float fps, int blur_kernel_size);
cv::Mat visualize_pixel(const cv::Mat& image, const cv::Mat& faces, float fps, int pixel_size);
cv::Mat visualize_mask(const cv::Mat& image, const cv::Mat& faces, const cv::Mat& mask_image, float fps);

#endif