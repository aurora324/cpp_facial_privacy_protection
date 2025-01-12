#include "opencv2/opencv.hpp"
#include "YuNet.h"
#include "util.h"
#include "config.h"

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

#include <thread>
#include <mutex>

std::string userInput;
std::mutex mtx;

void handleUserInput() {

        std::string input;
        std::cout << "input:";
        std::cin >> input;
        {
            std::lock_guard<std::mutex> lock(mtx);
            userInput = input;
        }
    
}
int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv,
        "{help  h           |                                                                   | Print this message}"
        "{input_base ib     | E:/Visual_Studio/code/privacy/resource/figure/                    | Set input to a certain image, omit if using camera}"
        "{input i           |                                                                   | Set input to a certain image, omit if using camera}"
        "{model_base mb     | E:/Visual_Studio/code/privacy/resource/face_detection_yunet/      | Set base path to the model}"
        "{model m           | face_detection_yunet_2023mar_int8.onnx                            | Set path to the model}"
        "{result r          | E:/Visual_Studio/code/privacy/resource/result/                    | Set base path to the model}"
        "{mask ma           | 2                                                                 | Set base path to the model}"
        "{backend b         | opencv                                                            | Set DNN backend}"
        "{target t          | cpu                                                               | Set DNN target}"
        "{save s            | true                                                              | Whether to save result image or not}"
        "{vis v             | true                                                              | Whether to visualize result image or not}"
        /* model params below*/     
        "{conf_threshold    | 0.6                                                               | Set the minimum confidence for the model to identify a face. Filter out faces of conf < conf_threshold}"
        "{nms_threshold     | 0.3                                                               | Set the threshold to suppress overlapped boxes. Suppress boxes if IoU(box1, box2) >= nms_threshold, the one of higher score is kept.}"
        "{top_k             | 5000                                                              | Keep top_k bounding boxes before NMS. Set a lower value may help speed up postprocessing.}"
        "{mode              | mask                                                              | state of output normal blur pixel mask}"
        "{blur_kernel_size  | 5                                                                 | blur_kernel_size 5 7 9}"
        "{pixel_size        | 10                                                                | blur_kernel_size 10 30 50}"
    );

    parser.printMessage();

    std::string input_path_base = parser.get<std::string>("input_base");
    std::string input_path = input_path_base + parser.get<std::string>("input");
    std::string model_path_base = parser.get<std::string>("model_base");
    std::string model_path = model_path_base + parser.get<std::string>("model");
    std::string result_path = parser.get<std::string>("result") + parser.get<std::string>("input");
    std::string mask_path = parser.get<std::string>("input_base") + "mask" + parser.get<std::string>("mask") + ".jpg";
    userInput = mask_path;
    std::string backend = parser.get<std::string>("backend");
    std::string target = parser.get<std::string>("target");
    bool save_flag = parser.get<bool>("save");
    bool vis_flag = parser.get<bool>("vis");
    std::string mode = parser.get<std::string>("mode");


    // model params
    float conf_threshold = parser.get<float>("conf_threshold");
    float nms_threshold = parser.get<float>("nms_threshold");
    int top_k = parser.get<int>("top_k");
    int blur_kernel_size = parser.get<int>("blur_kernel_size");
    int pixel_size = parser.get<int>("pixel_size");
    int index = parser.get<int>("mask");
    const int backend_id = str2backend.at(backend);
    const int target_id = str2target.at(target);

    // Instantiate YuNet
    YuNet model(model_path, cv::Size(320, 320), conf_threshold, nms_threshold, top_k, backend_id, target_id);

    // If input is an image
    if (input_path != input_path_base)
    {
        auto image = cv::imread(input_path);

        // Inference
        model.setInputSize(image.size());
        auto faces = model.infer(image);

        // Print faces
        std::cout << cv::format("%d faces detected:\n", faces.rows);
        for (int i = 0; i < faces.rows; ++i)
        {
            int x1 = static_cast<int>(faces.at<float>(i, 0));
            int y1 = static_cast<int>(faces.at<float>(i, 1));
            int w = static_cast<int>(faces.at<float>(i, 2));
            int h = static_cast<int>(faces.at<float>(i, 3));
            float conf = faces.at<float>(i, 14);
            std::cout << cv::format("%d: x1=%d, y1=%d, w=%d, h=%d, conf=%.4f\n", i, x1, y1, w, h, conf);
        }

        // Draw reults on the input image
        if (save_flag || vis_flag)
        {
            auto res_image = visualize(image, faces, -1.f);
            if (save_flag)
            {
                std::cout << "Results are saved to result.jpg\n";
                cv::imwrite(result_path, res_image);
            }
            if (vis_flag)
            {
                cv::namedWindow(input_path, cv::WINDOW_AUTOSIZE);
                cv::imshow(input_path, res_image);
                cv::waitKey(0);
            }
        }
    }
    else // Call default camera
    {
        int device_id = 0;
        auto cap = cv::VideoCapture(device_id);
        int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        model.setInputSize(cv::Size(w, h));

        auto tick_meter = cv::TickMeter();
        cv::Mat frame;
        while (cv::waitKey(1) < 0)
        {
            if (!std::filesystem::exists(userInput)) {
                if (mask_path == userInput) {
                    std::cout << "Error: Path does not exist." << std::endl;
                }
            }
            else {
                mask_path = userInput;
            }

            bool has_frame = cap.read(frame);
            if (!has_frame)
            {
                std::cout << "No frames grabbed! Exiting ...\n";
                break;
            }

            // Inference
            tick_meter.start();
            cv::Mat faces = model.infer(frame);
            tick_meter.stop();

            // Draw results on the input image
            cv::Mat res_image;
            if (mode == "normal")res_image = visualize(frame, faces, (float)tick_meter.getFPS());
            else if (mode == "blur")res_image = visualize_gaussian(frame, faces, (float)tick_meter.getFPS(), blur_kernel_size * blur_kernel_size);
            else if (mode == "pixel")res_image = visualize_pixel(frame, faces, (float)tick_meter.getFPS(), pixel_size);
            else if (mode =="mask")res_image = visualize_mask(frame, faces, load_mask_image(mask_path), (float)tick_meter.getFPS());
            // Visualize in a new window
            cv::imshow("PRIVACY PROTECTION", res_image);

            tick_meter.reset();

            int key = cv::waitKey(50);
            if (key == 'n') {
                mode = "normal";
            }
            else if (key == 'p') {
                mode = "pixel";
            }
            else if (key == 'b') {
                mode = "blur";
            }
            else if (key == 'm') {
                mode = "mask";
            }
            else if (key == 'c') {
                if(mode == "blur")blur_kernel_size = std::max(5, blur_kernel_size - 2);
                else if(mode == "pixel")pixel_size = std::max(10, pixel_size - 10);
                else if (mode == "mask") {
                    index = std::max(1, index - 1);
                    mask_path = parser.get<std::string>("input_base") + "mask" + std::to_string(index) + ".jpg";
                }
            }
            else if (key == 'v') {
                if (mode == "blur")blur_kernel_size = std::min(9, blur_kernel_size + 2);
                else if (mode == "pixel")pixel_size = std::min(50, pixel_size + 10);
                else if (mode == "mask") {
                    index = std::min(2, index + 1);
                    mask_path = parser.get<std::string>("input_base") + "mask" + std::to_string(index) + ".jpg";
                }
            }
            else if (key == 'u') {
                std::thread inputThread(handleUserInput);
                inputThread.detach();
            }
            else if (key == 'q') {
                break;
            }
        }
    }
    return 0;
}
