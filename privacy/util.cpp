#include "util.h"

// load mask
cv::Mat load_mask_image(const std::string& mask_path)
{
    cv::Mat mask_image = cv::imread(mask_path, cv::IMREAD_COLOR);
    if (mask_image.empty())
    {
        std::cerr << "Error: Unable to load mask image from " << mask_path << std::endl;
        return cv::Mat();
    }
    return mask_image;
}

cv::Mat visualize(const cv::Mat& image, const cv::Mat& faces, float fps = -1.f)
{
    static cv::Scalar box_color{ 0, 255, 0 };
    static std::vector<cv::Scalar> landmark_color{
        cv::Scalar(255,   0,   0), // right eye
        cv::Scalar(0,   0, 255), // left eye
        cv::Scalar(0, 255,   0), // nose tip
        cv::Scalar(255,   0, 255), // right mouth corner
        cv::Scalar(0, 255, 255)  // left mouth corner
    };
    static cv::Scalar text_color{ 0, 255, 0 };

    auto output_image = image.clone();

    if (fps >= 0)
    {
        cv::putText(output_image, cv::format("FPS: %.2f normal", fps), cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);
    }

    for (int i = 0; i < faces.rows; ++i)
    {
        // Draw bounding boxes
        int x1 = static_cast<int>(faces.at<float>(i, 0));
        int y1 = static_cast<int>(faces.at<float>(i, 1));
        int w = static_cast<int>(faces.at<float>(i, 2));
        int h = static_cast<int>(faces.at<float>(i, 3));
        cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2);

        // Confidence as text
        float conf = faces.at<float>(i, 14);
        cv::putText(output_image, cv::format("%.4f", conf), cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_DUPLEX, 0.5, text_color);

        // Draw landmarks
        for (int j = 0; j < landmark_color.size(); ++j)
        {
            int x = static_cast<int>(faces.at<float>(i, 2 * j + 4)), y = static_cast<int>(faces.at<float>(i, 2 * j + 5));
            cv::circle(output_image, cv::Point(x, y), 2, landmark_color[j], 2);
        }
    }
    return output_image;
}

cv::Mat visualize_gaussian(const cv::Mat& image, const cv::Mat& faces, float fps = -1.f, int blur_kernel_size = 0)
{
    static cv::Scalar box_color{ 0, 255, 0 };
    static cv::Scalar text_color{ 0, 255, 0 };

    auto output_image = image.clone();

    if (fps >= 0)
    {
        cv::putText(output_image, cv::format("FPS: %.2f blur blur_kernel_size: %d", fps, blur_kernel_size), cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);
    }

    for (int i = 0; i < faces.rows; ++i)
    {
        // Draw bounding boxes
        int x1 = static_cast<int>(faces.at<float>(i, 0));
        int y1 = static_cast<int>(faces.at<float>(i, 1));
        int w = static_cast<int>(faces.at<float>(i, 2));
        int h = static_cast<int>(faces.at<float>(i, 3));
        cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2);

        // Apply Gaussian blur to the face region if blur_kernel_size is greater than 0
        if (blur_kernel_size > 0)
        {
            cv::Mat face_region = output_image(cv::Rect(x1, y1, w, h));
            cv::Mat blurred_face;
            cv::GaussianBlur(face_region, blurred_face, cv::Size(blur_kernel_size, blur_kernel_size), 0);
            blurred_face.copyTo(face_region);
        }

        // Confidence as text
        float conf = faces.at<float>(i, 14);
        cv::putText(output_image, cv::format("%.4f", conf), cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_DUPLEX, 0.5, text_color);
    }
    return output_image;
}

cv::Mat visualize_pixel(const cv::Mat& image, const cv::Mat& faces, float fps = -1.f, int pixel_size = 0)
{
    static cv::Scalar box_color{ 0, 255, 0 };
    static cv::Scalar text_color{ 0, 255, 0 };

    auto output_image = image.clone();

    if (fps >= 0)
    {
        cv::putText(output_image, cv::format("FPS: %.2f pixel pixel_size: %d", fps, pixel_size), cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);
    }

    for (int i = 0; i < faces.rows; ++i)
    {
        // Draw bounding boxes
        int x1 = static_cast<int>(faces.at<float>(i, 0));
        int y1 = static_cast<int>(faces.at<float>(i, 1));
        int w = static_cast<int>(faces.at<float>(i, 2));
        int h = static_cast<int>(faces.at<float>(i, 3));
        cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2);

        // Extract the face region from the image
        cv::Mat face_region = output_image(cv::Rect(x1, y1, w, h));

        // Shrink the face region
        cv::Mat small_face;
        cv::resize(face_region, small_face, cv::Size(w / pixel_size, h / pixel_size), 0, 0, cv::INTER_LINEAR);

        // Enlarge the face region back to the original size
        cv::Mat pixelated_face;
        cv::resize(small_face, pixelated_face, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);

        // Put the pixelated face back into the output image
        pixelated_face.copyTo(face_region);

        // Confidence as text
        float conf = faces.at<float>(i, 14);
        cv::putText(output_image, cv::format("%.4f", conf), cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_DUPLEX, 0.5, text_color);
    }
    return output_image;
}

cv::Mat visualize_mask(const cv::Mat& image, const cv::Mat& faces, const cv::Mat& mask_image, float fps = -1.f)
{
    static cv::Scalar box_color{ 0, 255, 0 };
    static cv::Scalar text_color{ 0, 255, 0 };

    auto output_image = image.clone();

    if (fps >= 0)
    {
        cv::putText(output_image, cv::format("FPS: %.2f mask", fps), cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);
    }

    for (int i = 0; i < faces.rows; ++i)
    {
        // Draw bounding boxes
        int x1 = static_cast<int>(faces.at<float>(i, 0));
        int y1 = static_cast<int>(faces.at<float>(i, 1));
        int w = static_cast<int>(faces.at<float>(i, 2));
        int h = static_cast<int>(faces.at<float>(i, 3));
        cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2);

        // Resize the mask image to fit the face region
        cv::Mat resized_mask;
        cv::resize(mask_image, resized_mask, cv::Size(w, h));

        // Copy the resized mask image to the face region
        resized_mask.copyTo(output_image(cv::Rect(x1, y1, w, h)));

        // Confidence as text
        float conf = faces.at<float>(i, 14);
        cv::putText(output_image, cv::format("%.4f", conf), cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_DUPLEX, 0.5, text_color);
    }
    return output_image;
}