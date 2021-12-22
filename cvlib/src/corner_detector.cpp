/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

corner_detector_fast::corner_detector_fast()
{
    create_pattern_pairs();
}

void corner_detector_fast::fill_full_mask(int row, int col)
{
    int16_t center = img.at<int16_t>(row, col);   
    pix_mask.p2 = std::abs(img.at<int16_t>(row + circle_pix_y[1], col + circle_pix_x[1]) - center) > threshold;
    pix_mask.p3 = std::abs(img.at<int16_t>(row + circle_pix_y[2], col + circle_pix_x[2]) - center) > threshold;
    pix_mask.p4 = std::abs(img.at<int16_t>(row + circle_pix_y[3], col + circle_pix_x[3]) - center) > threshold;
    pix_mask.p6 = std::abs(img.at<int16_t>(row + circle_pix_y[5], col + circle_pix_x[5]) - center) > threshold;
    pix_mask.p7 = std::abs(img.at<int16_t>(row + circle_pix_y[6], col + circle_pix_x[6]) - center) > threshold;
    pix_mask.p8 = std::abs(img.at<int16_t>(row + circle_pix_y[7], col + circle_pix_x[7]) - center) > threshold;
    pix_mask.p10 = std::abs(img.at<int16_t>(row + circle_pix_y[9], col + circle_pix_x[9]) - center) > threshold;
    pix_mask.p11= std::abs(img.at<int16_t>(row + circle_pix_y[10], col + circle_pix_x[10]) - center) > threshold;
    pix_mask.p12 = std::abs(img.at<int16_t>(row + circle_pix_y[11], col + circle_pix_x[11]) - center) > threshold;
    pix_mask.p14 = std::abs(img.at<int16_t>(row + circle_pix_y[13], col + circle_pix_x[13]) - center) > threshold;
    pix_mask.p15 = std::abs(img.at<int16_t>(row + circle_pix_y[14], col + circle_pix_x[14]) - center) > threshold;
    pix_mask.p16 = std::abs(img.at<int16_t>(row + circle_pix_y[15], col + circle_pix_x[15]) - center) > threshold;
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    keypoints.clear();
    // \todo implement FAST with minimal LOCs(lines of code), but keep code readable.

    cv::cvtColor(image.getMat(), img, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(img, img, cv::Size(3,3), 0.1, 0.1);
    img.convertTo(img, CV_16SC1);
    
    int16_t center = 0;
    uint16_t mask9 = 511;

    for(int i = 3; i < img.cols - 3; i += stride)
        for(int j = 3; j < img.rows - 3; j += stride)
        {
            center = img.at<int16_t>(j, i);   
            pix_mask.p1 = std::abs(img.at<int16_t>(j + circle_pix_y[0], i + circle_pix_x[0]) - center) > threshold;
            pix_mask.p5 = std::abs(img.at<int16_t>(j + circle_pix_y[4], i + circle_pix_x[4]) - center) > threshold;
            pix_mask.p9 = std::abs(img.at<int16_t>(j + circle_pix_y[8], i + circle_pix_x[8]) - center) > threshold;
            pix_mask.p13 = std::abs(img.at<int16_t>(j + circle_pix_y[12], i + circle_pix_x[12]) - center) > threshold;

            if( (pix_mask.mask & mask_p1_5_9) == mask_p1_5_9 ||
                (pix_mask.mask & mask_p1_5_13) == mask_p1_5_13 ||
                (pix_mask.mask & mask_p1_9_13) == mask_p1_9_13 ||
                (pix_mask.mask & mask_p5_9_13) == mask_p5_9_13    )
            {
                fill_full_mask(j, i);
                for (int k = 0; k < 15; k++)
                {
                    if(pix_mask.mask & mask9 == mask9)
                    {
                        keypoints.emplace_back(i, j, 3 * 2 + 1, 0, 0, 0, 3);
                        mask9 = 511;
                        break;
                    }
                    mask9 <<= 1;
                     
                    if(k >= 7) mask9++;
                }
            }
            pix_mask.mask = 0;
        }

}

void corner_detector_fast::create_pattern_pairs()
{
    num_pairs = griid_size * griid_size * (griid_size * griid_size - 1) * 0.5;
    pattern_pairs.reserve(num_pairs);

    int center = griid_size / 2;
    cv::Point2i p1, p2;

    std::vector<cv::Point2i> all_point;
    for(int i = 0; i < griid_size; i++)
        for(int j = 0; j < griid_size; j++)
                all_point.emplace_back((i - center)*mean_area_size, (j - center)*mean_area_size);
    
    for(int i = 0; i < griid_size*griid_size - 1; i++)
        for(int j = i+1; j < griid_size*griid_size ; j++)
            pattern_pairs.emplace_back(all_point[i], all_point[j]); 
}

float corner_detector_fast::calc_mean_area(cv::Point2i point)
{
    float ret = 0, coef = 1.0/(mean_area_size*mean_area_size);

    int center = mean_area_size/2;
    for(int i = 0; i < mean_area_size; i++)
        for(int j = 0; j < mean_area_size; j++)
            ret += static_cast<float>(img.at<int16_t>(point.y + (i - center), point.x  + (j - center))) * coef;
    return ret;
}

void corner_detector_fast::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    //std::srand(unsigned(std::time(0))); // \todo remove me
    // \todo implement any binary descriptor
    cv::cvtColor(image.getMat(), img, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(img, img, cv::Size(3,3), 0.1, 0.1);
    img.convertTo(img, CV_16SC1);

    descriptors.create(static_cast<int>(keypoints.size()), num_pairs, CV_8U);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    float mean1 = 0, mean2 = 0;
    uint8_t* ptr = reinterpret_cast<uint8_t*>(desc_mat.ptr());
    for (const auto& pt : keypoints)
    {
        for (int i = 0; i < num_pairs; ++i)
        {
            mean1 = calc_mean_area(pattern_pairs[i].first);
            mean2 = calc_mean_area(pattern_pairs[i].second);
            if(mean1 > mean2)
                *ptr = 1;
            else 
                *ptr = 0;
            ++ptr;
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray, cv::InputArray, std::vector<cv::KeyPoint>&, cv::OutputArray descriptors, bool /*= false*/)
{
    // \todo implement me
}
} // namespace cvlib
