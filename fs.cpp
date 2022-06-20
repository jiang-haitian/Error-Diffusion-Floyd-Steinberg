#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>

#include "opencv2/core.hpp"

#include "fs.hpp"

cv::Mat error_diffusion_fs(cv::Mat& ct){
    assert(ct.type() == CV_8UC1);
    cv::Mat ht(ct.size(), CV_8UC1);

    const double threshold = 127.5;
    double* carry_line_0 = (double*)calloc(ct.cols+2, sizeof(double));
    double* carry_line_1 = (double*)calloc(ct.cols+2, sizeof(double));

    for (int h=0; h<ht.rows; h++){
        for (int w=0; w<ht.cols; w++){
            uint8_t ct_value = ct.at<uint8_t>(h, w);
            double ct_absorb = (double)ct_value + carry_line_0[w+1];
            double error;
            if (ct_absorb > threshold){
                error = ct_absorb - 255.;
                ht.at<uint8_t>(h, w) = 255;
            }else{
                error = ct_absorb;
                ht.at<uint8_t>(h, w) = 0;
            }

            carry_line_0[w+1+1] += error * (7.0 / 16.0);
            carry_line_1[w+1-1] += error * (3.0 / 16.0);
            carry_line_1[w+1]   += error * (5.0 / 16.0);
            carry_line_1[w+1+1] += error * (1.0 / 16.0);
        }
        std::swap(carry_line_0, carry_line_1);
        memset(carry_line_1, 0, (ht.cols+2)*sizeof(double));
    }

    free(carry_line_0);
    free(carry_line_1);

    return ht;
}
