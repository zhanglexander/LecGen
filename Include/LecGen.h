#pragma once
#include <opencv2/opencv.hpp>
#include <string.h>
#include <stdio.h>
#include <iostream>


class LecGen {
private:
    bool PSNR(const cv::Mat& I1, const cv::Mat& I2);
    bool SSIM(const cv::Mat& mat1, const cv::Mat& mat2);
    bool framesSimilar(const cv::Mat& mat1, const cv::Mat& mat2);
    bool frameExists(std::vector<cv::Mat>& vec, cv::Mat& frame);
    cv::Scalar getMSSIM(const cv::Mat& i1, const cv::Mat& i2);

public:
    std::string mPath;
    std::string mVideoName;
    char mFolder;
    void generate();
    LecGen(std::string mPath, std::string mVideoName, char mFolder);

};