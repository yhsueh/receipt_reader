#pragma once
#include "opencv2/opencv.hpp"
#include <vector>
#include <iostream>

class digit {
public:


private:
	std::vector <cv::Mat> image_list;
	std::vector <cv::Mat> gradient_list;
};