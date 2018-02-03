#pragma once
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "receipt_reader/Confirm.h"
#include <vector>
#include <iostream>

class image_obj {
public:
	image_obj();
	bool confirm_service(receipt_reader::Confirm::Request &req,
							receipt_reader::Confirm::Response &res);
	void callback(const sensor_msgs::ImageConstPtr& msg);
	void computeHOGs( const cv::Size wsize, const std::vector< cv::Mat > & img_lst, std::vector< cv::Mat > & gradient_lst);
	void image_process();
	void digit_localization();
	int classification(std::vector <cv::Mat> & digits);
	bool confirm_flag;
private:
	std::vector < std::vector < cv::Point > > contours;
	cv::HOGDescriptor hog;
	cv::Mat image;
};