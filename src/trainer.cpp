#include <iostream>
#include <vector>
#include <string>
#include <ros/package.h>
#include <opencv2/opencv.hpp>


void readFile(std::string &file_path) {
	std::vector<cv::String>filenames;
	cv::Mat test;
	test = cv::imread(file_path);
	std::cout << "FP" << file_path << std::endl;
	if ( !test.data) {
		std::cout << "ERROR LOADING" << std::endl;
	}
}	



int main() {
	std::string pkg_path = ros::package::getPath("receipt_reader");
	//package::V_string packages;
	//ros::package::get

	std::string file_path = pkg_path + "/Webcam/digits/1/1-1.jpg";
	readFile(file_path);
	return 0;
}