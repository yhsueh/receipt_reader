#include <iostream>
#include <vector>
#include <string>
#include <ros/package.h>
#include <opencv2/opencv.hpp>

void imgProcess(std::string &&file_path) {	
	std::vector<cv::String>filenames;
	cv::Mat input_image, output_image;
	input_image = cv::imread(file_path);	
	if ( !input_image.data) {
		std::cout << "ERROR LOADING" << std::endl;
	}
	cv::namedWindow("View");
	cv::startWindowThread();
	cv::imshow("View", input_image);
	cv::waitKey(0);
	if (cv::getWindowProperty("View",0) > 0){
		cv::destroyAllWindows();
	}
	cv::resize(input_image, output_image, cv::Size(20,30));
	cv::cvtColor(output_image, output_image, CV_RGB2GRAY);
	cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
	cv::GaussianBlur(output_image, output_image, cv::Size(7,7),1);
	cv::adaptiveThreshold(output_image, output_image, 255,
		cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,13,0);
	cv::dilate(output_image, output_image, kernel);
	for (int i = 0; i < 20; i++) {
		cv::dilate(output_image, output_image, getStructuringElement(cv::MORPH_RECT,
			cv::Size(1,1)));
	}
	cv::imshow("View", output_image);
	cv::waitKey(0);
	if (cv::getWindowProperty("View",0) > 0){
		cv::destroyAllWindows();
	}

	cv::imwrite(file_path, output_image);
}

void getDir(std::string &pkg_path) {
	for(int i = 0; i < 10; i++) {
		std::string file_path = pkg_path + "/Webcam/train_digits/";
		file_path = file_path + std::to_string(i);
		std::vector<cv::String>filenames;
		cv::String folder = file_path;
		std::cout << "FILEPATH"<< file_path << std::endl;
		cv::glob(folder, filenames);

		for (auto& img : filenames) {
			imgProcess(img);
		}
	}
}

int main() {	
	std::string pkg_path = ros::package::getPath("receipt_reader");
	getDir(pkg_path);
	return 0;
}