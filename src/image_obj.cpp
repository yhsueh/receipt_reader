#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include "image_obj.hpp"
#include "receipt_reader/Confirm.h"
#include "machine_learning.hpp"
#include <vector>
#include <iostream>
#include <string>

using namespace cv;
using namespace cv::ml;
using namespace std;

struct boundingBox {
	boundingBox(cv::Point tl, cv::Point br) :  TL(tl), BR(br) {}
	cv::Point TL, BR;

	bool operator < (const boundingBox&y) {
		return y.TL.x < TL.x;
	}
};

image_obj::image_obj() {    
    hog.winSize = cv::Size(30,20);
    hog.cellSize = cv::Size(10,10);
    hog.blockSize = cv::Size(10,10);
    hog.blockStride = cv::Size(5,5);
    confirm_flag = false;
}

bool image_obj::confirm_service(receipt_reader::Confirm::Request &req,
								receipt_reader::Confirm::Response &res) {

	confirm_flag = req.input;
	ROS_INFO("Confirmed");
	return true;
}

void image_obj::callback(const sensor_msgs::ImageConstPtr& msg) {
	image = cv_bridge::toCvShare(msg, "bgr8") -> image;
}

void image_obj::image_process() {
	if (!image.empty()) {
		cv::GaussianBlur(image, image, cv::Size(7,7),1);
		rectangle(image, cv::Point(100,200), cv::Point(530,270), cv::Scalar(0,200,0), 2);
		cv::imshow("View", image);
		cv::waitKey(100);
	}
}

void image_obj::computeHOGs( const cv::Size wsize, const std::vector< cv::Mat > & img_lst, std::vector< cv::Mat > & gradient_lst)
{
    cv::HOGDescriptor hog;
    
    hog.winSize = wsize;
    hog.cellSize = Size(10,10);
    hog.blockSize = Size(10,10);
    hog.blockStride = Size(5,5);

    std::cout << "SIZE" << wsize << std::endl;

    cv::Mat gray;
    std::vector< float > descriptors;

    std::cout << "HI2" << std::endl;
    for( size_t i = 0 ; i < img_lst.size(); i++ )
    {
            hog.compute( img_lst[i], descriptors, Size(8,8), Size(0,0));
            gradient_lst.push_back( Mat( descriptors ).clone() );        
    }
    std::cout << "HI3" << std::endl;
}

void convert_to_ml( const std::vector< Mat > & train_samples, Mat& trainData )
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    Mat tmp( 1, cols, CV_32FC1 ); //< used for transposition if needed
    trainData = Mat( rows, cols, CV_32FC1 );

    for( size_t i = 0 ; i < train_samples.size(); ++i )
    {
        CV_Assert( train_samples[i].cols == 1 || train_samples[i].rows == 1 );

        if( train_samples[i].cols == 1 )
        {
            transpose( train_samples[i], tmp );
            tmp.copyTo( trainData.row( (int)i ) );
        }
        else if( train_samples[i].rows == 1 )
        {
            train_samples[i].copyTo( trainData.row( (int)i ) );
        }
    }
}

int image_obj::classification(std::vector <cv::Mat> & digits) {
	std::vector <cv::Mat> grad_list;
	std::vector <float> result;
	computeHOGs(digits[0].size(), digits, grad_list);
	Mat samples;

	std::cout << "HI4" << std::endl;
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(ros::package::getPath("receipt_reader") + "/receipt_readerdigitSVM.yml");
	std::cout << "HI5" << std::endl;

	convert_to_ml(grad_list, samples);

	std::cout << "gradlistsize" << grad_list.size() << "samples size" << samples.size() << std::endl;

	//float results = svm->predict(samples);
	svm->predict(samples, result);
	
	cout << "result size" << result.size() << endl;

	for (auto& i : result) {
		cout << "values are" << i << endl;;
	}
	return 0;
}

void image_obj::digit_localization() {
	std::string path= ros::package::getPath("receipt_reader") + "/sample.jpg";
	std::cout << "THE PATH IS" << path << std::endl;
	cv::Mat cropped= cv::imread(path);

	std::vector<cv::Vec4i> hierarchy;
	std::vector<cv::Mat> digits;
	
	//cv::Rect r(100, 200, 430, 70);
	//cv::Mat cropped = image(r).clone();
	cv::Mat cropped_gray, cropped_binary, cropped_edge;
	cv::cvtColor(cropped, cropped_gray, CV_BGR2GRAY);
	cv::Scalar mean = cv::mean(cropped_gray);
	cv::threshold(cropped_gray, cropped_binary, mean[0], 255, 0);

	cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));		
	cv::dilate(cropped_binary, cropped_binary, kernel);
	cv::Canny(cropped_binary, cropped_edge, 1, 3);	
    cv::findContours( cropped_edge, contours, hierarchy,
        cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );

	std::cout << "The number of contours found is: " << contours.size() << std::endl;
	
	int max_x = -100;
	int max_y = -100;
	int min_x = 1000;
	int min_y = 1000;
	int diff_x;
	int diff_y;

	std::vector < boundingBox > bounding_box;
	
	for ( auto&j : contours) {		
		max_x = -100;
		max_y = -100;
		min_x = 1000;
		min_y = 1000;

		for (auto&i : j) {
			if (i.x > max_x) {
				max_x = i.x;
			}
			if (i.y > max_y) {
				max_y = i.y;
			}
			if (i.x < min_x) {
				min_x = i.x;
			}
			if (i.y < min_y) {
				min_y = i.y;
			}		
		}

		diff_x = max_x - min_x;
		diff_y = max_y - min_y;

		if (diff_x < 20 || diff_y < 45) {
			continue;
		}

		if (diff_x > 50) {
			boundingBox box1(cv::Point(min_x, min_y), cv::Point((min_x + max_x)/2, max_y));
			boundingBox box2(cv::Point((min_x + max_x)/2, min_y), cv::Point(max_x, max_y));
			bounding_box.push_back(box1);
			bounding_box.push_back(box2);
		}
		else {
			boundingBox box(cv::Point(min_x, min_y), cv::Point(max_x, max_y));
			bounding_box.push_back(box);
		}
		
	}

	std::sort(bounding_box.begin(),bounding_box.end());

	cv::Mat cropped_digit;
	cv::Scalar color(255,255,0);
	int result;
	int count = 1;
	for (auto&i : bounding_box) {
		if (count > 9) {
			break;
		}
		//cv::rectangle(cropped, i.TL, i.BR, color, 1);
		cv::Rect r(i.TL, i.BR);
		cropped_digit = cropped(r).clone();
		cv::resize(cropped_digit,cropped_digit,cv::Size(20,30));
		//if (count == 2)
		digits.push_back(cropped_digit);
		count ++;
	}

	result = classification(digits);
	//cv::imshow("View", cropped);
	//cv::waitKey();
	cv::destroyAllWindows();
	
}
