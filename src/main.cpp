#include "ros/ros.h"
#include "std_msgs/String.h"
#include "image_obj.hpp"
#include "receipt_reader/Confirm.h"
#include <iostream>

void user_input(std::vector<int> &lottery) {
  int num;
  std::cout << "Input the lottery number followed by ENTER." << std::endl;
  std::cout << "Press 0 when finish inputting lottery number." << std::endl;
  
  while(1) {
  	std::cin >> num;
  	if (num == 0) {
  		break;
  	}
  	else {
  		lottery.push_back(num);
  	}
  }
}

bool analyze(std::vector<int> &lottery, int result) {
  for (auto&i : lottery) {
    if (result == i ) {
      return true;
    }
  }
  return false;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "main");
  std::vector <int> lottery;
  image_obj imageObj;
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("/usb_cam/image_raw", 10, &image_obj::callback, &imageObj);
  ros::ServiceServer ss = nh.advertiseService("confirm", &image_obj::confirm_service, &imageObj);
  int detect_value;

  user_input(lottery);
  std::cout << "User input obtained" << std::endl;
  cv::namedWindow("View");
  cv::startWindowThread();
  ros::Rate loop_rate(5);
  while(ros::ok()) {
    if (imageObj.confirm_flag) {
      imageObj.digit_localization();
    	//imageObj.classification();
      //detect_value = imageObj.classification();
    	/*
      std::cout <<"the value returned is" << detect_value << std::endl;
      	analyze(lottery, detect_value);
      	if (analyze) {
      		break;
      	}
      	imageObj.confirm_flag = false;
        */
    }
    imageObj.image_process();
    ros::spinOnce();
    loop_rate.sleep();
  }

  cv::destroyAllWindows();
  return 0;
}