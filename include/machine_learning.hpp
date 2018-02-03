#pragma once
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include <ros/package.h>
#include <vector>
#include <string>
#include <iostream>

using namespace cv::ml;
using namespace cv;
using namespace std;

void computeHOGs(const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst);
void training(vector <int> &labels, Mat &train_data);
void convert_to_ml( const vector< Mat > & train_samples, Mat& trainData );
