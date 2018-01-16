#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include <ros/package.h>
#include <vector>
#include <string>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;


vector< float > get_svm_detector( const Ptr< SVM >& svm )
{
    // get the support vectors
    Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction( 0, alpha, svidx );

    CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
               (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    CV_Assert( sv.type() == CV_32F );

    vector< float > hog_detector( sv.cols + 1 );
    memcpy( &hog_detector[0], sv.ptr(), sv.cols*sizeof( hog_detector[0] ) );
    hog_detector[sv.cols] = (float)-rho;
    return hog_detector;
}

void computeHOGs( const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst)
{
    HOGDescriptor hog;
    
    hog.winSize = wsize;
    hog.cellSize = Size(10,10);
    hog.blockSize = Size(10,10);
    hog.blockStride = Size(5,5);

    cout << "SIZE" << wsize << endl;

    Mat gray;
    vector< float > descriptors;

    for( size_t i = 0 ; i < img_lst.size(); i++ )
    {
        if ( img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height )
        {
            Rect r = Rect(( img_lst[i].cols - wsize.width ) / 2,
                          ( img_lst[i].rows - wsize.height ) / 2,
                          wsize.width,
                          wsize.height);

            hog.compute( img_lst[i](r), descriptors, Size(8,8), Size(0,0));
            gradient_lst.push_back( Mat( descriptors ).clone() );
        }
    }
}

void load_image(vector <int> &label, vector <Mat> &img_list, vector <Mat> &grad_list) {
	int count;

	for(int i = 0; i < 10; i++) {
		count = 0;
		std::string file_path = ros::package::getPath("receipt_reader") + "/Webcam/train_digits/";
		file_path = file_path + std::to_string(i);
		//cout << "FILEPATH" << file_path << endl;
		std::vector<cv::String>filenames;
		cv::String folder = file_path;
		//std::cout << "FILEPATH"<< file_path << std::endl;
		cv::glob(folder, filenames);

		for (auto& img : filenames) {
			count++;
			img_list.push_back(imread(img, IMREAD_GRAYSCALE));
		}

		label.push_back(count);
	}

	computeHOGs(img_list[0].size(), img_list, grad_list);
	cout << "grad size" << grad_list.size() << endl;
}

void convert_to_ml( const vector< Mat > & train_samples, Mat& trainData )
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

void training(vector <int> &labels, Mat &train_data) {
	Ptr< SVM > svm = SVM::create();
    /* Default values to train SVM */
    svm->setCoef0( 0.0 );
    svm->setDegree( 3 );
    svm->setTermCriteria( TermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3 ) );
    svm->setGamma( 0 );
    svm->setKernel( SVM::LINEAR );
    svm->setNu( 0.5 );
    svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?
    svm->setC( 0.01 ); // From paper, soft classifier
    svm->setType( SVM::EPS_SVR ); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
	svm->train( train_data, ROW_SAMPLE, labels );

	HOGDescriptor hog;
	hog.cellSize = Size(10,10);
    hog.blockSize = Size(10,10);
    hog.blockStride = Size(5,5);
	hog.winSize = Size(20,30);
	cout << "HOG" << hog.winSize.width << endl;
	hog.setSVMDetector(get_svm_detector(svm));
	hog.save("obj_det");
}

int main() {
	vector <int> label;
	vector <int> label_vec;
	vector <Mat> img_list;
	vector <Mat> grad_list;
	Mat train_data;
	load_image(label, img_list, grad_list);

	int count = 0;

	for (auto &i : label) {
		label_vec.insert(label_vec.end(), i, count);
		count ++;
	}

	convert_to_ml(grad_list, train_data);
	training(label_vec, train_data);



	return 0;
}