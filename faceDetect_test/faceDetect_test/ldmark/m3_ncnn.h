#ifndef _M3_LDMARK_NCNN__H__
#define _M3_LDMARK_NCNN__H__

//Created by Jack Yu
#include "../cnn/include/net.h"
#include <string>
#include <vector>
#include <time.h>
#include <algorithm>
#include <map>
#include <iostream>
#include <math.h>
#include "../util/common.h"
using namespace std;

//#include "landmark.h"
//using namespace std;
//using namespace cv;

class M3_Ldmark {
public:
	M3_Ldmark();
	~M3_Ldmark();

	std::vector<cv::Point> run(const cv::Mat &img, cv::Rect& faceRect);
	void init(const std::string& modelDir);
	void set_param(int width, int height, int num_threads);
	void detect(ncnn::Mat& img_, ncnn::Mat& pred);

	int width;
	int height;
	int num_points;
	int num_threads;

	int* get_landmarks() {
		if (!landmarks) {
			return nullptr;
		}
		else {
			return landmarks;
		}
	}

public:
	ncnn::Net model;
	//ncnn::Mat img;
	int *landmarks;

	//Landmark _ldmarker;
	void  refineDlibBox(cv::Rect& box);
	void  refine98Box(cv::Rect& box);
	void  refine127Box(cv::Rect& box);
	cv::Mat cropImage(const cv::Mat &img, cv::Rect r);
};

#endif