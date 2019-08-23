#ifndef _include_SPOOF_OPENCV_h_
#define _include_SPOOF_OPENCV_h_
#include "../util/common.h"
#include <opencv2/dnn.hpp>

class cvSpoofNetwork {
private:
	cv::dnn::Net _net;
	float _threshold;

public:
	cvSpoofNetwork();
	~cvSpoofNetwork();
	int init(const std::string &model_path);
	std::vector<float> run(cv::Mat& img, cv::Rect& faceRect);
	void destroy();
};

#endif