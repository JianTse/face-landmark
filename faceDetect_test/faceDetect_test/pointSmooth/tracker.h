#ifndef _MT_FACE_TRACKER_H_
#define _MT_FACE_TRACKER_H_
#include "opencv2/core/core.hpp"
#include "../util/common.h"
#include  "../util/recttools.hpp"

#define  TMP_SIZE   3
#define  TMP_RECT_SIZE  TMP_SIZE * 2 + 1
#define  ROI_SIZE   7
#define  ROI_RECT_SIZE  ROI_SIZE * 2 + 1

class Tracker
{
public:
	Tracker();
	~Tracker();
	int  init(cv::Mat& img, cv::Point& pt);
	float update(cv::Mat& img, cv::Point& pt);
	
private:
	int _inited;
	cv::Point _tmpPt;
	cv::Size _maxSize;
	cv::Mat _template;
	float  _sim;	
};

#endif