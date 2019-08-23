#ifndef _include_NIR_DEPTH_h_
#define _include_NIR_DEPTH_h_
#include "../util/common.h"
#include "../mtcnn_facedetect/mtcnn_opencv.h"
#include "../mtcnn_facedetect/mtcnn_ncnn.h"

class SpoofNirDepth {
private:
	int check_triangle(std::vector<cv::Vec3i>& faceData);
	cv::Rect getFaceSanp(cv::Mat& img, cv::Rect& faceRect, cv::Mat& imgPatch);

public:
	SpoofNirDepth();
	~SpoofNirDepth();

	/*
	����
	0��false
	1��true
	-1��data invalid
	*/
	int depth_liveness_detection(cv::Mat& bgrImg, cv::Mat& depthImg, cv::Rect& faceRect, std::vector<cv::Point>& ldmark5);
	int  nir_liveness_detection(MTCNN_NCNN&  detector, cv::Mat& bgrImg, cv::Mat& nirImg, cv::Rect& faceRect);
};

#endif