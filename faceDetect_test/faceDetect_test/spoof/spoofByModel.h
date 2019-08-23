//
// Created by Lonqi on 2017/11/18.
//
#pragma once

#ifndef __SPOOF_BY_MODEL_H__
#define __SPOOF_BY_MODEL_H__
#include "../cnn/include/net.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <time.h>
#include <algorithm>
#include <map>
#include <iostream>
#include "../util/common.h"
#include <opencv2/dnn.hpp>

class SpoofByModel {

public:
    ~SpoofByModel();
	SpoofByModel();

	int init(const std::string &model_path);
	void destroy();
	std::vector<float> ncnnCheckFace(cv::Mat& img, cv::Rect& faceRect);
	std::vector<float> cvCheckFace(cv::Mat& img, cv::Rect& faceRect);

private:
    
	ncnn::Net _ncnnNet;
	cv::dnn::Net  _cvNet;	
};


#endif //__MTCNN_NCNN_H__
