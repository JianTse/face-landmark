#ifndef __COMMON_RECE_REC_H__
#define __COMMON_RECE_REC_H__

#include <iostream>
#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include <stdio.h>      /* printf */
#include <time.h>       /* time_t, struct tm, time, localtime, asctime */
#include <direct.h>
#include<io.h>
#include "recttools.hpp"

void GetTimeString(char *_timeString, int _strMaxLenth);
float cvGetPointDistance(cv::Point a, cv::Point b);
float cvGetPoint2fDistance(cv::Point2f a, cv::Point2f b);
cv::Rect valid_rect(cv::Point pt1, cv::Point pt2, int width, int height);
cv::Rect cvExtendRect(cv::Rect rect,int width,int height);
void cvValidateRect(cv::Size& _maxSize, cv::Rect& _rect);
cv::Rect getSqureRect(cv::Size& maxSize, cv::Rect& rect);
float getRectOverlap(CvRect _rect1, CvRect _rect2);
float getRectMinOverlap(CvRect _rect1, CvRect _rect2);
cv::Rect getBoundingBox(cv::Size maxSize, std::vector<cv::Point>& ldmark68);
int  isValidFrame(cv::Mat& img, int imgType);
void calculateDepthHistogram(float* pHistogram, int histogramSize, cv::Mat& depthData, int colorSize);
void  convDepth2YellowBgr(cv::Mat& depthData, cv::Mat& depthBgr);
void  convDepth2Bgr(cv::Mat& depthData, cv::Mat& depthBgr);
void  convDepth2Gray(cv::Mat& depthData, cv::Mat& depthGray);
float clarityJudge(cv::Mat& img, std::vector<cv::Point>& ldmark68);
void  drawEvas(std::vector<float>& pitchs, std::vector<float>& yaws, std::vector<float>& rolls);

int  fileExist(const std::string file);
int  creatDir(const std::string file);
void getFiles(std::string path, std::vector<std::string>& files);
void string_replace(std::string &strBig, const std::string &strsrc, const std::string &strdst);
#endif /* __COMMON_H__ */
