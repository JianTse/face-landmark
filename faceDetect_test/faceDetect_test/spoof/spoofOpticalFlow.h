#ifndef _SPOOF_OPTICALFLOW_H_
#define _SPOOF_OPTICALFLOW_H_
#include "..\util\common.h"

/*
基于光流的活体检测方法
1：计算人脸附近的稠密光流
2：统计眼睛和嘴巴区域内的光流场直方图
3：统计人脸附近除了步骤2以外区域的光流场直方图
4：如果3的直方图为空且2不为空，则是live
5：如果2,3直方图都不为空，计算两者相似度，相似度很低，则为live
6：其他情况均为spoof
*/
class SpoofOpticalFlow
{
public:
	SpoofOpticalFlow();
	~SpoofOpticalFlow();

	/*
	返回
	0：false
	1：true
	-1：unknow
	*/
	int  update(cv::Mat& lastGray, cv::Mat& curGray, cv::Rect& faceRect, std::vector<cv::Point>& ldmark5);

private:

	cv::Rect getFaceSanp(cv::Mat& lastGray, cv::Mat& curGray, cv::Rect& faceRect,
		std::vector<cv::Point>& ldmark5, cv::Mat& imgLast, cv::Mat& imgCur);

	int  getRectHist(cv::Mat& flow, cv::Rect& outRect, std::vector<cv::Rect>& inRects,
		std::vector<double>& outHist, std::vector<double>& inHist);
	int isInRect(cv::Rect& rect, cv::Point& pt);
	int isInRects(std::vector<cv::Rect>& rects, cv::Point& pt);
	void  updateHist(std::vector<double>& hist, double& angle, double& hypotenuse);
	void normalHist(std::vector<double>& hist);
	double computeSimilarity(std::vector<double>& hist1, std::vector<double>& hist2);
	void  drawHist(std::vector<double>& hist, char* name);
	int  isLiveFace(cv::Mat& img, cv::Mat& flow, cv::Rect& faceRect, std::vector<cv::Point>& ldmarks);
	
	void  showDetect(cv::Mat& img, cv::Rect& rect, std::vector<cv::Point>& ldmarks);
};

#endif