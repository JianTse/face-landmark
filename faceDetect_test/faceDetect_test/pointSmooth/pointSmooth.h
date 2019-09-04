#ifndef __POINT_SMOOTH_H__
#define __POINT_SMOOTH_H__

/************************************************************************
*	头文件
************************************************************************/
#include "opencv/cv.h"
#include "../util/common.h"
#include "opencv2/opencv.hpp" 
#include "tracker.h"
using namespace std;

class CPointSmooth
{
public:
	CPointSmooth();
	~CPointSmooth();
	cv::Point  update(cv::Point& srcPt);
private:
	cv::Point2f  _center;
	float  _dev;
	float  _thresh;
	float  _alpha;	
};

class CLdmarkLKSmooth
{
public:
	CLdmarkLKSmooth();
	~CLdmarkLKSmooth();
	void  trackLdmarks(cv::Mat& img, std::vector<cv::Point>& srcPts, std::vector<cv::Point>& dstPts);
	void  trackKeyPoints(cv::Mat& img, std::vector<cv::Point>& srcPts, std::vector<float>& evas, std::vector<cv::Point>& dstPts);
private:
	int inited;

	cv::Mat _lastImg;
	std::vector<cv::Point2f> _lastPts;
	std::vector<cv::Point2f> _curPts;
	std::vector<cv::Point2f> pointsFB;
	std::vector<uchar> status;
	std::vector<uchar> FB_status;
	std::vector<float> similarity;
	std::vector<float> FB_error;
	cv::Size window_size = cv::Size(9, 9);
	int level = 3;
	cv::TermCriteria term_criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.03);
	float lambda = 0.5;
	float fbmed;

	void normCrossCorrelation(const cv::Mat& img1, const cv::Mat& img2, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2);
	bool filterPts(vector<cv::Point2f>& srcPoints1, vector<cv::Point2f>& srcPoints2,
		vector<cv::Point2f>& dstPoints1, vector<cv::Point2f>& dstPoints2);

	//计算平均位移量
	cv::Point2f clcAveOffset(std::vector<cv::Point2f>& lastValPts, std::vector<cv::Point2f>&curValPts);

	//用圆滤波
	std::vector<CPointSmooth>  _points;
	void  gaussFilter(std::vector<cv::Point>& srcPts, std::vector<cv::Point>& dstPts);

	//用跟踪试试
	std::vector<Tracker> _trackers;
};


class CLdmarkPoseSmooth
{
public:
	CLdmarkPoseSmooth();
	~CLdmarkPoseSmooth();
	void  updateLdmarks(std::vector<float>& evas, std::vector<cv::Point>& srcPts, std::vector<cv::Point>& dstPts);

private:
	int inited;
	std::vector<cv::Point> outLinePts_last, outLinePts_cur;
	std::vector<cv::Point> leftEyePts_last, leftEyePts_cur;
	std::vector<cv::Point> rightEyePts_last, rightEyePts_cur;
	std::vector<cv::Point> mouthPts_last, mouthPts_cur;
	std::vector<float> last_evas;

	void  getOutlinePts(std::vector<cv::Point>& srcPts, std::vector<cv::Point>& dstPts);
	void  getLeftEyePts(std::vector<cv::Point>& srcPts, std::vector<cv::Point>& dstPts);
	void  getRightEyePts(std::vector<cv::Point>& srcPts, std::vector<cv::Point>& dstPts);
	void  getMouthEyePts(std::vector<cv::Point>& srcPts, std::vector<cv::Point>& dstPts);

	void  setOutlinePts(std::vector<cv::Point>& srcPts, std::vector<cv::Point>& dstPts);
	void  getAveOffset(std::vector<cv::Point>& pts1, std::vector<cv::Point>& pts2, cv::Point2f& offset);
	void  addAveOffset(std::vector<cv::Point>& srcPts, std::vector<cv::Point>& dstPts, cv::Point2f& offset);

};


#endif /* __ALIGMENT_H__ */