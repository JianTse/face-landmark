#ifndef _SPOOF_OPTICALFLOW_H_
#define _SPOOF_OPTICALFLOW_H_
#include "..\util\common.h"

/*
���ڹ����Ļ����ⷽ��
1���������������ĳ��ܹ���
2��ͳ���۾�����������ڵĹ�����ֱ��ͼ
3��ͳ�������������˲���2��������Ĺ�����ֱ��ͼ
4�����3��ֱ��ͼΪ����2��Ϊ�գ�����live
5�����2,3ֱ��ͼ����Ϊ�գ������������ƶȣ����ƶȺܵͣ���Ϊlive
6�����������Ϊspoof
*/
class SpoofOpticalFlow
{
public:
	SpoofOpticalFlow();
	~SpoofOpticalFlow();

	/*
	����
	0��false
	1��true
	-1��unknow
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