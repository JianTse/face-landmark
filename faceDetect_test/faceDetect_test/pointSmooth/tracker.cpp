#include "tracker.h"
using namespace cv;

Tracker::Tracker()
{
	_inited = -1;
	_maxSize = cv::Size(640, 480);
	_sim = 0.0f;
}

Tracker::~Tracker()
{
}

int  Tracker::init(cv::Mat& img, cv::Point& pt)
{	
	cv::Rect roi(pt.x - TMP_SIZE, pt.y - TMP_SIZE, TMP_RECT_SIZE, TMP_RECT_SIZE);
	roi = roi & cv::Rect(1, 1, _maxSize.width - 1, _maxSize.height - 1);
	cv::resize(img(roi), _template, cv::Size(TMP_RECT_SIZE, TMP_RECT_SIZE));
	_tmpPt = pt;
	_inited = 1;
	return _inited;
}
float Tracker::update(cv::Mat& img, cv::Point& pt)
{
	if (_inited != 1)
		return 0;

	//ËÑË÷·¶Î§
	cv::Rect roi(pt.x - ROI_SIZE, pt.y - ROI_SIZE, ROI_RECT_SIZE, ROI_RECT_SIZE);
	roi = roi & cv::Rect(1, 1, _maxSize.width - 1, _maxSize.height - 1);

	cv::Mat  imgSearch;
	cv::Mat patch = RectTools::subwindow(img, roi, cv::BORDER_REPLICATE);
	cv::resize(patch, imgSearch, cv::Size(ROI_RECT_SIZE, ROI_RECT_SIZE));

	cv::Mat corr_out;
	//cv::matchTemplate(imgSearch, _template, corr_out, CV_TM_CCOEFF_NORMED);
	cv::matchTemplate(imgSearch, _template, corr_out, CV_TM_CCORR);

	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(corr_out, &minVal, &maxVal, &minLoc, &maxLoc);

	pt.x = maxLoc.x + roi.x;
	pt.y = maxLoc.y + roi.y;

#if  0
	cv::imshow("tmp", _template);
	cv::circle(imgSearch, maxLoc, 1, cv::Scalar(0, 255, 0));
	cv::imshow("patch", imgSearch);
	cv::waitKey(0);
#endif

	_sim = maxVal;
	return maxVal;
}
