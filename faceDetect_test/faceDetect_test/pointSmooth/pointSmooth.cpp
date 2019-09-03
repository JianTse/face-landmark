#include "pointSmooth.h"
float median(vector<float> v) {
	int n = floor(v.size() / 2);
	nth_element(v.begin(), v.begin() + n, v.end());
	return v[n];
}

CPointSmooth::CPointSmooth()
{
	_thresh = 2.5;
	_alpha = 0.25;
	_dev = 2.0;
	_center = cv::Point2f(-1000,-1000);
}
CPointSmooth::~CPointSmooth()
{

}
cv::Point  CPointSmooth::update(cv::Point& srcPt)
{
	cv::Point dst = srcPt;

	float  distant = cvGetPointDistance(cv::Point(_center.x,  _center.y), srcPt);

	//如果是抖动，则输出平滑后的值
	if (distant < _dev * _thresh)
	{
		_center.x = _center.x * (1 - _alpha) + srcPt.x * _alpha;
		_center.y = _center.y * (1 - _alpha) + srcPt.y * _alpha;
		_dev = _dev *  (1 - _alpha) + distant * _alpha;
		_dev = std::min(5.0f, _dev);
		_dev = std::max(1.0f, _dev);
		dst = cv::Point(_center.x, _center.y);
	}
	else
	{
		_center.x = srcPt.x;
		_center.y = srcPt.y;
		dst = cv::Point(_center.x, _center.y);
	}
	return dst;
}

CLdmarkLKSmooth::CLdmarkLKSmooth()
{
	inited = 0;
}

CLdmarkLKSmooth::~CLdmarkLKSmooth()
{

}

void  CLdmarkLKSmooth::updateLdmarks(cv::Mat& img, std::vector<cv::Point>& srcPts, std::vector<cv::Point>& dstPts)
{
	dstPts.clear();
	if (inited == 0)
	{
		for (int i = 0; i < srcPts.size(); i++)
		{
			Tracker* _tracker = new Tracker;
			_tracker->init(img, srcPts[i]);
			_trackers.push_back(*_tracker);
			dstPts.push_back(srcPts[i]);
		}
		inited = 1;
	}
	else
	{
		for (int i = 0; i < _trackers.size(); i++)
		{
			cv::Point pt = srcPts[i];
			float sim = _trackers[i].update(img, pt);
			float  distant = cvGetPointDistance(srcPts[i], pt);
			if (sim < 0.65 || distant > 5)
			{
				dstPts.push_back(srcPts[i]);
			}
			else
			{
				dstPts.push_back(pt);
			}
		}

		//全部重新初始化
		for (int i = 0; i < _trackers.size(); i++)
		{
			_trackers[i].init(img, dstPts[i]);
		}
	}
	_lastImg = img.clone();
}

void  CLdmarkLKSmooth::getAveOffset(std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2, cv::Point2f& offset)
{
	offset = cv::Point2f(0, 0);
	int size = 0;
	for (int i = 0; i < pts1.size(); i++)
	{
		if (pts1[i].x < 0) continue;

		size++;
		offset.x += (pts1[i].x - pts2[i].x);
		offset.y += (pts1[i].y - pts2[i].y);
	}
	offset.x /= size;
	offset.y /= size;
}

void  CLdmarkLKSmooth::addAveOffset(std::vector<cv::Point2f>& srcPts, cv::Point2f& offset, std::vector<cv::Point2f>& dstPts)
{
	dstPts.clear();
	for (int i = 0; i < srcPts.size(); i++)
	{
		float x = srcPts[i].x + offset.x;
		float y = srcPts[i].y + offset.y;
		dstPts.push_back(cv::Point2f(x, y));
	}
}

void CLdmarkLKSmooth::trackKeyPoints(cv::Mat& img, std::vector<cv::Point>& srcPts, std::vector<float>& evas, std::vector<cv::Point>& dstPts)
{
	dstPts.clear();
	if (_lastImg.empty())
	{
		_lastImg = img.clone();		
		for (int i = 0; i < srcPts.size(); i++)
		{
			int x = srcPts[i].x;
			int y = srcPts[i].y;
			dstPts.push_back(cv::Point(x, y));
			_lastPts.push_back(cv::Point2f(x, y));
		}
		return;
	}

	//用中值光流跟踪一把，得到当前跟踪的结果
	cv::calcOpticalFlowPyrLK(_lastImg, img, _lastPts, _curPts, status, similarity, window_size, level, term_criteria, lambda, 0);
	cv::calcOpticalFlowPyrLK(img, _lastImg, _curPts, pointsFB, FB_status, FB_error, window_size, level, term_criteria, lambda, 0);

	//Compute the real FB-error
	for (uint i = 0; i<_lastPts.size(); ++i) {
		FB_error[i] = cv::norm(pointsFB[i] - _lastPts[i]);
	}
	//Filter out points with FB_error[i] > median(FB_error) && points with sim_error[i] > median(sim_error)
	normCrossCorrelation(_lastImg, img, _lastPts, _curPts);

	//得到匹配点对
	vector<cv::Point2f> lastValPts, curValPts;
	filterPts(_lastPts, _curPts, lastValPts, curValPts);

	//求变换矩阵
	cv::Mat matrix = cv::findHomography(lastValPts, curValPts);

	//将上一帧的特征点变换到当前帧
	_curPts.resize(_lastPts.size());
	cv::perspectiveTransform(_lastPts, _curPts, matrix);
	//cv::transform(_lastPts, _curPts, matrix);

	//cv::Mat  _lastImgClone = _lastImg.clone();
	//cv::Mat  _curImgClone = img.clone();
	//for (int i = 0; i < _lastPts.size(); i++)
	//{
	//	cv::circle(_lastImgClone, _lastPts[i], 1, cv::Scalar(0, 255, 0), -1);
	//	cv::circle(_curImgClone, _curPts[i], 1, cv::Scalar(0, 0, 255), -1);
	//}
	//cv::imshow("last", _lastImgClone);
	//cv::imshow("cur", _curImgClone);

	//比较检测的结果与跟踪的结果的误差
	for (int i = 0; i < _curPts.size(); i++)
	{
		int t_x = _curPts[i].x;
		int t_y = _curPts[i].y;
		float dist = cvGetPointDistance(cv::Point(t_x,t_y), srcPts[i]);
		if (dist < 5)
		{
			dstPts.push_back(cv::Point(t_x, t_y));
			_lastPts[i] = _curPts[i];
		}
		//else  if (dist < 10)
		//{
		//	int ave_x = (t_x + srcPts[i].x) / 2;
		//	int ave_y = (t_y + srcPts[i].y) / 2;
		//	dstPts.push_back(cv::Point(ave_x, ave_y));
		//	_lastPts[i] = cv::Point(ave_x, ave_y);
		//}
		else
		{
			dstPts.push_back(srcPts[i]);
			_lastPts[i] = cv::Point2f(srcPts[i].x, srcPts[i].y);
		}
	}

	_lastImg = img.clone();
}

void CLdmarkLKSmooth::normCrossCorrelation(const cv::Mat& img1, const cv::Mat& img2, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2) {
	cv::Mat rec0(10, 10, CV_8U);
	cv::Mat rec1(10, 10, CV_8U);
	cv::Mat res(1, 1, CV_32F);
	for (uint i = 0; i < points1.size(); i++) {
		if (status[i] == 1) {
			getRectSubPix(img1, cv::Size(10, 10), points1[i], rec0);
			getRectSubPix(img2, cv::Size(10, 10), points2[i], rec1);
			matchTemplate(rec0, rec1, res, CV_TM_CCOEFF_NORMED);
			similarity[i] = ((float *)(res.data))[0];
		}
		else similarity[i] = 0.0;
	}
	rec0.release();
	rec1.release();
	res.release();
}
bool CLdmarkLKSmooth::filterPts(vector<cv::Point2f>& srcPoints1, vector<cv::Point2f>& srcPoints2, 
	vector<cv::Point2f>& dstPoints1, vector<cv::Point2f>& dstPoints2) {
	//dstPoints1.resize(srcPoints1.size());
	//dstPoints2.resize(srcPoints2.size());
	//for (int i = 0; i < dstPoints1.size(); i++)
	//{
		//dstPoints1[i] = cv::Point2f(-1,-1);
		//dstPoints2[i] = cv::Point2f(-1, -1);
	//}

	//Get Error Medians
	float simmed = median(similarity);
	size_t i, k;
	for (i = k = 0; i<srcPoints2.size(); ++i) {
		if (!status[i])continue;
		if (similarity[i] >= simmed) {
			//points1[k] = points1[i];
			//points2[k] = points2[i];
			FB_error[k] = FB_error[i];
			k++;
		}
	}
	if (k == 0)return false;
	//points1.resize(k);
	//points2.resize(k);
	FB_error.resize(k);
	fbmed = median(FB_error);
	for (i = k = 0; i<srcPoints2.size(); ++i) {
		if (!status[i])continue;
		if (FB_error[i] <= fbmed && similarity[i] >= simmed) {
			//dstPoints1[i] = srcPoints1[i];
			//dstPoints2[i] = srcPoints2[i];
			dstPoints1.push_back(srcPoints1[i]);
			dstPoints2.push_back(srcPoints2[i]);
			k++;
		}
	}
	//points1.resize(k);
	//points2.resize(k);

	


	if (k>0)return true;
	else return false;
}


CLdmarkPoseSmooth::CLdmarkPoseSmooth()
{
}

CLdmarkPoseSmooth::~CLdmarkPoseSmooth()
{
}
void  CLdmarkPoseSmooth::getOutlinePts(std::vector<cv::Point>& srcPts, std::vector<cv::Point>& dstPts)
{
	dstPts.clear();
	for (int i = 0; i <= 16; i++)
	{
		dstPts.push_back(srcPts[i]);
	}
	for (int i = 27; i <= 35; i++)
	{
		dstPts.push_back(srcPts[i]);
	}
	for (int i = 114; i <= 126; i++)
	{
		dstPts.push_back(srcPts[i]);
	}
}
void  CLdmarkPoseSmooth::setOutlinePts(std::vector<cv::Point>& srcPts, std::vector<cv::Point>& dstPts)
{
	int j = 0;
	for (int i = 0; i <= 16; i++,j++)
	{
		dstPts[i] = srcPts[i];
	}
	for (int i = 27; i <= 35; i++, j++)
	{
		dstPts[i] = srcPts[j];
	}
	for (int i = 114; i <= 126; i++, j++)
	{
		dstPts[i] = srcPts[j];
	}
}

void  CLdmarkPoseSmooth::getAveOffset(std::vector<cv::Point>& pts1, std::vector<cv::Point>& pts2, cv::Point2f& offset)
{
	offset = cv::Point2f(0, 0);
	for (int i = 0; i < pts1.size(); i++)
	{
		offset.x += (pts1[i].x - pts2[i].x);
		offset.y += (pts1[i].y - pts2[i].y);
	}
	offset.x /= pts1.size();
	offset.y /= pts1.size();
}

void  CLdmarkPoseSmooth::addAveOffset(std::vector<cv::Point>& srcPts, std::vector<cv::Point>& dstPts, cv::Point2f& offset)
{
	dstPts.clear();
	for (int i = 0; i < srcPts.size(); i++)
	{
		int x = srcPts[i].x + offset.x;
		int y = srcPts[i].y + offset.y;
		dstPts.push_back(cv::Point(x, y));
	}
}


void  CLdmarkPoseSmooth::getLeftEyePts(std::vector<cv::Point>& srcPts, std::vector<cv::Point>& dstPts)
{
}
void  CLdmarkPoseSmooth::getRightEyePts(std::vector<cv::Point>& srcPts, std::vector<cv::Point>& dstPts)
{
}
void  CLdmarkPoseSmooth::getMouthEyePts(std::vector<cv::Point>& srcPts, std::vector<cv::Point>& dstPts)
{
}

void  CLdmarkPoseSmooth::updateLdmarks(std::vector<float>& evas, std::vector<cv::Point>& srcPts, std::vector<cv::Point>& dstPts)
{
	dstPts.assign(srcPts.begin(), srcPts.end());
	getOutlinePts(srcPts, outLinePts_cur);
	if (last_evas.size() != 3)
	{
		last_evas.assign(evas.begin(), evas.end());		
		outLinePts_last.assign(outLinePts_cur.begin(), outLinePts_cur.end());
	}
	if (fabs(evas[0] - last_evas[0]) > 3 || fabs(evas[1] - last_evas[1]) > 5 || fabs(evas[2] - last_evas[2]) > 3)
	{
		last_evas.assign(evas.begin(), evas.end());
		outLinePts_last.assign(outLinePts_cur.begin(), outLinePts_cur.end());
	}
	else
	{
		cv::Point2f offset;
		std::vector<cv::Point> outLinePts_tmp;
		getAveOffset(outLinePts_cur, outLinePts_last, offset);
		addAveOffset(outLinePts_last, outLinePts_tmp, offset);
		outLinePts_last.assign(outLinePts_tmp.begin(), outLinePts_tmp.end());
		setOutlinePts(outLinePts_last, dstPts);

		float alpha = 0.3;
		for (int i = 0; i < last_evas.size(); i++)
		{
			last_evas[i] = last_evas[i] * (1 - alpha) + evas[i] * alpha;
		}		
	}
}