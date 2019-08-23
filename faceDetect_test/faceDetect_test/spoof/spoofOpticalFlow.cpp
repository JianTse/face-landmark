#include "spoofOpticalFlow.h"
#define  OPTICAL_WIDTH  160
#define  OPTICAL_HEIGHT  160
#define  HYPO_THRESH      2

SpoofOpticalFlow::SpoofOpticalFlow()
{
}

SpoofOpticalFlow::~SpoofOpticalFlow()
{
}

cv::Rect SpoofOpticalFlow::getFaceSanp(cv::Mat& lastGray, cv::Mat& curGray, cv::Rect& faceRect,
	std::vector<cv::Point>& ldmark5,  cv::Mat& imgLast, cv::Mat& imgCur)
{
	int extend = std::min(faceRect.width, faceRect.height);
	int extendSize = extend / 8;
	extendSize = (extendSize / 4) * 4;
	cv::Rect roi = cvExtendRect(faceRect, extendSize, extendSize);
	cvValidateRect(cv::Size(curGray.cols, curGray.rows), roi);
	float scale_x = (float)roi.width / OPTICAL_WIDTH;
	float scale_y = (float)roi.height / OPTICAL_WIDTH;

	cv::resize(curGray(roi), imgCur, cv::Size(OPTICAL_WIDTH, OPTICAL_HEIGHT));
	cv::resize(lastGray(roi), imgLast, cv::Size(OPTICAL_WIDTH, OPTICAL_HEIGHT));
	faceRect.x = cvRound((float)(faceRect.x - roi.x) / scale_x);
	faceRect.y = cvRound((float)(faceRect.y - roi.y) / scale_y);
	faceRect.width = cvRound((float)(faceRect.width) / scale_x);
	faceRect.height = cvRound((float)(faceRect.height) / scale_y);
	for (int i = 0; i < ldmark5.size(); i++)
	{
		ldmark5[i].x = cvRound((float)(ldmark5[i].x - roi.x) / scale_x);
		ldmark5[i].y = cvRound((float)(ldmark5[i].y - roi.y) / scale_y);
	}
	return roi;
}

int SpoofOpticalFlow::update(cv::Mat& lastGray, cv::Mat& curGray, cv::Rect& faceRect, std::vector<cv::Point>& ldmark5)
{
	double t1 = cvGetTickCount() / (1000.0 * cvGetTickFrequency());

	//裁剪一个人脸出来
	cv::Rect rect = faceRect;
	std::vector<cv::Point> sanpLdmark5;
	cv::Mat imgLast, imgCur;
	sanpLdmark5.assign(ldmark5.begin(), ldmark5.end());
	cv::Rect roi = getFaceSanp(lastGray, curGray, rect, sanpLdmark5, imgLast, imgCur);
	//showDetect(imgCur, faceRect, ldmark5);
	//cv::imshow("patch", imgCur);
	//return 1;

	cv::Mat flow;
	cv::calcOpticalFlowFarneback(imgCur, imgLast, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
	int ret = isLiveFace(imgCur, flow, rect, ldmark5);

	double t2 = cvGetTickCount() / (1000.0 * cvGetTickFrequency());
	printf("%d, %d,  optical time: %f\n", roi.width, roi.height, t2 - t1);

#if   0
	cv::resize(src24(roi), src24Cur, cv::Size(OPTICAL_WIDTH, OPTICAL_HEIGHT));
	for (size_t y = 0; y<src24Cur.rows; y += 5) {
		for (size_t x = 0; x<src24Cur.cols; x += 5) {
			cv::Point2f fxy = flow.at<cv::Point2f>(y, x);
			line(src24Cur, cv::Point(x, y), cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), CV_RGB(0, 255, 0), 1, 8);
		}
	}
	cv::Scalar color = cv::Scalar(0, 0, 255);
	if (ret == 0)  //假
	{
		color = cv::Scalar(0, 0, 255);
	}
	else if (ret == 1) //真
	{
		color = cv::Scalar(0, 255, 0);
	}
	else if (ret == 2) //假
	{
		color = cv::Scalar(0, 0, 255);
	}
	else
	{
		color = cv::Scalar(0, 255, 255);
	}
	cv::rectangle(src24Cur, faceRect, color, 2);
	cv::imshow("optical", src24Cur);
#endif
	return ret;
}

int SpoofOpticalFlow::isInRect(cv::Rect& rect, cv::Point& pt)
{
	int x1 = rect.x;
	int y1 = rect.y;
	int x2 = x1 + rect.width;
	int y2 = y1 + rect.height;
	if (pt.x < x1 || pt.x > x2)
		return 0;
	if (pt.y < y1 || pt.y > y2)
		return 0;
	return 1;
}

int SpoofOpticalFlow::isInRects(std::vector<cv::Rect>& rects, cv::Point& pt)
{
	for (int i = 0; i < rects.size(); i++)
	{
		int ret = isInRect(rects[i], pt);
		if (ret == 1)
			return i;
	}
	return -1;
}

void  SpoofOpticalFlow::updateHist(std::vector<double>& hist, double& angle, double& hypotenuse)
{
	if (angle >= 0 && angle < 30) { hist[0] += hypotenuse; }
	else if (angle >= 30 && angle < 60) { hist[1] += hypotenuse; }
	else if (angle >= 60 && angle < 90) { hist[2] += hypotenuse; }
	else if (angle >= 90 && angle<120) { hist[3] += hypotenuse; }
	else if (angle >= 120 && angle<150) { hist[4] += hypotenuse; }
	else if (angle >= 150 && angle<180) { hist[5] += hypotenuse; }
	else if (angle >= 180 && angle<210) { hist[6] += hypotenuse; }
	else if (angle >= 210 && angle<240) { hist[7] += hypotenuse; }
	else if (angle >= 240 && angle<270) { hist[8] += hypotenuse; }
	else if (angle >= 270 && angle<300) { hist[9] += hypotenuse; }
	else if (angle >= 300 && angle<330) { hist[10] += hypotenuse; }
	else if (angle >= 330 && angle <= 360) { hist[11] += hypotenuse; }
}

void SpoofOpticalFlow::normalHist(std::vector<double>& hist)
{
	double sumVal = 0;
	for (int i = 0; i < hist.size(); i++)
	{
		sumVal += hist[i];
	}
	for (int i = 0; i < hist.size(); i++)
	{
		hist[i] /= sumVal;
	}
}

int  SpoofOpticalFlow::getRectHist(cv::Mat& flow, cv::Rect& outRect, std::vector<cv::Rect>& inRects,
	std::vector<double>& outHist, std::vector<double>& inHist)
{
	outHist.resize(12);
	inHist.resize(12);
	memset(&outHist[0], 0.0, 12);
	memset(&inHist[0], 0.0, 12);

	int x1 = outRect.x;
	int y1 = outRect.y;
	int x2 = x1 + outRect.width;
	int y2 = y1 + outRect.height;

	int  outCount = 0;
	int  inCount = 0;
	for (size_t y = y1; y < y2; y++)
	{
		for (size_t x = x1; x < x2; x++) 
		{
			cv::Point2f fxy = flow.at<cv::Point2f>(y, x);
			if (fxy.x > -HYPO_THRESH && fxy.x < HYPO_THRESH && fxy.y > -HYPO_THRESH && fxy.y < HYPO_THRESH)
				continue;
			int ret = isInRects(inRects, cv::Point(x,y));
			
			//cv::Point src = cv::Point(x, y);
			//cv::Point dst = cv::Point(cvRound(x + fxy.x), cvRound(y + fxy.y));
			//double angle = atan2((double)(src.y - dst.y), (double)(dst.x - src.x)) * 180 / CV_PI;
			//if (angle<0) angle += 360;
			//double hypotenuse = sqrt((src.y - dst.y)*(src.y - dst.y) + (src.x - dst.x)*(src.x - dst.x));

			double angle = atan2((double)-fxy.y, (double)fxy.x) * 180 / CV_PI;
			if (angle<0) angle += 360;
			double hypotenuse = sqrt(fxy.y*fxy.y + fxy.x*fxy.x);

			if (ret >= 0)
			{
				updateHist(inHist, angle, hypotenuse);
				inCount++;
			}
			else
			{
				updateHist(outHist, angle, hypotenuse);
				outCount++;
			}			
		}
	}

	//
	int ret = 0;
	if (inCount > 10)
	{		
		ret = 1;
	}
	if (outCount > 10)
	{
		ret = ret + 2;
	}
	return ret;	
}

double SpoofOpticalFlow::computeSimilarity(std::vector<double>& hist1, std::vector<double>& hist2)
{
	double conf = 0;
	for (unsigned int i = 0; i < hist1.size(); ++i) {
		conf += sqrt(hist1[i] * hist2[i]);
	}
	return conf;
}

void  SpoofOpticalFlow::drawHist(std::vector<double>& hist, char* name)
{
	int scale = 30;
	int hist_height = 200;
	cv::Mat hist_img = cv::Mat::zeros(cv::Size(12 * scale, hist_height), CV_8UC3);
	for (int j = 0; j<hist.size(); j++)
	{
		int intensity = cvRound(hist[j] * hist_height);
		int x = j * scale;
		int y = hist_height - intensity;
		int w = scale;
		int h = intensity;
		cv::Rect rect = cv::Rect(x, y, w, h);
		cv::rectangle(hist_img, rect, cv::Scalar(0, 255, 0), -1);
	}
	cv::imshow(name, hist_img);
}

int  SpoofOpticalFlow::isLiveFace(cv::Mat& img, cv::Mat& flow, cv::Rect& faceRect, std::vector<cv::Point>& ldmarks)
{
	//测试
	cv::Rect outRect = faceRect;
	std::vector<cv::Rect> inRects;
	std::vector<double> outHist;
	std::vector<double> inHist;

	int x1, y1, x2, y2;
	cv::Rect leftEye, rightEye, mouthRect;

	//取需要扩展的边长
	int maxSide = std::max(faceRect.width, faceRect.height) / 4;

	//左眼
	x1 = ldmarks[0].x - maxSide / 2;
	x2 = x1 + maxSide;
	y1 = ldmarks[0].y - maxSide / 2;
	y2 = y1 + maxSide;
	leftEye = cv::Rect(x1,y1,x2-x1,y2-y1);
	cvValidateRect(cv::Size(img.cols, img.rows), leftEye);
	inRects.push_back(leftEye);

	//右眼
	x1 = ldmarks[1].x - maxSide / 2;
	x2 = x1 + maxSide;
	y1 = ldmarks[1].y - maxSide / 2;
	y2 = y1 + maxSide;
	rightEye = cv::Rect(x1, y1, x2 - x1, y2 - y1);
	cvValidateRect(cv::Size(img.cols, img.rows), rightEye);
	inRects.push_back(rightEye);

	//嘴巴
	int y = (ldmarks[3].y + ldmarks[4].y) / 2;
	x1 = ldmarks[3].x - maxSide / 2;
	x2 = ldmarks[4].x + maxSide / 2;
	y1 = y - maxSide / 3;
	y2 = faceRect.y + faceRect.height;
	mouthRect = cv::Rect(x1, y1, x2 - x1, y2 - y1);
	cvValidateRect(cv::Size(img.cols, img.rows), mouthRect);
	inRects.push_back(mouthRect);
	
	//cv::rectangle(img, leftEye, cv::Scalar(0, 0, 0));
	//cv::rectangle(img, rightEye, cv::Scalar(0, 0, 0));
	//cv::rectangle(img, mouthRect, cv::Scalar(0, 0, 0));

	int  ret = getRectHist( flow,  outRect,  inRects, outHist,  inHist);
	if (ret == 0)  //假
	{
		ret = 0;
	}
	else if (ret == 1) //真
	{
		ret = 1;
	}
	else if (ret == 2) //假
	{
		ret = 0;
	}
	else  //需要计算相似度了，不确定
	{		
		normalHist(inHist);
		normalHist(outHist);
		double sim = computeSimilarity(outHist, inHist);
		if (sim < 0.2)
		{
			ret = 1;
		}
		else if (sim > 0.75)
		{
			ret = 0;
		}
		else
		{
			ret = -1;
		}		
	}
	//drawHist(outHist, "out");
	//drawHist(inHist, "in");
	return ret;
}

void  SpoofOpticalFlow::showDetect(cv::Mat& img,  cv::Rect& rect, std::vector<cv::Point>& ldmarks)
{
	cv::rectangle(img, rect, cv::Scalar(0, 255, 0));
	for (int i = 0; i < ldmarks.size(); i++)
	{
		cv::circle(img, ldmarks[i], 2, cv::Scalar(0, 0, 255), 2);
	}
}