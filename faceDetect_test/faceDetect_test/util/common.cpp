#include "common.h"

void GetTimeString(char *_timeString, int _strMaxLenth)
{
	time_t currTime;
	time(&currTime);
	tm *timeStamp = localtime(&currTime);
	strftime(_timeString, _strMaxLenth, "%Y-%m-%d-%H-%M-%S", timeStamp);
}

float cvGetPointDistance(cv::Point a, cv::Point b)
{
	float d = (float)(a.x - b.x)*(float)(a.x - b.x) + (float)(a.y - b.y)*(float)(a.y - b.y);
	return sqrtf(d);
}

float cvGetPoint2fDistance(cv::Point2f a, cv::Point2f b)
{
	float d = (float)(a.x - b.x)*(float)(a.x - b.x) + (float)(a.y - b.y)*(float)(a.y - b.y);
	return sqrtf(d);
}


cv::Rect valid_rect(cv::Point pt1, cv::Point pt2, int width, int height)
{
	if ((pt1.x > -1) && (pt1.y > -1) && (pt2.x < width) && (pt2.y < height))
	{
		cv::Rect rect_mid;
		rect_mid.x = pt1.x;
		rect_mid.y = pt1.y;
		rect_mid.width = pt2.x - pt1.x;
		rect_mid.height = pt2.y - pt1.y;
		return rect_mid;
	}
	else {
		return cv::Rect(0, 0, 0, 0);
	}
}

cv::Rect cvExtendRect(cv::Rect rect,int width,int height)
{
	cv::Rect extend = cv::Rect(rect.x - width,
											rect.y - height,
											rect.width + width*2,
											rect.height + height*2 );
	return extend;
}

void cvValidateRect(cv::Size& _maxSize, cv::Rect& _rect)
{
	int x1 = _rect.x;
	int y1 = _rect.y;
	int x2 = _rect.x + _rect.width;
	int y2 = _rect.y + _rect.height;
	if (x1 < 0) x1 = 0;
	if (y1 < 0) y1 = 0;
	if (x2 > _maxSize.width-1) x2 = _maxSize.width - 1;
	if (y2 > _maxSize.height-1) y2 = _maxSize.height - 1;
	_rect = cv::Rect(x1,y1,x2-x1,y2-y1);
	return;

	if(_rect.width > _maxSize.width-1)
	{
		_rect.width = _maxSize.width-1;
	}
	if(_rect.height > _maxSize.height-1)
	{
		_rect.height = _maxSize.height-1;
	}

	if(_rect.x < 0)
	{
		_rect.x = 0;
	}
	if(_rect.x+_rect.width >= _maxSize.width-1)
	{
		_rect.x = _maxSize.width-1-_rect.width;
	}
	if(_rect.y < 0)
	{
		_rect.y = 0;
	}
	if(_rect.y+_rect.height > _maxSize.height-1)
	{
		_rect.y = _maxSize.height - 1 - _rect.height;
	}
}

cv::Rect getSqureRect(cv::Size& maxSize, cv::Rect& rect)
{
	int  x1 = rect.x;
	int  y1 = rect.y;
	int  x2 = x1 + rect.width;
	int  y2 = y1 + rect.height;
	int w = x2 - x1 + 1;
	int h = y2 - y1 + 1;
	int maxSide = (h>w) ? h : w;
	x1 = x1 + w*0.5 - maxSide*0.5;
	y1 = y1 + h*0.5 - maxSide*0.5;
	x2 = cvRound(x1 + maxSide - 1);
	y2 = cvRound(y1 + maxSide - 1);
	x1 = std::max(0, x1);
	y1 = std::max(0, y1);
	x2 = std::min(maxSize.width - 1, x2);
	y2 = std::min(maxSize.height - 1, y2);
	cv::Rect ret = cv::Rect(x1, y1, x2 - x1, y2 - y1);
	return ret;
}

float getRectMinOverlap(CvRect _rect1, CvRect _rect2)
{
	if (_rect1.width <= 0 || _rect1.height <= 0 || _rect2.width <= 0 || _rect2.height <= 0)
	{
		return 0.0f;
	}

	float intersection, area1, area2;

	int overlapedWidth = std::min(_rect1.x + _rect1.width, _rect2.x + _rect2.width) - std::max(_rect1.x, _rect2.x);
	int overlapedHeight = std::min(_rect1.y + _rect1.height, _rect2.y + _rect2.height) - std::max(_rect1.y, _rect2.y);

	intersection = overlapedWidth * overlapedHeight;
	if (intersection <= 0 || overlapedWidth <= 0 || overlapedHeight <= 0)
		return 0.0f;

	//return intersection / (area1 + area2 - intersection);

	area1 = _rect1.width * _rect1.height;
	area2 = _rect2.width * _rect2.height;
	float  minArea = std::min(area1, area2);
	return intersection / minArea;
}

float getRectOverlap(CvRect _rect1, CvRect _rect2)
{
	if(_rect1.width <= 0 || _rect1.height <= 0 || _rect2.width <= 0 || _rect2.height <= 0)
	{
		return 0.0f;
	}

	float intersection, area1, area2;

	int overlapedWidth = std::min(_rect1.x+_rect1.width, _rect2.x+_rect2.width) - std::max(_rect1.x,_rect2.x);
	int overlapedHeight = std::min(_rect1.y+_rect1.height, _rect2.y + _rect2.height) - std::max(_rect1.y, _rect2.y);

	intersection = (float)(overlapedWidth * overlapedHeight);
	if(intersection <= 0 || overlapedWidth <= 0 || overlapedHeight <= 0)
		return 0.0f;

	area1 = (float)(_rect1.width * _rect1.height);
	area2 = (float)(_rect2.width * _rect2.height);

	return intersection / (area1 + area2 - intersection);
}

cv::Rect getBoundingBox(cv::Size maxSize, std::vector<cv::Point>& ldmark68)
{
	int min_x, min_y, max_x, max_y;
	int width = maxSize.width;
	int height = maxSize.height;
	min_x = min_y = 100000;
	max_x = max_y = -1;
	for (int i = 0; i < ldmark68.size(); i++)
	{
		if (ldmark68[i].x > max_x) max_x = ldmark68[i].x;
		if (ldmark68[i].x < min_x) min_x = ldmark68[i].x;
		if (ldmark68[i].y > max_y) max_y = ldmark68[i].y;
		if (ldmark68[i].y < min_y) min_y = ldmark68[i].y;
	}
	cv::Rect box = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
	cvValidateRect(cv::Size(width, height), box);
	return box;
}

//判断当前帧是否有效
int  isValidFrame(cv::Mat& img, int imgType)
{
	int width = img.cols;
	int height = img.rows;
	if (imgType == 0)   //nv21
	{
		height = img.rows * 2 / 3;
	}
	else 
	{
		height = img.rows;
	}
	//计算当前帧图像是否有效
	cv::Scalar mean;  //均值
	cv::Scalar stddev;  //标准差	
	cv::Rect  centerRect = cv::Rect(width / 6, height / 6, width * 2 / 3, height * 2 / 3);
	cv::Mat smallPatch;
	cv::resize(img(centerRect), smallPatch, cv::Size(64, 64), 0,0, cv::INTER_NEAREST);
	cv::meanStdDev(smallPatch, mean, stddev);  //计算均值和标准差
	double mean_pxl = mean.val[0];
	double stddev_pxl = stddev.val[0];
	//if(stddev.val[0])
	//LOGI("stddev: %f", stddev_pxl);
	if (stddev.val[0] < 5)
		return 0;
	return 1;
}

void calculateDepthHistogram(float* pHistogram, int histogramSize, cv::Mat& depthData,  int  colorSize)
{
	//const openni::DepthPixel* pDepth = (const openni::DepthPixel*)frame.getData();
	// Calculate the accumulative histogram (the yellow display...)
	memset(pHistogram, 0, histogramSize * sizeof(float));
	//int restOfRow = frame.getStrideInBytes() / sizeof(openni::DepthPixel) - frame.getWidth();
	//int height = frame.getHeight();
	//int width = frame.getWidth();

	unsigned int nNumberOfPoints = 0;
	for (int y = 0; y < depthData.rows; ++y)
	{
		short* ptr = depthData.ptr<short>(y);
		for (int x = 0; x < depthData.cols; ++x)
		{
			if (ptr[x] != 0)
			{
				pHistogram[ptr[x]]++;
				nNumberOfPoints++;
			}
		}
	}
	for (int nIndex = 1; nIndex < histogramSize; nIndex++)
	{
		pHistogram[nIndex] += pHistogram[nIndex - 1];
	}
	if (nNumberOfPoints)
	{
		for (int nIndex = 1; nIndex < histogramSize; nIndex++)
		{
			pHistogram[nIndex] = (colorSize * (1.0f - (pHistogram[nIndex] / nNumberOfPoints)));
		}
	}
}

void  getColorTable(std::vector<cv::Scalar>& colorTables, int lens)
{
	colorTables.clear();
	int maxLen = 256 * 256 * 256;
	int bin = maxLen / lens;
	for (int i = 0; i < lens; i++)
	{
		int  val = i * bin;
		int b = (0x00FF0000 & val) >> 16;
		int g = (0x0000FF00 & val) >> 8;
		int r = (0x000000FF & val);
		cv::Scalar color = cv::Scalar(b,g,r);
		colorTables.push_back(color);
	}
}

void  convDepth2YellowBgr(cv::Mat& depthData, cv::Mat& depthBgr)
{
#define MAX_DEPTH 10000
	float			m_pDepthHist[MAX_DEPTH];
	calculateDepthHistogram(m_pDepthHist, MAX_DEPTH, depthData, 256);
	for (int y = 0; y < depthData.rows; ++y)
	{		
		short* srcPtr = depthData.ptr<short>(y);
		uchar* dstPtr = depthBgr.ptr<uchar>(y);
		for (int x = 0; x < depthData.cols; ++x)
		{
			if (srcPtr[x] != 0)
			{
				int nHistValue = m_pDepthHist[srcPtr[x]];
				dstPtr[3 * x] = 0;
				dstPtr[3 * x + 1] = nHistValue;
				dstPtr[3 * x + 2] = nHistValue;
			}
		}
	}
}

void  convDepth2Bgr(cv::Mat& depthData, cv::Mat& depthBgr)
{
#define MAX_DEPTH 10000
	float			m_pDepthHist[MAX_DEPTH];
	std::vector<cv::Scalar> colorTables;
	int colorLen = 640;	
	cv::Mat  colorMat = cv::imread("sdcard/facerec/model/timg.jpg");
	uchar* ptr = colorMat.ptr<uchar>(100);
	for (int x = 0; x < colorMat.cols; x++)
	{
		uchar b = ptr[x * 3];
		uchar g = ptr[x * 3 + 1];
		uchar r = ptr[x * 3 + 2];
		cv::Scalar color = cv::Scalar(b, g, r);
		colorTables.push_back(color);
	}
	//colorLen = colorTables.size();
	//getColorTable(colorTables, colorLen);
	//calculateDepthHistogram(m_pDepthHist, MAX_DEPTH, depthData, colorLen);

	for (int y = 0; y < depthData.rows; ++y)
	{
		short* srcPtr = depthData.ptr<short>(y);
		uchar* dstPtr = depthBgr.ptr<uchar>(y);
		for (int x = 0; x < depthData.cols; ++x)
		{
			if (srcPtr[x] != 0)
			{
				int nIdx = srcPtr[x] / 40;
				//int nIdx = m_pDepthHist[srcPtr[x]];
				dstPtr[3 * x] = colorTables[nIdx][0];
				dstPtr[3 * x + 1] = colorTables[nIdx][1];
				dstPtr[3 * x + 2] = colorTables[nIdx][2];
			}
		}
	}
}

void  convDepth2Gray(cv::Mat& depthData, cv::Mat& depthGray)
{
	for (int y = 0; y < depthData.rows; ++y)
	{
		short* srcPtr = depthData.ptr<short>(y);
		uchar* dstPtr = depthGray.ptr<uchar>(y);
		for (int x = 0; x < depthData.cols; ++x)
		{
			if (srcPtr[x] != 0)
			{
				int val = srcPtr[x] / 10;
				dstPtr[x] = val;
			}
		}
	}
}

int  fileExist(const std::string file)
{
#if defined WIN32 || defined _WIN32
	if (_access(file.c_str(), 0) == -1)
	{
		return 0;
	}
#else
	if (access(file.c_str(), 0) == -1)
	{
		return 0;
	}
#endif
	return 1;
}

int  creatDir(const std::string file)
{
#if defined WIN32 || defined _WIN32
	if (_mkdir(file.c_str()) == -1)   // 返回 0 表示创建成功，-1 表示失败
	{
		printf("create dir: %s", file.c_str());
		return 0;
	}
#else
	if (mkdir(file.c_str(), 0755) == -1)
	{
		printf("create dir: %s", file.c_str());
		return 0;
	}
#endif
	return 1;
}

void getFiles(std::string path, std::vector<std::string>& files)
{
	//文件句柄 
	long  hFile = 0;
	//文件信息 
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					files.push_back(fileinfo.name);
					//files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

void string_replace(std::string &strBig, const std::string &strsrc, const std::string &strdst)
{
	std::string::size_type pos = 0;
	std::string::size_type srclen = strsrc.size();
	std::string::size_type dstlen = strdst.size();

	while ((pos = strBig.find(strsrc, pos)) != std::string::npos)
	{
		strBig.replace(pos, srclen, strdst);
		pos += dstlen;
	}
}

float clarityJudge(cv::Mat& img, std::vector<cv::Point>& ldmark68)
{
	cv::Rect boundingbox = getBoundingBox(cv::Size(img.cols, img.rows), ldmark68);

#define PATCH_WIDTH  48
#define PATCH_HEIGHT  48
	cv::Mat patch, grayPatch;
	cv::resize(img(boundingbox), patch, cv::Size(PATCH_WIDTH, PATCH_HEIGHT));
	cv::cvtColor(patch, grayPatch, cv::COLOR_BGR2GRAY);

	int intValue[PATCH_HEIGHT][PATCH_WIDTH];
	int sub[(PATCH_HEIGHT << 1) + 1][(PATCH_WIDTH << 1) + 1];

	cv::Mat_<cv::Vec3b>::iterator data = grayPatch.begin<cv::Vec3b>();
	for (int i = 0; i< PATCH_HEIGHT; i++) {
		for (int j = 0; j < PATCH_WIDTH; j++) {
			intValue[i][j] = (int)(*data++)[0];
		}
	}
	//Apply Sobel on x and y directions and add together.
	cv::Mat grad_x, grad_y;
	cv::Mat abs_grad_x, abs_grad_y, dst;
	cv::Sobel(grayPatch, grad_x, CV_16S, 1, 0, 3, 1, 1);
	cv::convertScaleAbs(grad_x, abs_grad_x);
	cv::Sobel(grayPatch, grad_y, CV_16S, 0, 1, 3, 1, 1);
	cv::convertScaleAbs(grad_y, abs_grad_y);
	cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);

	//Scalar mean;  //均值
	//Scalar stddev;  //标准差
	//cv::meanStdDev(dst, mean, stddev);  //计算均值和标准差
	//double mean_pxl = mean.val[0];
	//double stddev_pxl = stddev.val[0];	

	//Create a look-up table.
	cv::Mat_<cv::Vec3b>::iterator it = dst.begin<cv::Vec3b>();
	int counter = 0; //Record the total number of pixels involved.
	int res = 0;
	memset(sub, 0, sizeof(sub));
	for (int i = 1; i < PATCH_WIDTH; i++) {
		sub[1][i << 1] = abs(intValue[0][i] - intValue[0][i - 1]);
	}
	for (int i = 1; i < PATCH_HEIGHT; i++) {
		sub[i << 1][1] = abs(intValue[i][0] - intValue[i - 1][0]);
		for (int j = 1; j < PATCH_WIDTH; j++) {
			sub[(i << 1) + 1][j << 1] = abs(intValue[i][j] - intValue[i - 1][j]);
			sub[i << 1][(j << 1) + 1] = abs(intValue[i][j] - intValue[i][j - 1]);
			sub[i << 1][j << 1] = (abs(intValue[i][j] - intValue[i - 1][j - 1]));
		}
	}
	//Compute clarity.
	for (int i = 0; i < PATCH_HEIGHT; i++) {
		for (int j = 0; j < PATCH_WIDTH; j++) {
			if (((int)(*it++)[0] >= 20)) {
				res += (((sub[i << 1][(j << 1) + 1] +
					sub[(i << 1) + 2][(j << 1) + 1] +
					sub[(i << 1) + 1][j << 1] +
					sub[(i << 1) + 1][(j << 1) + 2]) << 1) +
					(sub[i << 1][j << 1] +
						sub[(i << 1) + 2][j << 1] +
						sub[i << 1][(j << 1) + 2] +
						sub[(i << 1) + 2][(j << 1) + 2]));
				counter++;
			}
		}
	}
	float  ratio = (double)res / counter;
	return ratio;
}

int  getYvalue(cv::Mat& img, float range, float srcVal)
{
	float bin = (img.rows / 2) / range;
	int dstVal = (range - srcVal) *  bin;
	return dstVal;
}
void  drawAngle(cv::Mat& img, float range, std::vector<float>& angles, cv::Scalar& color)
{
	cv::Point p1, p2;
	p1 = cv::Point(0, getYvalue(img, 45.0, angles[0]));
	for (int i = 0; i < angles.size(); i++)
	{
		p2 = cv::Point(i, getYvalue(img, 45.0, angles[i]));
		cv::line(img, p1, p2, color);
		p1 = p2;
	}
}
void  drawEvas(std::vector<float>& pitchs, std::vector<float>& yaws, std::vector<float>& rolls)
{
	cv::Mat  img = cv::Mat::zeros(cv::Size(640, 360), CV_8UC3);
	cv::Point p1, p2;

	//画坐标系
	p1 = cv::Point(0, img.rows/2);
	p2 = cv::Point(img.cols, img.rows/2);
	cv::line(img, p1, p2, cv::Scalar(120, 120, 120));

	for (int i = -45; i <= 45; i++)
	{
		int y = getYvalue(img, 45.0, i);
		p1 = cv::Point(0, y);
		int x = 10;
		if (i % 5 == 0)
		{
			x = img.cols;
		}
		p2 = cv::Point(x, y);
		cv::line(img, p1, p2, cv::Scalar(120, 120, 120));
	}

	//pitch
	drawAngle(img, 45.0, pitchs, cv::Scalar(255, 0, 0));
	drawAngle(img, 45.0, yaws, cv::Scalar(0, 255, 0));
	drawAngle(img, 45.0, rolls, cv::Scalar(0, 0, 255));

	cv::imshow("pose", img);
	//cv::waitKey(1);
}