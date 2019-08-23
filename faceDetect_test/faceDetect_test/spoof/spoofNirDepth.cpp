//
// Created by Longqi on 2017/11/18..
//

/*
 * TO DO : change the P-net and update the generat box
 */

#include "spoofNirDepth.h"

SpoofNirDepth::SpoofNirDepth()
{
}

SpoofNirDepth::~SpoofNirDepth(){
}

/*
函数判断是否为活体
将人脸3维数据在x,y平面内投影，只要不是在x,y平面内投影为一条直线即可用任意3个点拟合出一个平面
1：如果在人脸内的其他点有很多都在这个平面内，说明不是活体，是一个假人脸
2：如果在人脸内的其他点有不是很多点在这个平面内，说明是真人脸
*/
int SpoofNirDepth::check_triangle(std::vector<cv::Vec3i>& faceData)
{
	int faceno0_num = faceData.size();
	int x[3], y[3], z[3];  // 随机取三个点 
	float a, b, c;  // 拟合平面方程 z=ax+by+c
	int pt_idx[3];
	float check, distance;

	//随机取3个点
	do {
		pt_idx[0] = rand() % faceno0_num;
		pt_idx[1] = rand() % faceno0_num;
		pt_idx[2] = rand() % faceno0_num;
	} while (pt_idx[0] == pt_idx[1] || pt_idx[0] == pt_idx[2] || pt_idx[1] == pt_idx[2]);

	for (int n = 0; n < 3; n++)
	{
		int  idx = pt_idx[n];
		x[n] = faceData[idx][0];
		y[n] = faceData[idx][1];
		z[n] = faceData[idx][2];
	}
	check = (x[0] - x[1])*(y[0] - y[2]) - (x[0] - x[2])*(y[0] - y[1]);
	if (fabs(check) < 0.01)  // 防止提示浮点数例外 (核心已转储)
	{
		return -1;
	}
	a = ((z[0] - z[1])*(y[0] - y[2]) - (z[0] - z[2])*(y[0] - y[1])) / ((x[0] - x[1])*(y[0] - y[2]) - (x[0] - x[2])*(y[0] - y[1]));
	check = y[0] - y[2];
	if (fabs(check) < 0.01)  // 防止提示浮点数例外 (核心已转储)
	{
		return -1;
	}
	b = ((z[0] - z[2]) - a * (x[0] - x[2])) / (y[0] - y[2]);
	c = z[0] - a * x[0] - b * y[0];
	int total = 0;
	for (int n = 0; n < faceno0_num + 1; n++)
	{
		distance = fabs(a*faceData[n][0] + b*faceData[n][1] - 1 * faceData[n][2] + c * 1);
		if (distance < 1)
		{
			total += 1;
		}
	}
	return total;
}

int SpoofNirDepth::depth_liveness_detection(cv::Mat& bgrImg, cv::Mat& depthImg, cv::Rect& faceRect, std::vector<cv::Point>& ldmark5)
{
	if (depthImg.empty())
		return  -1;

	cv::Rect box = faceRect;
	cv::Size maxSize = cv::Size(depthImg.cols, depthImg.rows);
	box = getBoundingBox(maxSize, ldmark5);

	int  x1 = box.x;
	int  x2 = box.x + box.width;
	int  y1 = MAX(0, box.y);
	int  y2 = MIN(depthImg.rows - 1, y1 + box.height);
	if (bgrImg.cols == 1280 && bgrImg.rows == 720)
	{
		float  scale = 2;
		int offset_y = 64;
		x1 = int(float(box.x) / scale);
		x2 = int(x1 + float(box.width) / scale);
		y1 = MAX(0, int(offset_y + float(box.y) / scale));
		y2 = MIN(depthImg.rows - 1, int(y1 + float(box.height) / scale));
	}
	//LOGI("bgr size: %d, %d,  x1: %d, x2: %d, y1: %d, y2: %d", bgrImg.cols, bgrImg.rows, x1,x2, y1, y2);

	std::vector<cv::Vec3i> faceData;
	for (int y = y1; y < y2; y++)
	{
		short* ptr = depthImg.ptr<short>(y);
		for (int x = x1; x < x2; x++)
		{
			if (ptr[x] == 0) continue;
			cv::Vec3i  data;
			data[0] = x;
			data[1] = y;
			data[2] = ptr[x];
			faceData.push_back(data);
		}
	}
	if (faceData.size() < 50)
		return -1;

	int pretotal = 0;  // 统计最多点都在一个平面的点的个数
	for (int i = 0; i < 500; i++)
	{
		int ret = check_triangle(faceData);
		if (ret < 0)
		{
			if (i > 0)
			{
				i -= 1;
			}
			continue;
		}
		if (ret > pretotal)  // 找到符合拟合平面数据最多的拟合平面
		{
			pretotal = ret;
		}
	}
	float pretotal_ary = pretotal *1.0 / faceData.size();
	//LOGI("live-face:  %d-%d----%f\n", pretotal, faceData.size(), pretotal_ary);

	int isLive = 1;
	if (pretotal_ary > 0.15f)  //如果在同一个平面的点数超过了15%，假人脸
	{
		isLive = 0;
	}
	return isLive;
}

cv::Rect SpoofNirDepth::getFaceSanp(cv::Mat& img, cv::Rect& faceRect, cv::Mat& imgPatch)
{
	int extend = min(faceRect.width, faceRect.height);
	int extendSize = extend / 2;
	extendSize = (extendSize / 4) * 4;
	cv::Rect roi = cvExtendRect(faceRect, extendSize, extendSize);
	cvValidateRect(cv::Size(img.cols, img.rows), roi);
	imgPatch = img(roi).clone();
	return roi;
}

int SpoofNirDepth::nir_liveness_detection(MTCNN_NCNN&  detector, cv::Mat& bgrImg, cv::Mat& nirImg, cv::Rect& faceRect)
{
	if (nirImg.empty())
		return  -1;

	//取出faceRect在nirImg中对应位置的图像块
	cv::Mat imgPatch;
	getFaceSanp(nirImg, faceRect, imgPatch);
	
	//在imgPatch中检测人脸，如果能检测到人脸，则live
	int minFace = min(faceRect.width, faceRect.height)  * 0.75;

	std::vector<Bbox> finalBbox;
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels_resize(imgPatch.data, ncnn::Mat::PIXEL_BGR2RGB, imgPatch.cols, imgPatch.rows, imgPatch.cols, imgPatch.rows);

	std::vector<float>threshs(3);
	threshs[0] = 0.3;
	threshs[1] = 0.4;
	threshs[2] = 0.5;
	detector.setMinFace(minFace);
	detector.setThreshold(threshs);
	//detector.detect(ncnn_img, finalBbox);
	detector.detectMaxFace(ncnn_img, finalBbox);
	const int num_box = finalBbox.size();

#if  1
	//显示nir检测结果
	for (int i = 0; i < num_box; i++) {
		Bbox bbox = finalBbox[i];
		cv::Point pt1(bbox.x1, bbox.y1);
		cv::Point pt2(bbox.x2, bbox.y2);
		cv::Rect rect = valid_rect(pt1, pt2, imgPatch.cols, imgPatch.rows);
		if (rect.area() < 1)
		{
			continue;
		}
		float score = bbox.score;
		cv::rectangle(imgPatch, rect, cv::Scalar(0, 0, 0));
		char info[255];
		sprintf(info, "%f ", score);
		cv::putText(imgPatch, info, cv::Point(60, 80), CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255));
	}
	cv::imshow("patch", imgPatch);
	cv::waitKey(1);
#endif
	int ret = 0;
	if (num_box > 0)
	{
		ret = 1;
	}
	return  ret;
}

