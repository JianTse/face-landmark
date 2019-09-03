#ifndef __POSE_EST_H__
#define __POSE_EST_H__

/************************************************************************
*	Í·ÎÄ¼þ
************************************************************************/
#include "opencv/cv.h"
static const int samplePdim = 7;

static int estimatePosePtIndexs[] = { 36, 39, 42, 45, 30, 48, 54 };
//static int estimateHeadPosePointIndexs[] = { 19, 22, 25, 28, 15, 31, 33 };
static float estimatPose2dArray[] = {
	-0.208764f, -0.140359f, 0.458815f, 0.106082f, 0.00859783f, -0.0866249f, -0.443304f, -0.00551231f, -0.0697294f,
	-0.157724f, -0.173532f, 0.16253f, 0.0935172f, -0.0280447f, 0.016427f, -0.162489f, -0.0468956f, -0.102772f,
	0.126487f, -0.164141f, 0.184245f, 0.101047f, 0.0104349f, -0.0243688f, -0.183127f, 0.0267416f, 0.117526f,
	0.201744f, -0.051405f, 0.498323f, 0.0341851f, -0.0126043f, 0.0578142f, -0.490372f, 0.0244975f, 0.0670094f,
	0.0244522f, -0.211899f, -1.73645f, 0.0873952f, 0.00189387f, 0.0850161f, 1.72599f, 0.00521321f, 0.0315345f,
	-0.122839f, 0.405878f, 0.28964f, -0.23045f, 0.0212364f, -0.0533548f, -0.290354f, 0.0718529f, -0.176586f,
	0.136662f, 0.335455f, 0.142905f, -0.191773f, -0.00149495f, 0.00509046f, -0.156346f, -0.0759126f, 0.133053f,
	-0.0393198f, 0.307292f, 0.185202f, -0.446933f, -0.0789959f, 0.29604f, -0.190589f, -0.407886f, 0.0269739f,
	-0.00319206f, 0.141906f, 0.143748f, -0.194121f, -0.0809829f, 0.0443648f, -0.157001f, -0.0928255f, 0.0334674f,
	-0.0155408f, -0.145267f, -0.146458f, 0.205672f, -0.111508f, 0.0481617f, 0.142516f, -0.0820573f, 0.0329081f,
	-0.0520549f, -0.329935f, -0.231104f, 0.451872f, -0.140248f, 0.294419f, 0.223746f, -0.381816f, 0.0223632f,
	0.176198f, -0.00558382f, 0.0509544f, 0.0258391f, 0.050704f, -1.10825f, -0.0198969f, 1.1124f, 0.189531f,
	-0.0352285f, 0.163014f, 0.0842186f, -0.24742f, 0.199899f, 0.228204f, -0.0721214f, -0.0561584f, -0.157876f,
	-0.0308544f, -0.131422f, -0.0865534f, 0.205083f, 0.161144f, 0.197055f, 0.0733392f, -0.0916629f, -0.147355f,
	0.527424f, -0.0592165f, 0.0150818f, 0.0603236f, 0.640014f, -0.0714241f, -0.0199933f, -0.261328f, 0.891053f };


static float estimatePose2dArray_direct[] = {
	0.139791f, 27.4028f, 7.02636f,
	-2.48207f, 9.59384f, 6.03758f,
	1.27402f, 10.4795f, 6.20801f,
	1.17406f, 29.1886f, 1.67768f,
	0.306761f, -103.832f, 5.66238f,
	4.78663f, 17.8726f, -15.3623f,
	-5.20016f, 9.29488f, -11.2495f,
	-25.1704f, 10.8649f, -29.4877f,
	-5.62572f, 9.0871f, -12.0982f,
	-5.19707f, -8.25251f, 13.3965f,
	-23.6643f, -13.1348f, 29.4322f,
	67.239f, 0.666896f, 1.84304f,
	-2.83223f, 4.56333f, -15.885f,
	-4.74948f, -3.79454f, 12.7986f,
	-16.1f, 1.47175f, 4.03941f };

static cv::Mat ldmark2mat(const std::vector<cv::Point>& ldmarks)
{
	if (ldmarks.size() < 1)
		return cv::Mat();

	int miny = 10000;
	int maxy = 0;
	int sumx = 0;
	int sumy = 0;
	for (int i = 0; i < samplePdim; i++) {
		sumx += ldmarks[estimatePosePtIndexs[i]].x;
		int y = ldmarks[estimatePosePtIndexs[i]].y;
		sumy += y;
		if (miny > y)
			miny = y;
		if (maxy < y)
			maxy = y;
	}

	float dist = static_cast<float>(maxy - miny);
	float sx = (sumx * 1.0f) / samplePdim;
	float sy = (sumy * 1.0f) / samplePdim;

	cv::Mat tmp(1, 2 * samplePdim + 1, CV_32FC1);
	for (int i = 0; i < samplePdim; i++) {
		tmp.at<float>(i) = (ldmarks[estimatePosePtIndexs[i]].x - sx) / dist;
		tmp.at<float>(i + samplePdim) = (ldmarks[estimatePosePtIndexs[i]].y - sy) / dist;
	}
	tmp.at<float>(2 * samplePdim) = 1.0f;
	return tmp;
};

static bool estimateEav(const std::vector<cv::Point>& ldmarks, std::vector<float>& eavs)
{
	cv::Mat tmp = ldmark2mat(ldmarks);
	if (tmp.empty())
	{
		return false;
	}
	cv::Mat headPoseMat_direct = cv::Mat(15, 3, CV_32FC1, estimatePose2dArray_direct);
	cv::Mat predict = tmp * headPoseMat_direct;
	eavs.clear();
	for (int i = 0; i < 3; i++)
	{
		eavs.push_back(predict.at<float>(i));
	}
	return true;
};

#endif /* __ALIGMENT_H__ */