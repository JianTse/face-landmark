#ifndef __POSE_EST_H__
#define __POSE_EST_H__

/************************************************************************
*	头文件
************************************************************************/
#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
using namespace std;

#define EPSINON  1e-6

//结构体定义
struct PoseInfo
{
	int pose;
	int face_width;
	bool is_pitch_up;
	bool is_pitch_down;
	bool is_yaw_left;
	bool is_yaw_right;
	bool is_roll_left;
	bool is_roll_righ;
	bool is_front;
	float angle;
	float r;

	std::vector<float> eav;
	int depthInfo;
	float front_prob;
	PoseInfo()
	{
		pose = -1;
		face_width = 0;
		is_pitch_up = false;
		is_pitch_down = false;
		is_yaw_left = false;
		is_yaw_right = false;
		is_roll_left = false;
		is_roll_righ = false;
		is_front = false;
		eav.clear();
		depthInfo = -1;
		front_prob = 0.0f;
		angle = -1.0f;
		r = 0.0f;
	}
};

typedef enum eFacePose
{
	no_face = -1,
	pitch_up = 0,		// Picth：0--抬头
	pitch_down,		    // Picth：1--低头
	yaw_left,			// Yaw : 2--左侧脸
	yaw_right,			// Yaw : 3--右侧脸
	roll_left,			// Roll : 4--左歪头
	roll_right,			// Roll : 5--右歪头
	front,            // 6---正脸
	pose_register_over = 399,
	pose_error = 400
}eFacePose;

class CPoseEst
{
public:
	CPoseEst();
	~CPoseEst();

	int  facePoseEstimate(std::vector<cv::Point>& ldmark_pts, PoseInfo& poseInfo);
	bool estimateEav(const std::vector<cv::Point>& ldmarks, const int faceWidth, std::vector<float>& eavs);
private:
	void judgeSiglePose(int eav0, int eav1, int eav2, PoseInfo& faceInfo);
	int judgePose(const std::vector<float>& eav, PoseInfo& faceInfo);
	cv::Mat ldmark2mat(const std::vector<cv::Point>& ldmarks);
	
};

#endif /* __ALIGMENT_H__ */