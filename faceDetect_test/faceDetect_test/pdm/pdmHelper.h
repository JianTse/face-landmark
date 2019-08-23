#ifndef _PDM_HELPER_H_
#define _PDM_HELPER_H_
#include "opencv2/core/core.hpp"
//#include "common/androidlog.h"
//#include "common/ApiCommon.h"

#include "PDM.h"

#define ret_nuiFace_ok 1
#define ret_nuiFace_failed 0
#define ret_nuiFace_invalid_param 0
#define ret_nuiFace_not_init 0

//遍历3d角度与2d点的关系
struct  shapeCaches
{
	cv::Vec3d rotation_hypothese;
	std::vector<cv::Point> landmark68;//68pts
	std::vector<cv::Point> landmark5;//68pts
};

class PDM_Helper
{
public:
	PDM_Helper();

	int init(const char* modelDir);

	void destroy();

	int estInitShape(const cv::Rect faceRect, const cv::Vec3d rotation_hypothes, vector<cv::Point>& outShape);

	int estInitShapeWithMtcnn(const cv::Rect mtcnn_faceBox, const vector<cv::Point> mtcnn_ldmark,
		vector<cv::Point>& outShape);

	// Provided the landmark location compute global and local parameters best fitting it (can provide optional rotation for potentially better results)
	int CalcParams(cv::Vec6f& out_params_global, cv::Mat_<float>& out_params_local,
		const vector<cv::Point>& ldmark, const cv::Vec3f rotation = cv::Vec3f(0.0f), bool only_global = false);

	// Compute shape in image space (2D)
	int CalcShape2D(vector<cv::Point>& ldmark, const cv::Mat_<float>& params_local,
		const cv::Vec6f& params_global) const;

	int get_rotation_hypothese(const cv::Rect& face,
		const std::vector<cv::Point>& landmark5, cv::Vec3d& rotation_hypothese);

private:
	int PDM_Helper::initShapeCashes();

	vector<cv::Vec3d> rotation_hypotheses_inits;
	std::vector<shapeCaches>  m_shapeCaches;
	bool m_isInited;
	int m_width;
	int m_height;
	LandmarkDetector::PDM pdm;
};

#endif