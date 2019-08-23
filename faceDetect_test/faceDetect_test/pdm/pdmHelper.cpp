#include "pdmHelper.h"
#include "RotationHelpers.h"
using namespace cv;

#define  FACE_RECT_X   100
#define  FACE_RECT_Y  100
#define  FACE_RECT_W   291
#define  FACE_RECT_H  301
#define  FACE_EYE_DIST  140

PDM_Helper::PDM_Helper()
{
	m_isInited = false;
}

int PDM_Helper::init(const char* modelDir)
{
	if (m_isInited)
	{
		return ret_nuiFace_ok;
	}

	m_isInited = false;
	string location = modelDir;
	location= location + "/In-the-wild_aligned_PDM_68.txt";
	bool read_success = pdm.Read(location);
	if (!read_success)
	{
		return ret_nuiFace_failed;
	}
	initShapeCashes();
	m_isInited = true;
	return ret_nuiFace_ok;
}

//保证两只眼睛之间的距离固定
void  refineModelLdmark(vector<cv::Point>& landmark)
{
	int  m_cx = FACE_RECT_X + FACE_RECT_W / 2;
	int  m_cy = FACE_RECT_Y + FACE_RECT_H / 2;

	float  dist = Utilities::get_eye_dist(landmark);
	float  scale = dist / FACE_EYE_DIST;
	for (int i = 0; i < landmark.size(); i++)
	{
		int off_x = landmark[i].x - m_cx;
		int off_y = landmark[i].y - m_cy;
		landmark[i].x = m_cx + off_x / scale;
		landmark[i].y = m_cy + off_y / scale;
	}
}

int PDM_Helper::initShapeCashes()
{
	if (rotation_hypotheses_inits.size() < 1)
	{
		Utilities::generateEuler(rotation_hypotheses_inits);
	}

	cv::Rect face = cv::Rect(FACE_RECT_X, FACE_RECT_Y, FACE_RECT_W, FACE_RECT_H);
	m_shapeCaches.clear();
	for (int i = 0; i < rotation_hypotheses_inits.size(); i++)
	{
		//计算所有形状
		shapeCaches  shape;
		shape.rotation_hypothese = rotation_hypotheses_inits[i];
		estInitShape(face, shape.rotation_hypothese, shape.landmark68);
		refineModelLdmark(shape.landmark68);
		Utilities::conv_landmark68_2_landmark5(shape.landmark5, shape.landmark68);
		m_shapeCaches.push_back(shape);
	}
	return ret_nuiFace_ok;
}

void PDM_Helper::destroy()
{
	if (!m_isInited)
	{
		return;
	}
}

int PDM_Helper::estInitShape(const cv::Rect faceRect, const cv::Vec3d rotation_hypothes,
	vector<cv::Point>& outShape)
{
	if (faceRect.area() < 1)
	{
		return ret_nuiFace_invalid_param;
	}

	cv::Mat_<float>    params_local;
	params_local.create(pdm.NumberOfModes(), 1);
	params_local.setTo(0.0);
	// global parameters (pose) [scale, euler_x, euler_y, euler_z, tx, ty]
	cv::Vec6f    params_global = cv::Vec6f(1, 0, 0, 0, 0, 0);
	pdm.CalcParams(params_global, faceRect, params_local, rotation_hypothes);

	// Placeholder for the landmarks
	CalcShape2D(outShape, params_local,params_global);
	return ret_nuiFace_ok;
}


//获取当前人脸的ldmark的变化量
void  getCurLdmarkCof(const cv::Rect& face, std::vector<cv::Point>& landmark,
	float& offset_x, float& offset_y, float& scale)
{
	int  m_cx = FACE_RECT_X + FACE_RECT_W / 2;
	int  m_cy = FACE_RECT_Y + FACE_RECT_H / 2;
	int  c_cx = face.x + face.width / 2;
	int  c_cy = face.y + face.height / 2;
	offset_x = c_cx - m_cx;
	offset_y = c_cy - m_cy;
	float  dist = Utilities::get_eye_dist(landmark);
	scale = dist / FACE_EYE_DIST;
}

//矫正当前人脸的ldmark位置与model对齐
void refineLdmark(const cv::Rect& face, std::vector<cv::Point>& landmark)
{
	//获取当前人脸的ldmark的变化量
	float offset_x, offset_y, scale;
	getCurLdmarkCof(face, landmark, offset_x, offset_y, scale);

	//矫正当前人脸的ldmark位置与model对齐
	int  m_cx = FACE_RECT_X + FACE_RECT_W / 2;
	int  m_cy = FACE_RECT_Y + FACE_RECT_H / 2;
	int  c_cx = face.x + face.width / 2;
	int  c_cy = face.y + face.height / 2;
	for (int i = 0; i < landmark.size(); i++)
	{
		landmark[i].x = m_cx + (landmark[i].x - c_cx) / scale;
		landmark[i].y = m_cy + (landmark[i].y - c_cy) / scale;
	}
}

int PDM_Helper::get_rotation_hypothese(const cv::Rect& face,
	const std::vector<cv::Point>& landmark, cv::Vec3d& rotation_hypothese)
{
	if (face.area() < 1 || landmark.size()<1)
	{
		return ret_nuiFace_invalid_param;
	}
	//矫正当前人脸的ldmark位置与model对齐
	std::vector<cv::Point> refine_landmark(landmark);
	refineLdmark(face, refine_landmark);

	//模型库中匹配一个距离最近的模型作为初始shape
	std::vector<float> dist_caches;
	int id = 0;
	float  min_dis = 100000;
	for (int i = 0; i < m_shapeCaches.size(); i++)
	{
		double dis;
		if (refine_landmark.size() == 68)
		{
			dis = Utilities::get_points_dis(refine_landmark, m_shapeCaches[i].landmark68);
		}
		else {
			dis = Utilities::get_points_dis(refine_landmark, m_shapeCaches[i].landmark5);
		}
		dist_caches.push_back(dis);
		if (dis < min_dis)
		{
			min_dis = dis;
			id = i;
		}
	}
	rotation_hypothese = m_shapeCaches[id].rotation_hypothese;
	return ret_nuiFace_ok;
}

int PDM_Helper::CalcParams(cv::Vec6f& out_params_global, cv::Mat_<float>& out_params_local,
	const vector<cv::Point>& ldmark_pts, const cv::Vec3f rotation, bool only_global)
{
	if (ldmark_pts.size() < 1)
	{
		return ret_nuiFace_invalid_param;
	}

	int n = ldmark_pts.size();
	cv::Mat_<float> current_shape(2 * n, 1, 0.0f);
	for (size_t i = 0; i < n; i++)
	{
		current_shape.at<float>(i) = static_cast<float>(ldmark_pts[i].x);
		current_shape.at<float>(i + n) = static_cast<float>(ldmark_pts[i].y);
	}
	if (only_global)
	{
		pdm.CalcGlobalParams(out_params_global, out_params_local, current_shape, rotation);
	}
	else {
		pdm.CalcParams(out_params_global, out_params_local, current_shape, rotation);
	}
	return ret_nuiFace_ok;
}

// Compute shape in image space (2D)
int PDM_Helper::CalcShape2D(vector<cv::Point>& outShape, const cv::Mat_<float>& params_local,
	const cv::Vec6f& params_global) const
{
	// Placeholder for the landmarks
	cv::Mat_<float> current_shape(2 * pdm.NumberOfPoints(), 1, 0.0f);
	pdm.CalcShape2D(current_shape, params_local, params_global);

	outShape.clear();
	int n = pdm.NumberOfPoints();
	for (int l = 0; l < n; l++) {
		cv::Point featurePoint(cvRound(current_shape.at<float>(l)), cvRound(current_shape.at<float>(l + n)));
		outShape.push_back(featurePoint);
	}
	return ret_nuiFace_ok;
}

//input is mtcnn facebox
int PDM_Helper::estInitShapeWithMtcnn(const cv::Rect mtcnn_faceBox, const vector<cv::Point> mtcnn_ldmark,
	vector<cv::Point>& outShape)
{
	if (!m_isInited)
	{
		return ret_nuiFace_not_init;
	}

	if (mtcnn_faceBox.area() < 1 || mtcnn_ldmark.size() != 5)
	{
		return ret_nuiFace_invalid_param;
	}

	//get rotation_hypothese
	cv::Vec3d rotation_hypothese;
	cv::Rect  faceBox = mtcnn_faceBox;
	Utilities::refineMtcnnBox(faceBox);
	int ret = get_rotation_hypothese(faceBox,mtcnn_ldmark,rotation_hypothese);
	if (ret != ret_nuiFace_ok)
	{
		return ret;
	}
	estInitShape(faceBox, rotation_hypothese,outShape);
	printf("detect: pitch:%.1f,yaw:%.1f,roll:%.1f\n", rotation_hypothese[0] * 180 / CV_PI,
		rotation_hypothese[1] * 180 / CV_PI, rotation_hypothese[2] * 180 / CV_PI);
	return ret_nuiFace_ok;
}
