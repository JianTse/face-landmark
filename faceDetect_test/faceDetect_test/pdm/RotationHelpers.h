///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt
//
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////

#ifndef __ROTATION_HELPERS_h_
#define __ROTATION_HELPERS_h_

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

namespace Utilities
{
#define CVV_PI   3.1415926535897932384626433832795
	//class RotationHelper 
	//{
	//public:
	//	RotationHelper();
	//	void cmpRotationHypotheye(const cv::Mat& img, const cv::Rect& face,
	//		const std::vector<cv::Point>& landmark5, cv::Vec3d& rotation_hypothese);

	//private:
	//	std::vector<cv::Vec3d> rotation_hypotheses_inits;
	//	std::vector<shapeCaches>  m_shapeCaches;

	//};
	//===========================================================================
	// Angle representation conversion helpers
	//===========================================================================

	// Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
	static cv::Matx33f Euler2RotationMatrix(const cv::Vec3f& eulerAngles)
	{
		cv::Matx33f rotation_matrix;

		float s1 = sin(eulerAngles[0]);
		float s2 = sin(eulerAngles[1]);
		float s3 = sin(eulerAngles[2]);

		float c1 = cos(eulerAngles[0]);
		float c2 = cos(eulerAngles[1]);
		float c3 = cos(eulerAngles[2]);

		rotation_matrix(0, 0) = c2 * c3;
		rotation_matrix(0, 1) = -c2 *s3;
		rotation_matrix(0, 2) = s2;
		rotation_matrix(1, 0) = c1 * s3 + c3 * s1 * s2;
		rotation_matrix(1, 1) = c1 * c3 - s1 * s2 * s3;
		rotation_matrix(1, 2) = -c2 * s1;
		rotation_matrix(2, 0) = s1 * s3 - c1 * c3 * s2;
		rotation_matrix(2, 1) = c3 * s1 + c1 * s2 * s3;
		rotation_matrix(2, 2) = c1 * c2;

		return rotation_matrix;
	}

	// Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
	static cv::Vec3f RotationMatrix2Euler(const cv::Matx33f& rotation_matrix)
	{
		float q0 = sqrt(1 + rotation_matrix(0, 0) + rotation_matrix(1, 1) + rotation_matrix(2, 2)) / 2.0f;
		float q1 = (rotation_matrix(2, 1) - rotation_matrix(1, 2)) / (4.0f*q0);
		float q2 = (rotation_matrix(0, 2) - rotation_matrix(2, 0)) / (4.0f*q0);
		float q3 = (rotation_matrix(1, 0) - rotation_matrix(0, 1)) / (4.0f*q0);

		// Slower, but dealing with degenerate cases due to precision
		float t1 = 2.0f * (q0*q2 + q1*q3);
		if (t1 > 1) t1 = 1.0f;
		if (t1 < -1) t1 = -1.0f;

		float yaw = asin(t1);
		float pitch = atan2(2.0f * (q0*q1 - q2*q3), q0*q0 - q1*q1 - q2*q2 + q3*q3);
		float roll = atan2(2.0f * (q0*q3 - q1*q2), q0*q0 + q1*q1 - q2*q2 - q3*q3);

		return cv::Vec3f(pitch, yaw, roll);
	}

	static cv::Vec3f Euler2AxisAngle(const cv::Vec3f& euler)
	{
		cv::Matx33f rotMatrix = Euler2RotationMatrix(euler);
		cv::Vec3f axis_angle;
		cv::Rodrigues(rotMatrix, axis_angle);
		return axis_angle;
	}

	static cv::Vec3f AxisAngle2Euler(const cv::Vec3f& axis_angle)
	{
		cv::Matx33f rotation_matrix;
		cv::Rodrigues(axis_angle, rotation_matrix);
		return RotationMatrix2Euler(rotation_matrix);
	}

	static cv::Matx33f AxisAngle2RotationMatrix(const cv::Vec3f& axis_angle)
	{
		cv::Matx33f rotation_matrix;
		cv::Rodrigues(axis_angle, rotation_matrix);
		return rotation_matrix;
	}

	static cv::Vec3f RotationMatrix2AxisAngle(const cv::Matx33f& rotation_matrix)
	{
		cv::Vec3f axis_angle;
		cv::Rodrigues(rotation_matrix, axis_angle);
		return axis_angle;
	}

	// Generally useful 3D functions
	static void Project(cv::Mat_<float>& dest, const cv::Mat_<float>& mesh, float fx, float fy, float cx, float cy)
	{
		dest = cv::Mat_<float>(mesh.rows, 2, 0.0);

		int num_points = mesh.rows;

		float X, Y, Z;


		cv::Mat_<float>::const_iterator mData = mesh.begin();
		cv::Mat_<float>::iterator projected = dest.begin();

		for (int i = 0; i < num_points; i++)
		{
			// Get the points
			X = *(mData++);
			Y = *(mData++);
			Z = *(mData++);

			float x;
			float y;

			// if depth is 0 the projection is different
			if (Z != 0)
			{
				x = ((X * fx / Z) + cx);
				y = ((Y * fy / Z) + cy);
			}
			else
			{
				x = X;
				y = Y;
			}

			// Project and store in dest matrix
			(*projected++) = x;
			(*projected++) = y;
		}

	}

	static void  refineMtcnnBox(cv::Rect& box)
	{
		box.x = box.width * -0.0075f + box.x;
		box.y = box.height * 0.2459f + box.y;
		box.width = 1.0323f * box.width;
		box.height = 0.7751f * box.height;
	}

	static double get_points_dis(std::vector<cv::Point>& landmark1,
		std::vector<cv::Point>& landmark2)
	{
		double  total_dist = 0;
		for (int i = 0; i < landmark1.size(); i++)
		{
			total_dist += sqrtf((landmark1[i].x - landmark2[i].x) * (landmark1[i].x - landmark2[i].x)
				+ (landmark1[i].y - landmark2[i].y) * (landmark1[i].y - landmark2[i].y));
		}
		return total_dist;
	}

	static void  generateEuler(vector<cv::Vec3d>& rotation_hypothese)
	{
		rotation_hypothese.clear();
		float  pitch_degree = 0;
		float  yaw_degree = 0;
		float  roll_degree = 0;
		for (int pitch = -30; pitch <= 30; pitch += 5)
		{
			pitch_degree = pitch * CVV_PI / 180.0;
			for (int yaw = -90; yaw <= 90; yaw += 5)
			{
				yaw_degree = yaw * CVV_PI / 180.0;
				for (int roll = -50; roll <= 50; roll += 5)
				{
					roll_degree = roll * CVV_PI / 180.0;
					rotation_hypothese.push_back(cv::Vec3d(pitch_degree, yaw_degree, roll_degree));
				}
			}
		}
	}

	static float get_eye_dist(const vector<cv::Point>& landmark)
	{ 
		float dist = 1.0f;
		if (!(landmark.size() == 68 || landmark.size() == 5))
		{
			return dist;
		}
		cv::Point  left_eye = cv::Point(0, 0);
		cv::Point  right_eye = cv::Point(0, 0);
		if (landmark.size() == 68)
		{
			//×óÑÛ
			for (int i = 36; i < 42; i++)
			{
				left_eye.x += landmark[i].x;
				left_eye.y += landmark[i].y;
			}
			left_eye.x /= 6;
			left_eye.y /= 6;

			//ÓÒÑÛ
			for (int i = 42; i < 48; i++)
			{
				right_eye.x += landmark[i].x;
				right_eye.y += landmark[i].y;
			}
			right_eye.x /= 6;
			right_eye.y /= 6;
		}
		else if (landmark.size() == 5)
		{
			left_eye = landmark[0];
			right_eye = landmark[1];
		}
		dist = sqrtf((left_eye.x - right_eye.x)*(left_eye.x - right_eye.x)
			+ (left_eye.y - right_eye.y)*(left_eye.y - right_eye.y));
		return dist;
	}


	static void conv_landmark68_2_landmark5(vector<cv::Point>& landmark5, vector<cv::Point>& landmark68)
	{
		landmark5.clear();
		//×óÑÛ
		cv::Point  left_eye68 = cv::Point(0, 0);
		for (int i = 36; i < 42; i++)
		{
			left_eye68.x += landmark68[i].x;
			left_eye68.y += landmark68[i].y;
		}
		left_eye68.x /= 6;
		left_eye68.y /= 6;
		landmark5.push_back(left_eye68);

		//ÓÒÑÛ
		cv::Point  right_eye68 = cv::Point(0, 0);
		for (int i = 42; i < 48; i++)
		{
			right_eye68.x += landmark68[i].x;
			right_eye68.y += landmark68[i].y;
		}
		right_eye68.x /= 6;
		right_eye68.y /= 6;
		landmark5.push_back(right_eye68);

		landmark5.push_back(landmark68[30]);  //nose
		landmark5.push_back(landmark68[48]);  //mouth_left
		landmark5.push_back(landmark68[54]);  //mouth_right
	}

}
#endif