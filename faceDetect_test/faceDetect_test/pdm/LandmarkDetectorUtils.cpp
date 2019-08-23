///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
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


#include "LandmarkDetectorUtils.h"
#include "RotationHelpers.h"

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;

namespace LandmarkDetector
{

	//===========================================================================
	// Fast patch expert response computation (linear model across a ROI) using normalised cross-correlation
	//===========================================================================

	void crossCorr_m(const cv::Mat_<float>& img, cv::Mat_<double>& img_dft, const cv::Mat_<float>& _templ, map<int, cv::Mat_<double> >& _templ_dfts, cv::Mat_<float>& corr)
	{
		// Our model will always be under min block size so can ignore this
		//const double blockScale = 4.5;
		//const int minBlockSize = 256;

		int maxDepth = CV_64F;

		cv::Size dftsize;

		dftsize.width = cv::getOptimalDFTSize(corr.cols + _templ.cols - 1);
		dftsize.height = cv::getOptimalDFTSize(corr.rows + _templ.rows - 1);

		// Compute block size
		cv::Size blocksize;
		blocksize.width = dftsize.width - _templ.cols + 1;
		blocksize.width = MIN(blocksize.width, corr.cols);
		blocksize.height = dftsize.height - _templ.rows + 1;
		blocksize.height = MIN(blocksize.height, corr.rows);

		cv::Mat_<double> dftTempl;

		// if this has not been precomputed, precompute it, otherwise use it
		if (_templ_dfts.find(dftsize.width) == _templ_dfts.end())
		{
			dftTempl.create(dftsize.height, dftsize.width);

			cv::Mat_<float> src = _templ;

			cv::Mat_<double> dst(dftTempl, cv::Rect(0, 0, dftsize.width, dftsize.height));

			cv::Mat_<double> dst1(dftTempl, cv::Rect(0, 0, _templ.cols, _templ.rows));

			if (dst1.data != src.data)
				src.convertTo(dst1, dst1.depth());

			if (dst.cols > _templ.cols)
			{
				cv::Mat_<double> part(dst, cv::Range(0, _templ.rows), cv::Range(_templ.cols, dst.cols));
				part.setTo(0);
			}

			// Perform DFT of the template
			dft(dst, dst, 0, _templ.rows);

			_templ_dfts[dftsize.width] = dftTempl;

		}
		else
		{
			// use the precomputed version
			dftTempl = _templ_dfts.find(dftsize.width)->second;
		}

		cv::Size bsz(std::min(blocksize.width, corr.cols), std::min(blocksize.height, corr.rows));
		cv::Mat src;

		cv::Mat cdst(corr, cv::Rect(0, 0, bsz.width, bsz.height));

		cv::Mat_<double> dftImg;

		if (img_dft.empty())
		{
			dftImg.create(dftsize);
			dftImg.setTo(0.0);

			cv::Size dsz(bsz.width + _templ.cols - 1, bsz.height + _templ.rows - 1);

			int x2 = std::min(img.cols, dsz.width);
			int y2 = std::min(img.rows, dsz.height);

			cv::Mat src0(img, cv::Range(0, y2), cv::Range(0, x2));
			cv::Mat dst(dftImg, cv::Rect(0, 0, dsz.width, dsz.height));
			cv::Mat dst1(dftImg, cv::Rect(0, 0, x2, y2));

			src = src0;

			if (dst1.data != src.data)
				src.convertTo(dst1, dst1.depth());

			dft(dftImg, dftImg, 0, dsz.height);
			img_dft = dftImg.clone();
		}

		cv::Mat dftTempl1(dftTempl, cv::Rect(0, 0, dftsize.width, dftsize.height));
		cv::mulSpectrums(img_dft, dftTempl1, dftImg, 0, true);
		cv::dft(dftImg, dftImg, cv::DFT_INVERSE + cv::DFT_SCALE, bsz.height);

		src = dftImg(cv::Rect(0, 0, bsz.width, bsz.height));

		src.convertTo(cdst, CV_32F);

	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////

	void matchTemplate_m(const cv::Mat_<float>& input_img, cv::Mat_<double>& img_dft, cv::Mat& _integral_img, cv::Mat& _integral_img_sq, const cv::Mat_<float>&  templ, map<int, cv::Mat_<double> >& templ_dfts, cv::Mat_<float>& result, int method)
	{

		int numType = method == CV_TM_CCORR || method == CV_TM_CCORR_NORMED ? 0 :
			method == CV_TM_CCOEFF || method == CV_TM_CCOEFF_NORMED ? 1 : 2;
		bool isNormed = method == CV_TM_CCORR_NORMED ||
			method == CV_TM_SQDIFF_NORMED ||
			method == CV_TM_CCOEFF_NORMED;

		// Assume result is defined properly
		if (result.empty())
		{
			cv::Size corrSize(input_img.cols - templ.cols + 1, input_img.rows - templ.rows + 1);
			result.create(corrSize);
		}
		LandmarkDetector::crossCorr_m(input_img, img_dft, templ, templ_dfts, result);

		if (method == CV_TM_CCORR)
			return;

		double invArea = 1. / ((double)templ.rows * templ.cols);

		cv::Mat sum, sqsum;
		cv::Scalar templMean, templSdv;
		double *q0 = 0, *q1 = 0, *q2 = 0, *q3 = 0;
		double templNorm = 0, templSum2 = 0;

		if (method == CV_TM_CCOEFF)
		{
			// If it has not been precomputed compute it now
			if (_integral_img.empty())
			{
				integral(input_img, _integral_img, CV_64F);
			}
			sum = _integral_img;

			templMean = cv::mean(templ);
		}
		else
		{
			// If it has not been precomputed compute it now
			if (_integral_img.empty())
			{
				integral(input_img, _integral_img, _integral_img_sq, CV_64F);
			}

			sum = _integral_img;
			sqsum = _integral_img_sq;

			meanStdDev(templ, templMean, templSdv);

			templNorm = templSdv[0] * templSdv[0] + templSdv[1] * templSdv[1] + templSdv[2] * templSdv[2] + templSdv[3] * templSdv[3];

			if (templNorm < DBL_EPSILON && method == CV_TM_CCOEFF_NORMED)
			{
				result.setTo(1.0);
				return;
			}

			templSum2 = templNorm + templMean[0] * templMean[0] + templMean[1] * templMean[1] + templMean[2] * templMean[2] + templMean[3] * templMean[3];

			if (numType != 1)
			{
				templMean = cv::Scalar::all(0);
				templNorm = templSum2;
			}

			templSum2 /= invArea;
			templNorm = std::sqrt(templNorm);
			templNorm /= std::sqrt(invArea); // care of accuracy here

			q0 = (double*)sqsum.data;
			q1 = q0 + templ.cols;
			q2 = (double*)(sqsum.data + templ.rows*sqsum.step);
			q3 = q2 + templ.cols;
		}

		double* p0 = (double*)sum.data;
		double* p1 = p0 + templ.cols;
		double* p2 = (double*)(sum.data + templ.rows*sum.step);
		double* p3 = p2 + templ.cols;

		int sumstep = sum.data ? (int)(sum.step / sizeof(double)) : 0;
		int sqstep = sqsum.data ? (int)(sqsum.step / sizeof(double)) : 0;

		int i, j;

		for (i = 0; i < result.rows; i++)
		{
			float* rrow = result.ptr<float>(i);
			int idx = i * sumstep;
			int idx2 = i * sqstep;

			for (j = 0; j < result.cols; j++, idx += 1, idx2 += 1)
			{
				double num = rrow[j], t;
				double wndMean2 = 0, wndSum2 = 0;

				if (numType == 1)
				{

					t = p0[idx] - p1[idx] - p2[idx] + p3[idx];
					wndMean2 += t*t;
					num -= t*templMean[0];

					wndMean2 *= invArea;
				}

				if (isNormed || numType == 2)
				{

					t = q0[idx2] - q1[idx2] - q2[idx2] + q3[idx2];
					wndSum2 += t;

					if (numType == 2)
					{
						num = wndSum2 - 2 * num + templSum2;
						num = MAX(num, 0.);
					}
				}

				if (isNormed)
				{
					t = std::sqrt(MAX(wndSum2 - wndMean2, 0))*templNorm;
					if (fabs(num) < t)
						num /= t;
					else if (fabs(num) < t*1.125)
						num = num > 0 ? 1 : -1;
					else
						num = method != CV_TM_SQDIFF_NORMED ? 0 : 1;
				}

				rrow[j] = (float)num;
			}
		}
	}


	//===========================================================================
	// Point set and landmark manipulation functions
	//===========================================================================
	// Using Kabsch's algorithm for aligning shapes
	//This assumes that align_from and align_to are already mean normalised
	cv::Matx22d AlignShapesKabsch2D(const cv::Mat_<double>& align_from, const cv::Mat_<double>& align_to)
	{

		cv::SVD svd(align_from.t() * align_to);

		// make sure no reflection is there
		// corr ensures that we do only rotaitons and not reflections
		double d = cv::determinant(svd.vt.t() * svd.u.t());

		cv::Matx22d corr = cv::Matx22d::eye();
		if (d > 0)
		{
			corr(1, 1) = 1;
		}
		else
		{
			corr(1, 1) = -1;
		}

		cv::Matx22d R;
		cv::Mat(svd.vt.t()*cv::Mat(corr)*svd.u.t()).copyTo(R);

		return R;
	}

	cv::Matx22f AlignShapesKabsch2D_f(const cv::Mat_<float>& align_from, const cv::Mat_<float>& align_to)
	{

		cv::SVD svd(align_from.t() * align_to);

		// make sure no reflection is there
		// corr ensures that we do only rotaitons and not reflections
		float d = cv::determinant(svd.vt.t() * svd.u.t());

		cv::Matx22f corr = cv::Matx22f::eye();
		if (d > 0)
		{
			corr(1, 1) = 1;
		}
		else
		{
			corr(1, 1) = -1;
		}

		cv::Matx22f R;
		cv::Mat(svd.vt.t()*cv::Mat(corr)*svd.u.t()).copyTo(R);

		return R;
	}

	//=============================================================================
	// Basically Kabsch's algorithm but also allows the collection of points to be different in scale from each other
	cv::Matx22d AlignShapesWithScale(cv::Mat_<double>& src, cv::Mat_<double> dst)
	{
		int n = src.rows;

		// First we mean normalise both src and dst
		double mean_src_x = cv::mean(src.col(0))[0];
		double mean_src_y = cv::mean(src.col(1))[0];

		double mean_dst_x = cv::mean(dst.col(0))[0];
		double mean_dst_y = cv::mean(dst.col(1))[0];

		cv::Mat_<double> src_mean_normed = src.clone();
		src_mean_normed.col(0) = src_mean_normed.col(0) - mean_src_x;
		src_mean_normed.col(1) = src_mean_normed.col(1) - mean_src_y;

		cv::Mat_<double> dst_mean_normed = dst.clone();
		dst_mean_normed.col(0) = dst_mean_normed.col(0) - mean_dst_x;
		dst_mean_normed.col(1) = dst_mean_normed.col(1) - mean_dst_y;

		// Find the scaling factor of each
		cv::Mat src_sq;
		cv::pow(src_mean_normed, 2, src_sq);

		cv::Mat dst_sq;
		cv::pow(dst_mean_normed, 2, dst_sq);

		double s_src = sqrt(cv::sum(src_sq)[0] / n);
		double s_dst = sqrt(cv::sum(dst_sq)[0] / n);

		src_mean_normed = src_mean_normed / s_src;
		dst_mean_normed = dst_mean_normed / s_dst;

		double s = s_dst / s_src;

		// Get the rotation
		cv::Matx22d R = AlignShapesKabsch2D(src_mean_normed, dst_mean_normed);

		cv::Matx22d	A;
		cv::Mat(s * R).copyTo(A);

		cv::Mat_<double> aligned = (cv::Mat(cv::Mat(A) * src.t())).t();
		cv::Mat_<double> offset = dst - aligned;

		double t_x = cv::mean(offset.col(0))[0];
		double t_y = cv::mean(offset.col(1))[0];

		return A;

	}

	cv::Matx22f AlignShapesWithScale_f(cv::Mat_<float>& src, cv::Mat_<float> dst)
	{
		int n = src.rows;

		// First we mean normalise both src and dst
		float mean_src_x = cv::mean(src.col(0))[0];
		float mean_src_y = cv::mean(src.col(1))[0];

		float mean_dst_x = cv::mean(dst.col(0))[0];
		float mean_dst_y = cv::mean(dst.col(1))[0];

		cv::Mat_<float> src_mean_normed = src.clone();
		src_mean_normed.col(0) = src_mean_normed.col(0) - mean_src_x;
		src_mean_normed.col(1) = src_mean_normed.col(1) - mean_src_y;

		cv::Mat_<float> dst_mean_normed = dst.clone();
		dst_mean_normed.col(0) = dst_mean_normed.col(0) - mean_dst_x;
		dst_mean_normed.col(1) = dst_mean_normed.col(1) - mean_dst_y;

		// Find the scaling factor of each
		cv::Mat src_sq;
		cv::pow(src_mean_normed, 2, src_sq);

		cv::Mat dst_sq;
		cv::pow(dst_mean_normed, 2, dst_sq);

		float s_src = sqrt(cv::sum(src_sq)[0] / n);
		float s_dst = sqrt(cv::sum(dst_sq)[0] / n);

		src_mean_normed = src_mean_normed / s_src;
		dst_mean_normed = dst_mean_normed / s_dst;

		float s = s_dst / s_src;

		// Get the rotation
		cv::Matx22f R = AlignShapesKabsch2D_f(src_mean_normed, dst_mean_normed);

		cv::Matx22f	A = s * R;

		cv::Mat_<float> aligned = (cv::Mat(cv::Mat(A) * src.t())).t();
		cv::Mat_<float> offset = dst - aligned;

		float t_x = cv::mean(offset.col(0))[0];
		float t_y = cv::mean(offset.col(1))[0];

		return A;

	}

	// Useful utility for grabing a bounding box around a set of 2D landmarks (as a 1D 2n x 1 vector of xs followed by doubles or as an n x 2 vector)
	void ExtractBoundingBox(const cv::Mat_<float>& landmarks, float &min_x, float &max_x, float &min_y, float &max_y)
	{

		if (landmarks.cols == 1)
		{
			int n = landmarks.rows / 2;
			cv::MatConstIterator_<float> landmarks_it = landmarks.begin();

			for (int i = 0; i < n; ++i)
			{
				float val = *landmarks_it++;
				
				if (i == 0 || val < min_x)
					min_x = val;

				if (i == 0 || val > max_x)
					max_x = val;

			}

			for (int i = 0; i < n; ++i)
			{
				float val = *landmarks_it++;

				if (i == 0 || val < min_y)
					min_y = val;

				if (i == 0 || val > max_y)
					max_y = val;

			}
		}
		else
		{
			int n = landmarks.rows;
			for (int i = 0; i < n; ++i)
			{
				float val_x = landmarks.at<float>(i, 0);
				float val_y = landmarks.at<float>(i, 1);

				if (i == 0 || val_x < min_x)
					min_x = val_x;

				if (i == 0 || val_x > max_x)
					max_x = val_x;

				if (i == 0 || val_y < min_y)
					min_y = val_y;

				if (i == 0 || val_y > max_y)
					max_y = val_y;

			}

		}


	}

	// Computing landmarks (to be drawn later possibly)
	vector<cv::Point2f> CalculateVisibleLandmarks(const cv::Mat_<float>& shape2D, const cv::Mat_<int>& visibilities)
	{
		int n = shape2D.rows / 2;
		vector<cv::Point2f> landmarks;

		for (int i = 0; i < n; ++i)
		{
			if (visibilities.at<int>(i))
			{
				cv::Point2f featurePoint(shape2D.at<float>(i), shape2D.at<float>(i + n));

				landmarks.push_back(featurePoint);
			}
		}

		return landmarks;
	}

	// Computing landmarks (to be drawn later possibly)
	vector<cv::Point2f> CalculateAllLandmarks(const cv::Mat_<float>& shape2D)
	{

		int n = 0;
		vector<cv::Point2f> landmarks;

		if (shape2D.cols == 2)
		{
			n = shape2D.rows;
		}
		else if (shape2D.cols == 1)
		{
			n = shape2D.rows / 2;
		}

		for (int i = 0; i < n; ++i)
		{
			cv::Point2f featurePoint;
			if (shape2D.cols == 1)
			{
				featurePoint = cv::Point2f(shape2D.at<float>(i), shape2D.at<float>(i + n));
			}
			else
			{
				featurePoint = cv::Point2f(shape2D.at<float>(i, 0), shape2D.at<float>(i, 1));
			}

			landmarks.push_back(featurePoint);
		}

		return landmarks;
	}


//============================================================================
// Matrix reading functionality
//============================================================================

// Reading in a matrix from a stream
void ReadMat(std::ifstream& stream, cv::Mat &output_mat)
{
	// Read in the number of rows, columns and the data type
	int row, col, type;

	stream >> row >> col >> type;

	output_mat = cv::Mat(row, col, type);

	switch (output_mat.type())
	{
	case CV_64FC1:
	{
		cv::MatIterator_<double> begin_it = output_mat.begin<double>();
		cv::MatIterator_<double> end_it = output_mat.end<double>();

		while (begin_it != end_it)
		{
			stream >> *begin_it++;
		}
	}
	break;
	case CV_32FC1:
	{
		cv::MatIterator_<float> begin_it = output_mat.begin<float>();
		cv::MatIterator_<float> end_it = output_mat.end<float>();

		while (begin_it != end_it)
		{
			stream >> *begin_it++;
		}
	}
	break;
	case CV_32SC1:
	{
		cv::MatIterator_<int> begin_it = output_mat.begin<int>();
		cv::MatIterator_<int> end_it = output_mat.end<int>();
		while (begin_it != end_it)
		{
			stream >> *begin_it++;
		}
	}
	break;
	case CV_8UC1:
	{
		cv::MatIterator_<uchar> begin_it = output_mat.begin<uchar>();
		cv::MatIterator_<uchar> end_it = output_mat.end<uchar>();
		while (begin_it != end_it)
		{
			stream >> *begin_it++;
		}
	}
	break;
	default:
		printf("ERROR(%s,%d) : Unsupported Matrix type %d!\n", __FILE__, __LINE__, output_mat.type()); abort();


	}
}

void ReadMatBin(std::ifstream& stream, cv::Mat &output_mat)
{
	// Read in the number of rows, columns and the data type
	int row, col, type;

	stream.read((char*)&row, 4);
	stream.read((char*)&col, 4);
	stream.read((char*)&type, 4);

	output_mat = cv::Mat(row, col, type);
	int size = output_mat.rows * output_mat.cols * output_mat.elemSize();
	stream.read((char *)output_mat.data, size);

}

// Skipping lines that start with # (together with empty lines)
void SkipComments(std::ifstream& stream)
{
	while (stream.peek() == '#' || stream.peek() == '\n' || stream.peek() == ' ' || stream.peek() == '\r')
	{
		std::string skipped;
		std::getline(stream, skipped);
	}
}

}
