// faceDetect_test.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "mtcnn_facedetect\mtcnn_opencv.h"
#include "mtcnn_facedetect\mtcnn_ncnn.h"
#include "util\common.h"
#include "ldmark\shufflenet_ncnn.h"
#include "ldmark\m3_ncnn.h"
#include "sdm\ldmarkmodel.h"
#include "poseEst\poseEst.h"
#include "pdm\pdmHelper.h"

struct mtcnnRet
{
	std::vector<cv::Point> ldmark;
	cv::Rect rect;
};

void  drawMtcnnRet(cv::Mat& img, std::vector<mtcnnRet>& ret)
{
	for (int i = 0; i < ret.size(); i++)
	{
		for (size_t j = 0; j < 5; j++)
		{
			cv::circle(img, ret[i].ldmark[j], 2, cv::Scalar(0, 255, 0), -1);
		}
		cv::rectangle(img, ret[i].rect, cv::Scalar(0, 255, 0));
	}
}

int detectFaceImg(MTCNN_NCNN&  detector, cv::Mat& img, cv::Rect& roi, std::vector<mtcnnRet>& ret)
{
	ret.clear();
	cv::Mat  detectPatch;
	detectPatch = img(roi).clone();

	std::vector<Bbox> finalBbox;
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels_resize(detectPatch.data, ncnn::Mat::PIXEL_BGR2RGB, detectPatch.cols, detectPatch.rows, detectPatch.cols, detectPatch.rows);

	std::vector<float>threshs(3);
	threshs[0] = 0.7;
	threshs[1] = 0.8;
	threshs[2] = 0.8;
	detector.setThreshold(threshs);
	//detector.detect(ncnn_img, finalBbox);
	detector.detectMaxFace(ncnn_img, finalBbox);

	const int num_box = finalBbox.size();

	std::vector<cv::Rect> bbox;
	bbox.resize(num_box);
	for (int i = 0; i < num_box; i++) {
		Bbox bbox = finalBbox[i];
		cv::Point pt1(bbox.x1, bbox.y1);
		cv::Point pt2(bbox.x2, bbox.y2);
		cv::Rect rect = valid_rect(pt1, pt2, img.cols, img.rows);
		if (rect.area() < 1)
		{
			continue;
		}

		mtcnnRet  _ret;

		//5个关键点
		cv::Point pt;
		std::vector<cv::Point> bgrLdmark5;
		std::vector<cv::Point> nirLdmark5;
		for (size_t j = 0; j < 5; j++)
		{
			pt.x = finalBbox[i].ppoint[j] + roi.x;
			pt.y = finalBbox[i].ppoint[j + 5] + roi.y;
			_ret.ldmark.push_back(pt);
		}

		pt1.x = pt1.x + roi.x;
		pt1.y = pt1.y + roi.y;
		pt2.x = pt2.x + roi.x;
		pt2.y = pt2.y + roi.y;
		cv::Rect rect1 = valid_rect(pt1, pt2, img.cols, img.rows);
		_ret.rect = valid_rect(pt1, pt2, img.cols, img.rows);

		ret.push_back(_ret);
	}

	return 1;
}

std::string  savePose(char* imgFn, std::vector<cv::Point>& ldmarks, std::vector<float>& eva)
{
	std::string retStr = imgFn;
	char retInfo[1280];
	for (int i = 0; i < ldmarks.size(); i++)
	{
		sprintf(retInfo, " %d %d", ldmarks[i].x, ldmarks[i].y);
		retStr += retInfo;
	}
	for (int i = 0; i < eva.size(); i++)
	{
		sprintf(retInfo, " %f", eva[i]);
		retStr += retInfo;
	}
	return retStr;
}

void  savePoseSample(cv::Mat& img, std::vector<cv::Point>& ldmarks, std::vector<float>& eva, int imgIdx, std::string& saveImgDir, FILE* fp)
{
	char imgFn[1280];
	sprintf(imgFn, "%s%d.jpg", saveImgDir, imgIdx);
	cv::imwrite(imgFn, img);

	std::string retStr = savePose(imgFn, ldmarks, eva);

	fprintf(fp, "%s\n", retStr.c_str());
	fflush(fp);
}

int detectVideo()
{
	const char* detectModelDir = "./mtcnn_model";
	MTCNN_NCNN  detector;
	detector.init(detectModelDir);

	const char* ldmarkModelDir = "./ldmark-model/ncnn";
	ShuffleNet_Ldmark  ldmark;
	ldmark.init(ldmarkModelDir);

	M3_Ldmark m3_ldmark;
	m3_ldmark.init(ldmarkModelDir);

	//pdm
	const char* pdmModelDir = "./ldmark-model";
	PDM_Helper m_pdmHelper;
	m_pdmHelper.init(pdmModelDir);

	//sdm
	ldmarkmodel sdm_model;
	const char* sdmModelFilePath = "./ldmark-model/ldmark.bin";
	sdm_model.init(sdmModelFilePath);
	if (!load_ldmarkmodel(sdmModelFilePath, sdm_model)) {
		return 0;
	}

	//pose
	CPoseEst  _pose;

	std::vector<mtcnnRet> retBgr, retNir;

#if 0
	cv::Mat  img = cv::imread("E:/work/data/landmark/beadwallet/images/2069.jpg");
	cv::Mat  bigImg;
	cv::resize(img, bigImg, cv::Size(img.cols * 1.7, img.rows * 1.7));
	detectFaceImg(detector, bigImg, cv::Rect(0,0, bigImg.cols, bigImg.rows), retBgr);
	char info[1280];
	for (int i = 0; i < retBgr.size(); i++)
	{
		std::vector<cv::Point>ldmarks;

		sdm_model.sdmProcessOneFaceByDlib(bigImg, retBgr[i].rect, ldmarks);
		//ldmarks = ldmark.run(bigImg, retBgr[i].rect);
		//ldmarks = m3_ldmark.run(bigImg, retBgr[i].rect);
		cv::rectangle(bigImg, retBgr[i].rect, cv::Scalar(0, 255, 0));
		//cv::putText(frame, info, cv::Point(retBgr[i].rect.x, retBgr[i].rect.y), 1, 1, color);
		for (int k = 0; k < ldmarks.size(); k++)
		{
			cv::circle(bigImg, ldmarks[k], 1, cv::Scalar(0, 0, 255), -1);
			sprintf(info, "%d", k);
			cv::putText(bigImg, info, ldmarks[k], 1, 0.8, cv::Scalar(0, 255, 0));
		}
	}
	cv::imwrite("E:/work/data/landmark/ibug-300W/label_68.png", bigImg);
	cv::imshow("img", bigImg);
	cv::waitKey(0);
#endif

	//cv::VideoCapture mCamera(videoFn.c_str());
	cv::VideoCapture mCamera(0);
	if (!mCamera.isOpened()) {
		std::cout << "Camera opening failed..." << std::endl;
		system("pause");
		return 0;
	}

	cv::Mat frame;
	int saveFlag = 0;	
	int nframe = 0;
	for (;;) {
		nframe++;
		mCamera >> frame;
		if (frame.empty()) break;         // check if at end
		cv::Rect bgrRect = cv::Rect(0, 0, frame.cols, frame.rows);
		detectFaceImg(detector, frame, bgrRect, retBgr);

		cv::Mat frameClone = frame.clone();

		char info[1280];
		for (int i = 0; i < retBgr.size(); i++)
		{
			std::vector<float> eva;
			std::vector<cv::Point>src_ldmarks;
			std::vector<cv::Point>ldmarks;
			//src_ldmarks = ldmark.run(frame, retBgr[i].rect);
			ldmarks = m3_ldmark.run(frame, retBgr[i].rect);
			//_pose.estimateEav(src_ldmarks, 0, eva);

			//cv::rectangle(frame, retBgr[i].rect, cv::Scalar(0,255,0));

			//sprintf(info, "%.2f, %.2f, %.2f", eva[0], eva[1], eva[2]);
			//cv::putText(frameClone, info, cv::Point(10, 50), 2, 2, cv::Scalar(0,0,255));

			for (int k = 0; k < ldmarks.size(); k++)
			{
				cv::circle(frame, ldmarks[k], 1, cv::Scalar(0,255,0),-1);
			}
			for (int k = 0; k < src_ldmarks.size(); k++)
			{
				cv::circle(frameClone, src_ldmarks[k], 1, cv::Scalar(0, 255, 0), -1);
			}
		}

		//drawMtcnnRet(frame, retBgr);

		cv::imshow("frame", frame);
		cv::imshow("src", frameClone);
		int key = cv::waitKey(1);
		if ((char)key == ' ')
		{
			cv::waitKey();
		}
		if ((char)key == 's')
		{
			saveFlag = 1;			
		}
		else if ((char)key == 27)//esc
		{
			mCamera.release();
			break;
		}		
	}	

	return 0;
}




int main()
{
	detectVideo();
	

	return 0;
}
