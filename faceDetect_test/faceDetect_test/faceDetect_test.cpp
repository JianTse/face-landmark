// faceDetect_test.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "mtcnn_facedetect\mtcnn_opencv.h"
#include "mtcnn_facedetect\mtcnn_ncnn.h"

int detectFaceImg(MTCNN&  detector, cv::Mat& img)
{
	float factor = 0.709f;
	float threshold[3] = { 0.8, 0.75, 0.7 };// { 0.7f, 0.6f, 0.6f };
	int minSize = 60;
	vector<FaceInfo> mFaceInfos = detector.Detect_mtcnn(img, minSize, threshold, factor, 3);

	//显示结果
	for (int i = 0; i < mFaceInfos.size(); i++) {
		int x = (int)mFaceInfos[i].bbox.xmin;
		int y = (int)mFaceInfos[i].bbox.ymin;
		int w = (int)(mFaceInfos[i].bbox.xmax - mFaceInfos[i].bbox.xmin + 1);
		int h = (int)(mFaceInfos[i].bbox.ymax - mFaceInfos[i].bbox.ymin + 1);
		cv::Rect faceRect = cv::Rect(x, y, w, h);

		cv::rectangle(img, faceRect, cv::Scalar(0, 0, 255));
		for (int k = 0; k < 5; k++)
		{
			int x = int(mFaceInfos[i].landmark[k * 2]);
			int y = int(mFaceInfos[i].landmark[k * 2 + 1]);
			cv::Point pt = cv::Point(x, y);			
			cv::circle(img, pt, 2, cv::Scalar(0, 255, 0), 2);
		}
	}
	
	return 1;
}

int detectFaceImg(MTCNN_NCNN&  detector, cv::Mat& imgBgr, cv::Mat& imgIR)
{
	std::vector<Bbox> finalBbox;
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels_resize(imgBgr.data, ncnn::Mat::PIXEL_BGR2RGB, imgBgr.cols, imgBgr.rows, imgBgr.cols, imgBgr.rows);

	std::vector<float>threshs(3);
	threshs[0] = 0.65;
	threshs[1] = 0.75;
	threshs[2] = 0.8;
	detector.setThreshold(threshs);
	detector.detect(ncnn_img, finalBbox);

	const int num_box = finalBbox.size();

	std::vector<cv::Rect> bbox;
	bbox.resize(num_box);
	for (int i = 0; i < num_box; i++) {
		Bbox bbox = finalBbox[i];
		cv::Point pt1(bbox.x1, bbox.y1);
		cv::Point pt2(bbox.x2, bbox.y2);
		cv::Rect rect = valid_rect(pt1, pt2, imgBgr.cols, imgBgr.rows);
		if (rect.area() < 1)
		{
			continue;
		}		
		//detector.checkFace(imgIR, rect);
		cv::rectangle(imgBgr, rect, cv::Scalar(0, 0, 0));
		//cv::rectangle(imgIR, rect, cv::Scalar(0, 0, 0));
	}

	return 1;
}

void  testImg()
{
	const char* modelDir = "./mtcnn_model";
	MTCNN  detector;
	detector.init(modelDir);

	//基于路径检测人脸及关键点
	const char* imgFn = "./img_28.jpg";
	cv::Mat img = cv::imread(imgFn);
	if (img.empty())
		return;

	detectFaceImg(detector, img);

	cv::imshow("frame", img);
	int key = cv::waitKey(0);
}

int detectFaceImg(MTCNN_NCNN&  detector, cv::Mat& imgBgr, cv::Mat& imgIR)
{
	std::vector<Bbox> finalBbox;
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels_resize(imgBgr.data, ncnn::Mat::PIXEL_BGR2RGB, imgBgr.cols, imgBgr.rows, imgBgr.cols, imgBgr.rows);

	std::vector<float>threshs(3);
	threshs[0] = 0.65;
	threshs[1] = 0.75;
	threshs[2] = 0.8;
	detector.setThreshold(threshs);
	detector.detect(ncnn_img, finalBbox);

	const int num_box = finalBbox.size();

	std::vector<cv::Rect> bbox;
	bbox.resize(num_box);
	for (int i = 0; i < num_box; i++) {
		Bbox bbox = finalBbox[i];
		cv::Point pt1(bbox.x1, bbox.y1);
		cv::Point pt2(bbox.x2, bbox.y2);
		cv::Rect rect = valid_rect(pt1, pt2, imgBgr.cols, imgBgr.rows);
		if (rect.area() < 1)
		{
			continue;
		}
		//detector.checkFace(imgIR, rect);
		cv::rectangle(imgBgr, rect, cv::Scalar(0, 0, 0));
		//cv::rectangle(imgIR, rect, cv::Scalar(0, 0, 0));
	}

	return 1;
}

int testVideo()
{
	const char* modelDir = "./mtcnn_model";
	//MTCNN  detector;
	MTCNN_NCNN  detector;
	detector.init(modelDir);

	char* fn = "E:/work/data/faceTrack/movie/1804_03_006_sylvester_stallone.avi";
	//cv::VideoCapture mCamera(fn);
	cv::VideoCapture mCamera1(1);
	cv::VideoCapture mCamera2(2);
	if (!mCamera1.isOpened() || !mCamera2.isOpened()) {
		std::cout << "Camera opening failed..." << std::endl;
		system("pause");
		return 0;
	}
	mCamera1.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	mCamera1.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	mCamera2.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	mCamera2.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	cv::Mat frame1, frame2;
	mCamera1 >> frame1;
	mCamera1 >> frame1;
	mCamera1 >> frame1;
	mCamera2 >> frame2;
	mCamera2 >> frame2;
	mCamera2 >> frame2;
	
	int nframe = 0;
	for (;;) {
		nframe++;
		mCamera1 >> frame1;
		if (frame1.empty()) break;         // check if at end
		mCamera2 >> frame2;
		if (frame2.empty()) break;         // check if at end

		//detectFaceImg(detector, frame1);
		//detectFaceImg(detector, frame2);
		detectFaceImg(detector, frame2, frame1);

		cv::imshow("IR", frame1);
		cv::imshow("BGR", frame2);
		int key = cv::waitKey(1);
		if ((char)key == ' ')
		{
			cv::waitKey();
		}
		else if ((char)key == 27)//esc
		{
			mCamera1.release();
			mCamera2.release();
			break;
		}
	}

	return 0;
}


int main()
{
	//testImg();
	testVideo();
	return 0;
}



