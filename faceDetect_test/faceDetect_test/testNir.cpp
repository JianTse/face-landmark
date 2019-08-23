// faceDetect_test.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "mtcnn_facedetect\mtcnn_opencv.h"
#include "mtcnn_facedetect\mtcnn_ncnn.h"

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

void  matchImgs(cv::Mat& nir, cv::Mat& bgr)
{
	cv::Point2f nirPts[4] = { cv::Point2f(340, 210), cv::Point2f(368, 265), cv::Point2f(405, 259), cv::Point2f(417, 199) };
	cv::Point2f bgrPts[4] = { cv::Point2f(310, 200), cv::Point2f(336, 256), cv::Point2f(376, 249), cv::Point2f(386, 189) };

	cv::Mat nir_warp;
	cv::Mat Trans = cv::getPerspectiveTransform(nirPts, bgrPts);
	warpPerspective(nir, nir_warp, Trans, cv::Size(bgr.cols, bgr.rows), CV_INTER_CUBIC);

	for (int i = 0; i < 4; i++)
	{
		cv::circle(nir, nirPts[i], 2, cv::Scalar(0, 0, 255), 2);
		cv::circle(bgr, bgrPts[i], 2, cv::Scalar(0, 0, 255), 2);
		cv::circle(nir_warp, bgrPts[i], 2, cv::Scalar(0, 0, 255), 2);
	}

	cv::imshow("nir", nir);
	cv::imshow("bgr", bgr);
	cv::imshow("nir_warp", nir_warp);
	cv::waitKey(0);
}

void  covBgrPt2NirPt(cv::Point& bgrPt, cv::Point& nirPt)
{
	nirPt.x = bgrPt.x + 30;
	nirPt.y = bgrPt.y + 10;
}

void  covBgrRect2NirRect(cv::Rect& bgrRect, cv::Rect& nirRect)
{
	nirRect = bgrRect;
	nirRect.x = bgrRect.x + 30;
	nirRect.y = bgrRect.y + 10;
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
	ret.resize(num_box);
	for (int i = 0; i < num_box; i++) {
		Bbox bbox = finalBbox[i];
		cv::Point pt1(bbox.x1, bbox.y1);
		cv::Point pt2(bbox.x2, bbox.y2);
		cv::Rect rect = valid_rect(pt1, pt2, img.cols, img.rows);
		if (rect.area() < 1)
		{
			continue;
		}

		//5个关键点
		cv::Point pt;
		std::vector<cv::Point> bgrLdmark5;
		std::vector<cv::Point> nirLdmark5;
		for (size_t j = 0; j < 5; j++)
		{
			pt.x = finalBbox[i].ppoint[j] + roi.x;
			pt.y = finalBbox[i].ppoint[j + 5] + roi.y;
			ret[i].ldmark.push_back(pt);
		}

		pt1.x = pt1.x + roi.x;
		pt1.y = pt1.y + roi.y;
		pt2.x = pt2.x + roi.x;
		pt2.y = pt2.y + roi.y;
		cv::Rect rect1 = valid_rect(pt1, pt2, img.cols, img.rows);		
		ret[i].rect = valid_rect(pt1, pt2, img.cols, img.rows);
	}

	return 1;
}

void  saveDetectInfo(cv::Mat& img, std::string& sampleDir, std::string& videoName, int frameCount, 
	std::vector<mtcnnRet>& bgrRet, std::vector<mtcnnRet>& nirRet)
{
	char info[1280];
	sprintf(info, "%s/%s_%d.txt", sampleDir.c_str(), videoName.c_str(), frameCount);
	FILE *fp;
	if ((fp = fopen(info, "w")) == NULL)
	{
		return;
	}
	
	fprintf(fp, "%d\n", bgrRet.size());
	fflush(fp);
	for (int i = 0; i < bgrRet.size(); i++)
	{
		fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", 
			bgrRet[i].rect.x, bgrRet[i].rect.y, bgrRet[i].rect.width, bgrRet[i].rect.height,
			bgrRet[i].ldmark[0].x, bgrRet[i].ldmark[0].y,
			bgrRet[i].ldmark[1].x, bgrRet[i].ldmark[1].y, 
			bgrRet[i].ldmark[2].x, bgrRet[i].ldmark[2].y, 
			bgrRet[i].ldmark[3].x, bgrRet[i].ldmark[3].y, 
			bgrRet[i].ldmark[4].x, bgrRet[i].ldmark[4].y);
		fflush(fp);
	}

	fprintf(fp, "%d\n", nirRet.size());
	fflush(fp);
	for (int i = 0; i < nirRet.size(); i++)
	{
		fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
			nirRet[i].rect.x, nirRet[i].rect.y, nirRet[i].rect.width, nirRet[i].rect.height,
			nirRet[i].ldmark[0].x, nirRet[i].ldmark[0].y,
			nirRet[i].ldmark[1].x, nirRet[i].ldmark[1].y,
			nirRet[i].ldmark[2].x, nirRet[i].ldmark[2].y,
			nirRet[i].ldmark[3].x, nirRet[i].ldmark[3].y,
			nirRet[i].ldmark[4].x, nirRet[i].ldmark[4].y);
		fflush(fp);
	}
	fclose(fp);

	sprintf(info, "%s/%s_%d.jpg", sampleDir.c_str(), videoName.c_str(), frameCount);
	cv::imwrite(info, img);
}

int detectVideo()
{
	const char* modelDir = "./mtcnn_model";
	MTCNN_NCNN  detector;
	detector.init(modelDir);

	std::string  videoName = "2019-07-18-15-57-09";
	std::string  videoType = "computer";

	std::string  rootDir = "E:/work/data/Anti-spoofing/beadwallet/";	
	std::string  videoFn = rootDir + "video/" + videoName +  ".avi";
	std::string  sampleRootDir;
	if (videoType == "pos")
	{
		sampleRootDir = rootDir + "sample/pos/";
	}
	else if (videoType == "paint")
	{
		sampleRootDir = rootDir + "sample/neg/paint/";
	}
	else if (videoType == "phone")
	{
		sampleRootDir = rootDir + "sample/neg/phone/";
	}
	else if (videoType == "monitor")
	{
		sampleRootDir = rootDir + "sample/neg/monitor/";
	}
	std::string  sampleDir = sampleRootDir + videoName;
	if (!fileExist(sampleDir))
	{
		creatDir(sampleDir);
	}

	cv::VideoCapture mCamera(videoFn.c_str());
	//cv::VideoCapture mCamera(1);
	if (!mCamera.isOpened()) {
		std::cout << "Camera opening failed..." << std::endl;
		system("pause");
		return 0;
	}

	cv::Mat frame;
	std::vector<mtcnnRet> retBgr, retNir;
	int nframe = 0;
	for (;;) {
		nframe++;
		mCamera >> frame;
		if (frame.empty()) break;         // check if at end
		cv::Rect nirRect = cv::Rect(0, 0, frame.cols / 2, frame.rows);
		cv::Rect bgrRect = cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows);
		
		detectFaceImg(detector, frame, bgrRect, retBgr);
		detectFaceImg(detector, frame, nirRect, retNir);
		
		//保存结果		
		saveDetectInfo(frame, sampleDir,  videoName, nframe, retBgr, retNir);

		drawMtcnnRet(frame, retBgr);
		drawMtcnnRet(frame, retNir);		
		
		//char info[1280];
		//sprintf(info, "img/%d.jpg", nframe);
		//cv::imwrite(info, frame);

		//cv::Mat bgr = frame(bgrRect).clone();
		//cv::Mat nir = frame(nirRect).clone();
		//matchImgs(nir,bgr);

		cv::imshow("frame", frame);
		int key = cv::waitKey(1);
		if ((char)key == ' ')
		{
			cv::waitKey();
		}
		else if ((char)key == 27)//esc
		{
			mCamera.release();	
			break;
		}
	}

	return 0;
}

void  resizeFace(cv::Mat& src, cv::Rect& faceRect, cv::Mat& dst)
{
	int extend = min(faceRect.width, faceRect.height);
	int extendSize = extend / 3;
	extendSize = (extendSize / 4) * 4;
	cv::Rect roi = cvExtendRect(faceRect, extendSize, extendSize);
	cvValidateRect(cv::Size(src.cols, src.rows), &roi);
	
	int maxSize = max(roi.width, roi.height);
	float scale = 480.0 / (float)maxSize;
	int dstW = int(roi.width * scale);
	int dstH = int(roi.height * scale);
	cv::resize(src(roi), dst, cv::Size(dstW, dstH));
}

int splitVideo(MTCNN_NCNN&  detector, std::string&  srcRootDir, std::string&  dstRootDir, std::string&  videoName)
{
	std::string  srcVideoFn = srcRootDir + videoName + ".avi";

	cv::Mat  bgr;
	char vName[1280];
	sprintf(vName, "%s%s_bgr.avi", dstRootDir.c_str(), videoName.c_str());
	cv::VideoWriter pVideo(vName, CV_FOURCC('M', 'J', 'P', 'G'), 25, cv::Size(640, 480));

	std::string  imgDir = dstRootDir + videoName;
	if (!fileExist(imgDir))
	{
		creatDir(imgDir.c_str());
	}

	cv::VideoCapture mCamera(srcVideoFn.c_str());
	//cv::VideoCapture mCamera(1);
	if (!mCamera.isOpened()) {
		std::cout << "Camera opening failed..." << std::endl;
		system("pause");
		return 0;
	}
	char info[1280];
	cv::Mat frame;
	std::vector<mtcnnRet> retBgr;
	int nframe = 0;
	for (;;) {
		nframe++;
		mCamera >> frame;
		if (frame.empty()) break;         // check if at end
		cv::Rect bgrRect = cv::Rect(0, 0, frame.cols / 2, frame.rows);
		cv::Rect nirRect = cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows);
				
		cv::Mat bgr = frame(bgrRect).clone();
		pVideo.write(bgr);
		detectFaceImg(detector, frame, bgrRect, retBgr);
		if (nframe % 30 == 0 && retBgr.size() > 0)
		{			
			sprintf(info, "%s/%s-%d.jpg", imgDir.c_str(), videoName.c_str(), nframe);
			
			cv::Mat dst;
			resizeFace(bgr, retBgr[0].rect, dst);
			cv::imwrite(info, dst);
		}

		drawMtcnnRet(frame, retBgr);
		cv::imshow("bgr", bgr);
		cv::imshow("frame", frame);
		int key = cv::waitKey(1);
		if ((char)key == ' ')
		{
			cv::waitKey();
		}
		else if ((char)key == 27)//esc
		{
			mCamera.release();
			break;
		}
	}
	pVideo.release();

	return 0;
}

int splitResizeVideo(MTCNN_NCNN&  detector, std::string&  srcRootDir, std::string&  dstRootDir, std::string&  videoName)
{
	std::string  srcVideoFn = srcRootDir + videoName + ".avi";

	cv::Mat  bgr;
	char vName[1280];
	sprintf(vName, "%s%s_bgr.avi", dstRootDir.c_str(), videoName.c_str());
	cv::VideoWriter pVideo(vName, CV_FOURCC('M', 'J', 'P', 'G'), 25, cv::Size(480, 480));

	std::string  imgDir = dstRootDir + videoName;
	//if (!fileExist(imgDir))
	//{
	//	creatDir(imgDir.c_str());
	//}

	cv::VideoCapture mCamera(srcVideoFn.c_str());
	//cv::VideoCapture mCamera(1);
	if (!mCamera.isOpened()) {
		std::cout << "Camera opening failed..." << std::endl;
		system("pause");
		return 0;
	}
	char info[1280];
	cv::Mat frame;
	std::vector<mtcnnRet> retBgr;
	int nframe = 0;
	for (;;) {
		nframe++;
		mCamera >> frame;
		if (frame.empty()) break;         // check if at end
		cv::Rect bgrRect = cv::Rect(0, 0, frame.cols / 2, frame.rows);
		cv::Rect nirRect = cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows);

		cv::Mat bgr = frame(bgrRect).clone();

		detectFaceImg(detector, frame, bgrRect, retBgr);
		cv::Mat dst;
		if (retBgr.size() > 0)
		{			
			resizeFace(bgr, retBgr[0].rect, dst);
			pVideo.write(dst);
			cv::imshow("bgr", dst);
		}		
		drawMtcnnRet(frame, retBgr);		
		cv::imshow("frame", frame);
		int key = cv::waitKey(1);
		if ((char)key == ' ')
		{
			cv::waitKey();
		}
		else if ((char)key == 27)//esc
		{
			mCamera.release();
			break;
		}
	}
	pVideo.release();

	return 0;
}

void  splitDir()
{
	const char* modelDir = "./mtcnn_model";
	MTCNN_NCNN  detector;
	detector.init(modelDir);

	std::string  srcRootDir = "E:/work/data/Anti-spoofing/beadwallet/videoSample/pos/record/";
	std::string  dstRootDir = "E:/work/data/Anti-spoofing/beadwallet/videoSample/pos/bgr_big/";	

	std::vector<std::string> files;

	//读入每个人脸是否笑
	const std::string list_file = srcRootDir + "video_list.txt";
	FILE* file_ptr = fopen(list_file.c_str(), "r");
	vector<string> imgPath;
	char video_name[1280];
	while (true) {
		const int status = fscanf(file_ptr, "%s\n",
			&video_name);
		if (status == EOF) {
			break;
		}
		string path = video_name;
		files.push_back(path);
	}
	fclose(file_ptr);

	for (int i = 0; i < files.size(); i++)
	{
		splitResizeVideo(detector, srcRootDir, dstRootDir, files[i]);
	}
}

int main()
{
	//detectVideo();

	splitDir();
	//std::string  videoName = "2019-07-18-13-34-41";
	//std::string  srcRootDir = "E:/work/data/Anti-spoofing/beadwallet/videoSample/pos/record/";
	//std::string  dstRootDir = "E:/work/data/Anti-spoofing/beadwallet/videoSample/pos/bgr/";
	//splitVideo(srcRootDir, dstRootDir, videoName);

	return 0;
}
