// faceDetect_test.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "mtcnn_facedetect\mtcnn_opencv.h"
#include "mtcnn_facedetect\mtcnn_ncnn.h"
#include "util\common.h"
#include "spoof\spoof_ncnn.h"

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

int  getMaxIdx(vector<float>& probs)
{
	int idx = 0;
	float maxScore = 0;
	for (int i = 0; i < probs.size(); i++)
	{
		if (probs[i] > maxScore)
		{
			maxScore = probs[i];
			idx = i;
		}
	}
	return idx;
}

int detectVideo()
{
	const char* modelDir = "./mtcnn_model";
	MTCNN_NCNN  detector;
	detector.init(modelDir);

	SPOOF_NCNN  spoof;
	spoof.init(modelDir);

	//cv::VideoCapture mCamera(videoFn.c_str());
	cv::VideoCapture mCamera(2);
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
		cv::Rect bgrRect = cv::Rect(0, 0, frame.cols, frame.rows);
		detectFaceImg(detector, frame, bgrRect, retBgr);

		char info[1280];
		for (int i = 0; i < retBgr.size(); i++)
		{
			vector<float> scores = spoof.checkFace(frame, retBgr[i].rect);
			sprintf(info, "%.2f, %.2f, %.2f, %.2f", scores[0], scores[1], scores[2], scores[3]);

			cv::Scalar color = cv::Scalar(0, 255, 0);
			int idx = getMaxIdx(scores);
			if (idx == 0)
			{
				color = cv::Scalar(0, 255, 0);
			}
			else
			{
				color = cv::Scalar(0, 0, 255);
			}
			cv::rectangle(frame, retBgr[i].rect, color);
			cv::putText(frame, info, cv::Point(retBgr[i].rect.x, retBgr[i].rect.y), 1, 1, color);
		}

		//drawMtcnnRet(frame, retBgr);

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


int testSpoof()
{
	const char* modelDir = "./mtcnn_model";
	MTCNN_NCNN  detector;
	detector.init(modelDir);

	SPOOF_NCNN  spoof;
	spoof.init(modelDir);

	//paint
	std::string  imgDir = "E:/work/data/Anti-spoofing/beadwallet/sample/";
	std::string  rootDir = "E:/work/data/Anti-spoofing/beadwallet/sample/beadwallet/";

	std::vector<std::string> files;

	//读入每个人脸是否笑
	const std::string list_file = rootDir + "val_list.txt";
	FILE* file_ptr = fopen(list_file.c_str(), "r");
	vector<string> imgPath;
	vector<int> labels;
	int label;
	char video_name[1280];
	while (true) {
		const int status = fscanf(file_ptr, "%s %d\n",
			&video_name, &label);
		if (status == EOF) {
			break;
		}
		string path = video_name;
		labels.push_back(label);
		files.push_back(path);
	}
	fclose(file_ptr);

	int  rightCount = 0;
	int  errorCount = 0;
	for (int i = 0; i < files.size(); i++)
	{
		std::string  imgFn = imgDir + files[i];
		cv::Mat img = cv::imread(imgFn.c_str());
		cv::Rect rect = cv::Rect(0, 0, img.cols, img.rows);
		vector<float> probs = spoof.checkFace(img, rect);

		int clsId = getMaxIdx(probs);
		if (clsId == labels[i])
		{
			rightCount++;
		}
		else
		{
			errorCount++;
		}
	}

	float acc = (float)rightCount / float(rightCount + errorCount);
	printf("acc: %f\n", acc);

	return 0;
}


float  getSampleSim(cv::Mat& tmp, cv::Mat& img, cv::Rect& faceRect)
{
	cv::Mat image_source;
	cv::resize(img(faceRect), image_source, cv::Size(64, 64));
	cv::Mat image_matched;
	cv::matchTemplate(image_source, tmp, image_matched, cv::TM_CCOEFF_NORMED);

	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(image_matched, &minVal, &maxVal, &minLoc, &maxLoc);

	return maxVal;
}

int  filterSample(cv::Mat& tmp, cv::Mat& img, cv::Rect& faceRect)
{
	if (faceRect.width < 10 || faceRect.height < 10)
		return 0;

	int ret = 1;
	if (tmp.empty())
	{
		cv::Mat patch = RectTools::subwindow(img, faceRect, cv::BORDER_REPLICATE);
		cv::resize(patch, tmp, cv::Size(64, 64));
	}
	else
	{
		float sim = getSampleSim(tmp, img, faceRect);
		if (sim > 0.85)
		{
			ret = 0;
		}
		else
		{
			cv::Mat patch = RectTools::subwindow(img, faceRect, cv::BORDER_REPLICATE);
			cv::resize(patch, tmp, cv::Size(64, 64));
		}
	}
	return ret;
}

int getFaceInVideo(MTCNN_NCNN&  detector, SPOOF_NCNN&  spoof, std::string&  srcRootDir, std::string&  dstRootDir, std::string&  videoName, int label)
{
	std::string  srcVideoFn = srcRootDir + videoName + ".avi";

	//cv::VideoCapture mCamera(srcVideoFn.c_str());
	cv::VideoCapture mCamera(2);
	if (!mCamera.isOpened()) {
		std::cout << "Camera opening failed..." << std::endl;
		system("pause");
		return 0;
	}
	char timeS[1280];
	char info[1280];
	cv::Mat tmp;
	cv::Mat frame;
	int nframe = 0;
	for (;;) {
		nframe++;
		mCamera >> frame;
		if (frame.empty()) break;         // check if at end
		cv::Rect bgrRect = cv::Rect(0, 0, frame.cols, frame.rows);
		cv::Rect nirRect = cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows);

		std::vector<mtcnnRet> retBgr;
		cv::Mat bgr = frame(bgrRect).clone();
		detectFaceImg(detector, frame, bgrRect, retBgr);
		
		GetTimeString(timeS, 1280);
		if (retBgr.size() > 0)
		{
			cv::Rect faceRect = retBgr[0].rect;
			vector<float> probs = spoof.checkFace(frame, faceRect);
			int clsId = getMaxIdx(probs);
			if (clsId != label)
			{
				int ret = filterSample(tmp, frame, faceRect);
				if (ret == 1)
				{
					sprintf(info, "%s%s-%d.jpg", dstRootDir.c_str(), timeS, nframe);
					cv::Mat patch = RectTools::subwindow(frame, faceRect, cv::BORDER_REPLICATE);
					cv::imwrite(info, patch);
				}				
			}

			//画框
			cv::Scalar color = cv::Scalar(0, 255, 0);
			if (clsId == label)
			{
				color = cv::Scalar(0, 255, 0);
			}
			else
			{
				color = cv::Scalar(0, 0, 255);
			}
			sprintf(info, "%.2f, %.2f, %.2f, %.2f", probs[0], probs[1], probs[2], probs[3]);
			cv::rectangle(frame, faceRect, color);
			cv::putText(frame, info, cv::Point(faceRect.x, faceRect.y), 1, 1, cv::Scalar(0, 255, 255));
		}

		if (!tmp.empty())
		{
			cv::imshow("tmp", tmp);
		}

		char info[1280];
		sprintf(info, "%d", nframe);
		cv::putText(frame, info, cv::Point(10, 50), 1, 1, cv::Scalar(0, 255, 255));

		//drawMtcnnRet(frame, retBgr);
		//cv::imshow("bgr", bgr);
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

void  detectDir()
{
	const char* modelDir = "./mtcnn_model";
	MTCNN_NCNN  detector;
	detector.init(modelDir);

	SPOOF_NCNN  spoof;
	spoof.init(modelDir);

	//pos
	//std::string  srcRootDir = "E:/work/data/Anti-spoofing/beadwallet/videoSample/pos/record/";
	//std::string  dstRootDir = "E:/work/data/Anti-spoofing/beadwallet/videoSample/pos/pos_img/";	

	//phone
	//std::string  srcRootDir = "E:/work/data/Anti-spoofing/beadwallet/videoSample/neg/phone/video/";
	//std::string  dstRootDir = "E:/work/data/Anti-spoofing/beadwallet/videoSample/neg/phone/img/";

	//monitor
	//std::string  srcRootDir = "E:/work/data/Anti-spoofing/beadwallet/videoSample/neg/monitor/video/filter/";
	//std::string  dstRootDir = "E:/work/data/Anti-spoofing/beadwallet/videoSample/neg/monitor/img/";

	//paint
	std::string  srcRootDir = "E:/work/data/Anti-spoofing/beadwallet/videoSample/neg/paint/video/";
	std::string  dstRootDir = "E:/work/data/Anti-spoofing/beadwallet/sample/hardSample/neg/phone/img/";

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
		//getFaceInVideo(detector, spoof, srcRootDir, dstRootDir, files[i]);
	}
}

int main()
{
	//detectVideo();
	//testSpoof();

	const char* modelDir = "./mtcnn_model";
	MTCNN_NCNN  detector;
	detector.init(modelDir);

	SPOOF_NCNN  spoof;
	spoof.init(modelDir);

	std::string  srcRootDir = "E:/work/data/Anti-spoofing/beadwallet/videoSample/neg/paint/video/";
	std::string  dstRootDir = "E:/work/data/Anti-spoofing/beadwallet/sample/hardSample/neg/phone/img/";
	std::string  videoName = "phone";
	int label = 2;
	getFaceInVideo(detector, spoof, srcRootDir, dstRootDir, videoName, label);

	//detectDir();
	//std::string  videoName = "2019-07-18-13-34-41";
	//std::string  srcRootDir = "E:/work/data/Anti-spoofing/beadwallet/videoSample/pos/record/";
	//std::string  dstRootDir = "E:/work/data/Anti-spoofing/beadwallet/videoSample/pos/bgr/";
	//splitVideo(srcRootDir, dstRootDir, videoName);

	return 0;
}
