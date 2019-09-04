// faceDetect_test.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "util\common.h"
#include "mtcnn_facedetect\mtcnn_opencv.h"
#include "mtcnn_facedetect\mtcnn_ncnn.h"
#include "sdm\ldmarkmodel.h"
#include "poseEst\poseEst.h"
#include "pdm\pdmHelper.h"

struct ldmarkInfo
{
	cv::Mat img;
	cv::Rect rect;
	std::vector<cv::Point> ldmark5;
	std::vector<cv::Point> ldmark68;
	std::vector<float> eva;
};

void  drawLdmarkInfo(cv::Mat& img, ldmarkInfo& info, cv::Scalar& color)
{
	cv::rectangle(img, info.rect, color, 2);
	for (int i = 0; i < info.ldmark5.size(); i++)
	{
		cv::circle(img, info.ldmark5[i], 2, color, -1);
	}
	for (int i = 0; i < info.ldmark68.size(); i++)
	{
		cv::circle(img, info.ldmark68[i], 2, color, -1);
	}

	//char buff[255];
	//sprintf(buff, "%.2f,%.2f,%.2f", info.eva[0], info.eva[1], info.eva[2]);
	//putText(img, buff, cv::Point(info.rect.x, info.rect.y+info.rect.height), CV_FONT_HERSHEY_COMPLEX, 0.5, color);
}

void getFaceSanp(cv::Mat& bgr, cv::Rect faceRect, ldmarkInfo& info, ldmarkInfo& dstInfo)
{
	int extend = min(faceRect.width, faceRect.height);
	int extendSize = extend / 2;
	extendSize = (extendSize / 4) * 4;
	cv::Rect roi = cvExtendRect(faceRect, extendSize, extendSize);
	cvValidateRect(cv::Size(bgr.cols, bgr.rows), roi);
	dstInfo.img = bgr(roi).clone();
	int x = faceRect.x - roi.x;
	int y = faceRect.y - roi.y;
	dstInfo.rect = info.rect;
	dstInfo.rect.x = faceRect.x - roi.x;
	dstInfo.rect.y = faceRect.y - roi.y;
	for (int i = 0; i < info.ldmark68.size(); i++)
	{
		int x = info.ldmark68[i].x - roi.x;
		int y = info.ldmark68[i].y - roi.y;
		dstInfo.ldmark68.push_back(cv::Point(x, y));
	}
	for (int i = 0; i < info.ldmark5.size(); i++)
	{
		int x = info.ldmark5[i].x - roi.x;
		int y = info.ldmark5[i].y - roi.y;
		dstInfo.ldmark5.push_back(cv::Point(x, y));
	}
	for (int i = 0; i < info.eva.size(); i++)
	{
		dstInfo.eva.push_back(info.eva[i]);
	}
}

void cropFaceSanp(cv::Mat& srcImg, std::vector<cv::Point>& srcLdmarks, cv::Mat& dstImg, std::vector<cv::Point2f>& dstLdmarks)
{
	cv::Rect box = getBoundingBox(cv::Size(srcImg.cols, srcImg.rows), srcLdmarks);
	int y = box.y - box.height / 4;
	int h = box.height + box.height / 4;
	cv::Rect box1 = box;
	box1.y = y;
	box1.height = h;

	dstLdmarks.clear();
	int extend = min(box1.width, box1.height);
	int extendSize = extend / 2;
	extendSize = (extendSize / 4) * 4;
	cv::Rect roi = cvExtendRect(box1, extendSize, extendSize);
	cvValidateRect(cv::Size(srcImg.cols, srcImg.rows), roi);
	dstImg = srcImg(roi).clone();
	for (int i = 0; i < srcLdmarks.size(); i++)
	{
		float x = float(srcLdmarks[i].x - roi.x) / float(dstImg.cols);
		float y = float(srcLdmarks[i].y - roi.y) / float(dstImg.rows);
		dstLdmarks.push_back(cv::Point2f(x, y));
	}
}


int getPoseBin(std::vector<float>& eva)
{
	int pitch_bin = 0;
	if (eva[0]  > 15) pitch_bin = 0;
	else if (eva[0]  > -15) pitch_bin = 1;
	else  pitch_bin = 2;

	int yaw_bin = 0;
	if (eva[1]  > 45) yaw_bin = 0;
	else if (eva[1]  > 15) yaw_bin = 1;
	else if (eva[1]  > -15) yaw_bin = 2;
	else if (eva[1]  > -45) yaw_bin = 3;
	else yaw_bin = 4;

	int bin = pitch_bin * 5 + yaw_bin;
	return bin;
}

void saveSample(std::string& sampleRootDir ,ldmarkInfo&  _info, std::string& dataSet, int img_idx, int face_idx)
{
	int bin = getPoseBin(_info.eva);

	char info[255];
	sprintf(info, "%s/%d/%s_%d_%d.jpg", sampleRootDir.c_str(), bin, dataSet.c_str(), img_idx, face_idx);
	cv::imwrite(info, _info.img);	
}

void  usePdm(PDM_Helper& m_pdmHelper, cv::Rect&faceRect, std::vector<cv::Point>&ldmark5, 
	std::vector<float>& eva, std::vector<cv::Point>&ldmark68)
{
	cv::Vec3d rotation_hypothese;
	m_pdmHelper.get_rotation_hypothese(faceRect,ldmark5, rotation_hypothese);
	eva.push_back(rotation_hypothese[0] * 180.0 / CV_PI);
	eva.push_back(rotation_hypothese[1] * 180.0 / CV_PI);
	eva.push_back(rotation_hypothese[2] * 180.0 / CV_PI);

	//用pdm估计初始形状
	m_pdmHelper.estInitShapeWithMtcnn(faceRect, ldmark5, ldmark68);
}

void  useSdm(cv::Mat& img, cv::Rect&faceRect, std::vector<cv::Point>&ldmark5, ldmarkmodel& sdm_model, 
	std::vector<float>& eva, std::vector<cv::Point>&ldmark68)
{
	sdm_model.sdmProcessOneFaceByDlib(img, faceRect, ldmark68);
	estimateEav(ldmark68, eva);
}

void  detectImg(std::string& sampleRootDir, MTCNN_NCNN&  _detector, PDM_Helper& m_pdmHelper, ldmarkmodel& sdm_model, 
	cv::Mat& img, int img_idx, std::string& dataSet)
{
	std::vector<Bbox> finalBbox;
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, img.cols, img.rows);

	std::vector<float>threshs(3);
	threshs[0] = 0.65;
	threshs[1] = 0.75;
	threshs[2] = 0.8;
	_detector.setThreshold(threshs);
	_detector.detect(ncnn_img, finalBbox);

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
		ldmarkInfo  _info;
		_info.rect = rect;
		_info.rect = getSqureRect(cv::Size(img.cols, img.rows), _info.rect);
		std::vector<cv::Point> ldmark5;
		for (size_t j = 0; j < 5; j++)
		{
			float x = finalBbox[i].ppoint[j];
			float y = finalBbox[i].ppoint[j + 5];
			_info.ldmark5.push_back(cv::Point(x, y));
		}
	
		//sdm_model.sdmProcessOneFaceByDlib(img, _info.rect, _info.ldmark68);				
		//_pose.estimateEav(_info.ldmark68, 0, _info.eva);		

		//用sdm估计
		ldmarkInfo  sdm_info;
		std::vector<float> sdm_eva;
		std::vector<cv::Point>sdm_ldmark68;
		useSdm(img, _info.rect, _info.ldmark5, sdm_model, _info.eva, _info.ldmark68);

		//用pdm估计
		ldmarkInfo  pdm_info;
		std::vector<float> pdm_eva;
		std::vector<cv::Point>pdm_ldmark68;
		//usePdm(m_pdmHelper, _info.rect, _info.ldmark5, pdm_info.eva, pdm_info.ldmark68);

		ldmarkInfo dstInfo;
		getFaceSanp(img, _info.rect, _info, dstInfo);
		saveSample(sampleRootDir, dstInfo, dataSet, img_idx, i);

		char info[255];
		std::sprintf(info, "%d", i);
		drawLdmarkInfo(dstInfo.img, dstInfo, cv::Scalar(0, 255, 0));
		cv::imshow(info, dstInfo.img);

		//drawLdmarkInfo(img, _info);
		//drawLdmarkInfo(img, sdm_info, cv::Scalar(0,255,0));
		//drawLdmarkInfo(img, pdm_info, cv::Scalar(0,0,255));		
		//sprintf(buff, "%.2f,%.2f,%.2f", sdm_info.eva[0], sdm_info.eva[1], sdm_info.eva[2]);
		//putText(img, buff, cv::Point(10, 30), CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0));
		//sprintf(buff, "%.2f,%.2f,%.2f", pdm_info.eva[0], pdm_info.eva[1], pdm_info.eva[2]);
		//putText(img, buff, cv::Point(10, 60), CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255));
	}
}

void  cov_ldmark98_68(std::vector<cv::Point>& ldmark98, std::vector<cv::Point>& ldmark68)
{
	ldmark68.resize(68);
	//轮廓
	for (int i = 0; i <= 16; i++)
	{
		int dstIdx = i * 2;
		ldmark68[i] = ldmark98[dstIdx];
	}
	//左眉 
	for (int i = 17; i <= 21; i++)
	{
		int srcIdx = i+16;
		ldmark68[i] = ldmark98[srcIdx];
	}
	//右眉 
	for (int i = 22; i <= 26; i++)
	{
		int srcIdx = i + 20;
		ldmark68[i] = ldmark98[srcIdx];
	}

	//左眼
	ldmark68[36] = ldmark98[60];
	ldmark68[37] = ldmark98[61];
	ldmark68[38] = ldmark98[63];
	ldmark68[39] = ldmark98[64];
	ldmark68[40] = ldmark98[65];
	ldmark68[41] = ldmark98[67];

	//右眼
	ldmark68[42] = ldmark98[68];
	ldmark68[43] = ldmark98[69];
	ldmark68[44] = ldmark98[71];
	ldmark68[45] = ldmark98[72];
	ldmark68[46] = ldmark98[73];
	ldmark68[47] = ldmark98[75];

	//鼻子
	for (int i = 27; i <= 35; i++)
	{
		int srcIdx = i + 24;
		ldmark68[i] = ldmark98[srcIdx];
	}

	//嘴巴
	for (int i = 48; i <= 67; i++)
	{
		int srcIdx = i + 28;
		ldmark68[i] = ldmark98[srcIdx];
	}
}

std::string getDiffName(std::string dstImgDir, std::string& srcName)
{	
	std::string strBig = srcName;
	int idx = 1;
	while (1)
	{		
		std::string saveImgFn = dstImgDir + strBig;
		int  ret = fileExist(saveImgFn);
		if (ret == 1)  //如果已经存在
		{
			char info[1280];
			sprintf(info, "_%d.jpg", idx);			
			const std::string strsrc = ".jpg";
			const std::string strdst = info;
			string_replace(strBig, strsrc, strdst);
			idx++;
		}
		else
		{
			break;
		}
	}
	return strBig;
}

void  get_WFLW(MTCNN_NCNN&  _detector, PDM_Helper& m_pdmHelper, ldmarkmodel& sdm_model)
{
	std::string  rootDir = "E:/work/data/landmark/WFLW/";
	std::string  srcListFn = rootDir + "WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt";
	//std::string  srcListFn = rootDir + "WFLW_annotations/list_98pt_test/list_98pt_test.txt";
	//std::string  srcListFn = rootDir + "samples/list_98pt_pose_total.txt";

	ifstream infile(srcListFn);
	std::vector<std::string> caches;
	std::string s;
	while (getline(infile, s))
	{
		//从s中解析出整数
		caches.push_back(s);
	}	

	std::string  dstListFn = rootDir + "samples/WFLW_68pt_train.txt";

	std::string imgDir = "E:/work/data/landmark/WFLW/WFLW_images/";
	std::string sampleRootDir = "E:/work/data/landmark/clusterSamples/WFLW";
	std::string dataSet = "WFLW";
	std::string dstImgDir = rootDir + "samples/WFLW/";


	FILE *fp;
	if ((fp = fopen(dstListFn.c_str(), "w")) == NULL)
	{
		return;
	}

#if   1
	std::string pattern = " ";
	int startId = 0;
	int endId = 196;
	for (int i = 0; i < caches.size(); i++)
	{
		std::vector<std::string>params = split2(caches[i], pattern);
		std::string imgName = params[206];  //train
		//std::string imgName = params[196];  //train
		std::string imgFn = imgDir + imgName;
		cv::Mat img = cv::imread(imgFn.c_str());

		std::vector<cv::Point> ldmark98;
		str2Ldmak2(params, startId, endId, ldmark98);

		std::vector<cv::Point> ldmark68;
		cov_ldmark98_68(ldmark98, ldmark68);

		//detectImg(sampleRootDir, _detector, m_pdmHelper, sdm_model,  _pose,  img, i, dataSet);

		//计算pose
		std::vector<float> eva;
		estimateEav(ldmark68, eva);

		cv::Mat  faceSnap;
		std::vector<cv::Point2f> dstLdmark68;
		cropFaceSanp(img, ldmark68, faceSnap, dstLdmark68);

		std::string strBig = imgName;
		const std::string strsrc = "/";
		const std::string strdst = "_";
		string_replace(strBig, strsrc, strdst);
		std::string onlyImgName = getDiffName(dstImgDir, strBig);
		std::string saveImgFn = dstImgDir + onlyImgName;
		cv::imwrite(saveImgFn.c_str(), faceSnap);

		std::string ldStr = FloatLdmark2Str(dstLdmark68);
		std::string evaStr = eva2Str(eva);
		std::string retStr = onlyImgName + ldStr + evaStr;
		fprintf(fp, "WFLW/%s\n", retStr.c_str());
		fflush(fp);

		char buff[255];
		for (int j = 0; j < ldmark98.size(); j++)
		{
			cv::circle(img, ldmark98[j], 2, cv::Scalar(0, 0, 255), -1);
		}
		for (int j = 0; j < ldmark68.size(); j++)
		{
			int x = int(dstLdmark68[j].x * faceSnap.cols);
			int y = int(dstLdmark68[j].y * faceSnap.rows);

			sprintf(buff, "%d", j);
			putText(faceSnap, buff, cv::Point(x,y), CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255));
			cv::circle(faceSnap, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
		}
		
		sprintf(buff, "%.2f,%.2f,%.2f", eva[0], eva[1], eva[2]);
		putText(img, buff, cv::Point(10, 50), CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0));

		cv::imshow("img", faceSnap);
		cv::waitKey(1);		
	}
#else

	int filterCount = 0;
	std::string pattern = " ";
	int startId = 5;
	int endId = 142;
	for (int i = 0; i < caches.size(); i++)
	{
		std::vector<std::string>params = split2(caches[i], pattern);
		std::string imgName = params[0];
		std::string imgFn = rootDir + imgName;
		cv::Mat img = cv::imread(imgFn.c_str());

		std::vector<cv::Point> ldmark68;
		str2Ldmak2(params, startId, endId, ldmark68);

		//计算pose
		std::vector<float> eva;
		_pose.estimateEav(ldmark68, 0, eva);

		cv::Mat  faceSnap;
		std::vector<cv::Point> dstLdmark68;
		cropFaceSanp(img, ldmark68, faceSnap, dstLdmark68);

		std::string strBig = imgName;
		const std::string strsrc = "/";
		const std::string strdst = "_";
		string_replace(strBig, strsrc, strdst);
		std::string saveImgFn = dstImgDir + strBig;
		cv::imwrite(saveImgFn.c_str(), faceSnap);

		std::string ldStr = ldmark2Str(dstLdmark68);
		std::string evaStr = eva2Str(eva);
		std::string retStr = strBig + ldStr + evaStr;
		fprintf(fp, "%s\n", retStr.c_str());
		fflush(fp);

		for (int j = 0; j < dstLdmark68.size(); j++)
		{
			cv::circle(faceSnap, dstLdmark68[j], 2, cv::Scalar(0, 255, 0), -1);
		}
		char buff[255];
		sprintf(buff, "%.2f,%.2f,%.2f", eva[0], eva[1], eva[2]);
		putText(faceSnap, buff, cv::Point(10, 50), CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0));

		cv::imshow("img", faceSnap);
		cv::waitKey(0);
	}
#endif
	
	fclose(fp);
}

void  get_MTFL(MTCNN_NCNN&  _detector, PDM_Helper& m_pdmHelper, ldmarkmodel& sdm_model)
{
	ifstream infile("E:/work/data/landmark/MTFL/training.txt");
	std::vector<std::string> caches;
	std::string s;
	while (getline(infile, s))
	{
		//从s中解析出整数
		caches.push_back(s);
	}

	std::string imgDir = "E:/work/data/landmark/MTFL/";
	std::string sampleRootDir = "E:/work/data/landmark/clusterSamples/MTFL";
	std::string dataSet = "MTFL";

	std::string pattern = " ";
	int startId = 1;
	int endId = 11;
	for (int i = 0; i < caches.size(); i++)
	{
		std::vector<std::string>params = split2(caches[i], pattern);
		std::string imgFn = imgDir + params[0];
		cv::Mat img = cv::imread(imgFn.c_str());

		std::vector<cv::Point> ldmark5;
		//str2Ldmak5(params, startId, endId, ldmark5);
		detectImg(sampleRootDir, _detector, m_pdmHelper, sdm_model, img, i, dataSet);

		//cv::imshow("img", img);
		cv::waitKey(1);
	}
}

void  get_FDDB(MTCNN_NCNN&  _detector, PDM_Helper& m_pdmHelper, ldmarkmodel& sdm_model)
{
	ifstream infile("E:/work/data/faceDetect/FDDB/FDDB-folds/imgListAll.txt");
	std::vector<std::string> caches;
	std::string s;
	while (getline(infile, s))
	{
		//从s中解析出整数
		caches.push_back(s);
	}

	std::string imgDir = "E:/work/data/faceDetect/FDDB/";
	std::string sampleRootDir = "E:/work/data/landmark/clusterSamples/FDDB";
	std::string dataSet = "FDDB";

	std::string pattern = " ";
	for (int i = 0; i < caches.size(); i++)
	{
		std::string imgFn = imgDir + caches[i] + ".jpg";
		cv::Mat img = cv::imread(imgFn.c_str());
		detectImg(sampleRootDir, _detector, m_pdmHelper, sdm_model, img, i, dataSet);

		cv::imshow("img", img);
		cv::waitKey(1);
	}
}

void  get_CelebA(MTCNN_NCNN&  _detector, PDM_Helper& m_pdmHelper, ldmarkmodel& sdm_model)
{
	ifstream infile("E:/work/data/attribute/celebA/CelebA/Eval/list_eval_partition.txt");
	std::vector<std::string> caches;
	std::string s;
	while (getline(infile, s))
	{
		//从s中解析出整数
		caches.push_back(s);
	}

	std::string imgDir = "E:/work/data/attribute/celebA/CelebA/Img/img_celeba.7z/img_celeba/";
	std::string sampleRootDir = "E:/work/data/landmark/clusterSamples/CelebA";
	std::string dataSet = "CelebA";

	std::string pattern = " ";
	for (int i = 0; i < caches.size(); i++)
	{
		std::vector<std::string>params = split2(caches[i], pattern);
		std::string imgFn = imgDir + params[0];
		cv::Mat img = cv::imread(imgFn.c_str());
		detectImg(sampleRootDir, _detector, m_pdmHelper, sdm_model, img, i, dataSet);

		//cv::imshow("img", img);
		cv::waitKey(1);
	}
}

void  get_ibug300w(MTCNN_NCNN&  _detector, PDM_Helper& m_pdmHelper, ldmarkmodel& sdm_model)
{
	std::string  rootDir = "E:/work/data/landmark/ibug-300W/ibug_300W_large_face_landmark_dataset/";
	std::string  srcListFn = rootDir + "labels_ibug_300W_mirror_total.txt";

	ifstream infile(srcListFn);
	std::vector<std::string> caches;
	std::string s;
	while (getline(infile, s))
	{
		//从s中解析出整数
		caches.push_back(s);
	}

	std::string  dstListFn = rootDir + "labels_ibug_300W_total.txt";

	std::string  dstImgDir = rootDir + "ibug300w/";


	FILE *fp;
	if ((fp = fopen(dstListFn.c_str(), "w")) == NULL)
	{
		return;
	}

	int filterCount = 0;
	std::string pattern = " ";
	int startId = 5;
	int endId = 142;
	for (int i = 0; i < caches.size(); i++)
	{
		std::vector<std::string>params = split2(caches[i], pattern);
		std::string imgName = params[0]; 
		std::string imgFn = rootDir + imgName;

		int idx = imgName.find("_mirror");//在a中查找b.
		if (idx != string::npos)//不存在。
		{
			filterCount++;
			printf("%d, %d, filter mirror: %s\n", caches.size(), filterCount, imgName.c_str());
			continue;
		}			

		cv::Mat img = cv::imread(imgFn.c_str());

		std::vector<cv::Point> ldmark68;
		str2Ldmak2(params, startId, endId, ldmark68);

		//计算pose
		std::vector<float> eva;
		estimateEav(ldmark68, eva);

		cv::Mat  faceSnap;
		std::vector<cv::Point2f> dstLdmark68;
		cropFaceSanp(img, ldmark68, faceSnap, dstLdmark68);

		std::string strBig = imgName;
		const std::string strsrc = "/";
		const std::string strdst = "_";
		string_replace(strBig, strsrc, strdst);
		std::string saveImgFn = dstImgDir + strBig;
		cv::imwrite(saveImgFn.c_str(), faceSnap);

		std::string ldStr = FloatLdmark2Str(dstLdmark68);
		std::string evaStr = eva2Str(eva);
		std::string retStr = strBig + ldStr + evaStr;
		fprintf(fp, "ibug300w/%s\n", retStr.c_str());
		fflush(fp);		

		char buff[255];
		for (int j = 0; j < dstLdmark68.size(); j++)
		{			
			int x = int(dstLdmark68[j].x * faceSnap.cols);
			int y = int(dstLdmark68[j].y * faceSnap.rows);

			sprintf(buff, "%d", j);
			putText(faceSnap, buff, cv::Point(x, y), CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255));
			cv::circle(faceSnap, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
		}
		sprintf(buff, "%.2f,%.2f,%.2f", eva[0], eva[1], eva[2]);
		putText(faceSnap, buff, cv::Point(10, 50), CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0));

		cv::imshow("img", faceSnap);
		cv::waitKey(1);
	}
	printf("total: %d, mirror: %d\n", caches.size(), filterCount);

	fclose(fp);
}

void  get_BeadWallet(MTCNN_NCNN&  _detector, PDM_Helper& m_pdmHelper, ldmarkmodel& sdm_model)
{
	ifstream infile("E:/work/data/landmark/beadwallet/samples/srcAnnoLists.txt");
	std::vector<std::string> caches;
	std::string s;
	while (getline(infile, s))
	{
		//从s中解析出整数
		caches.push_back(s);
	}

	std::string imgDir = "E:/work/data/landmark/beadwallet/images/";
	std::string sampleRootDir = "E:/work/data/landmark/clusterSamples/MTFL";
	std::string dataSet = "MTFL";
	

	std::string addPoseAnno = "E:/work/data/landmark/beadwallet/samples/srcPoseAnnoLists.txt";
	FILE *fp;
	if ((fp = fopen(addPoseAnno.c_str(), "w")) == NULL)
	{
		return;
	}
	std::vector<int> poseBins(15);
	for (int i = 0; i < poseBins.size(); i++)
	{
		poseBins[i] = 0;
	}
	std::string pattern = " ";
	int startId = 1;
	int endId = 137;
	for (int i = 0; i < caches.size(); i++)
	{
		if (i % 100 == 0)
		{
			printf("%d\n", i);
		}		
		std::vector<std::string>params = split2(caches[i], pattern);
		std::string imgFn = imgDir + params[0];
		cv::Mat img = cv::imread(imgFn.c_str());
		cv::Mat imgClone = img.clone();

		startId = 1;
		endId = 137;
		std::vector<cv::Point> ldmark68;
		str2Ldmak2(params, startId, endId, ldmark68);

		startId = 1;
		endId = 255;
		std::vector<cv::Point> ldmark127;
		str2Ldmak2(params, startId, endId, ldmark127);	
		cv::Rect box = getBoundingBox(cv::Size(img.cols, img.rows), ldmark127);

		//计算pose
		std::vector<float> eva;
		estimateEav(ldmark68, eva);		

		for (int j = 0; j < ldmark127.size(); j++)
		{
			cv::circle(imgClone, ldmark127[j], 2, cv::Scalar(0, 0, 255), -1);
		}
		for (int j = 0; j < ldmark68.size(); j++)
		{
			cv::circle(imgClone, ldmark68[j], 2, cv::Scalar(0,255,0), -1);
		}				
		char buff[255];
		sprintf(buff, "%.2f,%.2f,%.2f", eva[0], eva[1], eva[2]);
		putText(imgClone, buff, cv::Point(10, 50), CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0));
		cv::rectangle(imgClone, box, cv::Scalar(0, 0, 0));
		cv::imshow("img", imgClone);
		cv::waitKey(1);
		
		float  score = _detector.checkFace(img, box);		
		if (score < 0.65f)
		{
			std::string strBig = params[0];
			const string strsrc = "/";
			const string strdst = "_";
			//string_replace(strBig, strsrc, strdst);
			sprintf(buff, "img/%s", strBig.c_str());
			cv::imwrite(buff, imgClone);
			//printf("bad landmark: %s, %f\n", imgFn, score);
			//cv::waitKey(1);
		}
		int bin = getPoseBin(eva);
		poseBins[bin] += 1;
		fprintf(fp, "images/%s %f %f %f\n", caches[i].c_str(), eva[0], eva[1], eva[2]);
		fflush(fp);
	}
	fclose(fp);	

	//保存统计pose的统计数量
	std::string poseStatic = "E:/work/data/landmark/beadwallet/samples/srcPose_static.txt";
	FILE *fp1;
	if ((fp1 = fopen(poseStatic.c_str(), "w")) == NULL)
	{
		return;
	}
	for (int i = 0; i < poseBins.size(); i++)
	{
		fprintf(fp1, "%d: %d\n", i, poseBins[i]);
		fflush(fp1);
	}
	fclose(fp1);
}


void normLdmark2Img(cv::Mat& img, std::vector<cv::Point2f>& srcLdmark, std::vector<cv::Point>& dstLdmark)
{
	dstLdmark.clear();
	for (int i = 0; i < srcLdmark.size(); i++)
	{
		int x = int(srcLdmark[i].x * img.cols);
		int y = int(srcLdmark[i].y * img.rows);
		dstLdmark.push_back(cv::Point(x, y));
	}
}
void  static_BeadWallet68(MTCNN_NCNN&  _detector, PDM_Helper& m_pdmHelper, ldmarkmodel& sdm_model)
{
	std::string  rootDir = "E:/work/data/landmark/beadwallet/68/";
	std::string  srcListFn = rootDir + "samples/anno_68pt_float.txt";

	ifstream infile(srcListFn);
	std::vector<std::string> caches;
	std::string s;
	while (getline(infile, s))
	{
		//从s中解析出整数
		caches.push_back(s);
	}

	std::string  dstListFn_int = rootDir + "samples/filter_anno_68pt_int.txt";
	FILE *fp_int;
	if ((fp_int = fopen(dstListFn_int.c_str(), "w")) == NULL)
	{
		return;
	}
	std::string  dstListFn_float = rootDir + "samples/filter_anno_68pt_float.txt";
	FILE *fp_float;
	if ((fp_float = fopen(dstListFn_float.c_str(), "w")) == NULL)
	{
		return;
	}

	std::vector<int> poseBins(15);
	for (int i = 0; i < poseBins.size(); i++)
	{
		poseBins[i] = 0;
	}
	std::string pattern = " ";
	int startId = 1;
	int endId = 137;
	char buff[1280];
	for (int i = 0; i < caches.size(); i++)
	{
		if (i % 100 == 0)
		{
			printf("total: %d, %d\n", caches.size(), i);
		}		
		std::vector<std::string>params = split2(caches[i], pattern);
		std::string imgFn = rootDir + params[0];
		cv::Mat img = cv::imread(imgFn.c_str());
		cv::Mat imgClone = img.clone();

		startId = 1;
		endId = 137;
		std::vector<cv::Point2f> ldmark68_float;
		str2FLdmak(params, startId, endId, ldmark68_float);

		std::vector<cv::Point> ldmark68_int;
		normLdmark2Img(img, ldmark68_float, ldmark68_int);

		//计算pose
		std::vector<float> eva;
		estimateEav(ldmark68_int, eva);
		std::string evaStr = eva2Str(eva);		

		//获取boundbox
		cv::Rect boundbox = getBoundingBox(cv::Size(img.cols, img.rows), ldmark68_int);
		float clearCof = clarityJudge(img, ldmark68_int);
		int size = boundbox.width * boundbox.height;

		std::string strBig = imgFn;		

		//显示结果
		for (int j = 0; j < ldmark68_int.size(); j++)
		{
			cv::circle(imgClone, ldmark68_int[j], 2, cv::Scalar(0, 255, 0), -1);
		}
		sprintf(buff, "%.2f,%.2f", eva[1], clearCof);
		putText(imgClone, buff, cv::Point(10, 30), CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0));
		cv::imshow("img", imgClone);
		cv::waitKey(1);
		if (size < 300 * 300 || clearCof < 120)
		{
			const string strsrc2 = "images/";
			const string strdst2 = "filter_size_clear/bad/images/";
			string_replace(strBig, strsrc2, strdst2);
			cv::imwrite(strBig, img);

			printf("filter: %d, %.2f, %s\n", size, clearCof, params[0].c_str());
			cv::waitKey(1);
			continue;
		}		
		else
		{
			const string strsrc1 = "images/";
			const string strdst1 = "filter_size_clear/good/images/";
			string_replace(strBig, strsrc1, strdst1);
		}
		
		//sprintf(buff, "img/%s", strBig.c_str());
		cv::imwrite(strBig, img);
		//printf("bad landmark: %s, %f\n", imgFn, score);
		//cv::waitKey(1);	

		//保存为int
		std::string ldStr_int = IntLdmark2Str(ldmark68_int);		
		std::string retStr_int = params[0] + ldStr_int + evaStr;
		fprintf(fp_int, "%s\n", retStr_int.c_str());
		fflush(fp_int);

		//保存为float
		std::string ldStr_float = FloatLdmark2Str(ldmark68_float);
		std::string retStr_float = params[0] + ldStr_float + evaStr;
		fprintf(fp_float, "%s\n", retStr_float.c_str());
		fflush(fp_float);

		int bin = getPoseBin(eva);
		poseBins[bin] += 1;		
	}
	fclose(fp_int);
	fclose(fp_float);

	//保存统计pose的统计数量
	std::string poseStatic = "E:/work/data/landmark/beadwallet/68/srcFilterPose_static.txt";
	FILE *fp1;
	if ((fp1 = fopen(poseStatic.c_str(), "w")) == NULL)
	{
		return;
	}
	for (int i = 0; i < poseBins.size(); i++)
	{
		fprintf(fp1, "%d: %d\n", i, poseBins[i]);
		fflush(fp1);
	}
	fclose(fp1);
}


int main()
{
	//mtcnn
	const char* mtcnnModelDir = "./mtcnn_model";
	//MTCNN  detector;
	MTCNN_NCNN  _detector;
	_detector.init(mtcnnModelDir);

	//pdm
	const char* ldmarkModelDir = "./ldmark-model";
	PDM_Helper m_pdmHelper;
	m_pdmHelper.init(ldmarkModelDir);

	//sdm
	ldmarkmodel sdm_model;
	char sdmModelFilePath[1280];
	sprintf(sdmModelFilePath, "%s/ldmark.bin", ldmarkModelDir);
	sdm_model.init(sdmModelFilePath);
	if (!load_ldmarkmodel(sdmModelFilePath, sdm_model)) {
		return 0;
	}

	//get_ibug300w(_detector, m_pdmHelper, sdm_model);
	//get_WFLW( _detector, m_pdmHelper, sdm_model);
	//get_FDDB(_detector, m_pdmHelper, sdm_model);
	//get_MTFL(_detector, m_pdmHelper, sdm_model);
	//get_CelebA(_detector, m_pdmHelper, sdm_model);
	//get_BeadWallet(_detector, m_pdmHelper, sdm_model);

	static_BeadWallet68(_detector, m_pdmHelper, sdm_model);

	//testVideo();
	return 0;
}



