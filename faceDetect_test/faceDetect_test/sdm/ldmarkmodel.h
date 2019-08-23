#pragma once
#ifndef LDMARKMODEL_H_
#define LDMARKMODEL_H_

#include <iostream>
#include <vector>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "cereal.hpp"
#include "string.hpp"
#include "vector.hpp"
#include "binary.hpp"
#include "mat_cerealisation.hpp"

#include "helper.h"
#include "feature_descriptor.h"

//#include "../mtcnn_facedetect/mtcnn_opencv.h"

#define SDM_NO_ERROR        0       //�޴���
#define SDM_ERROR_FACEDET   200     //����ͨ��CascadeClassifier��⵽����
#define SDM_ERROR_FACEPOS   201     //����λ�ñ仯�ϴ󣬿���
#define SDM_ERROR_FACESIZE  202     //������С�仯�ϴ󣬿���
#define SDM_ERROR_FACENO    203     //�Ҳ�������
#define SDM_ERROR_IMAGE     204     //ͼ�����

#define SDM_ERROR_ARGS      400     //�������ݴ���
#define SDM_ERROR_MODEL     401     //ģ�ͼ��ش���

#define SDM_MAX_FACE_NUM        3       //����ͬʱ������������

using namespace std;
using namespace cv;
using std::string;

//�ع�����
class LinearRegressor{

public:
    LinearRegressor();

    cv::Mat predict(cv::Mat values);

private:
    cv::Mat weights;
    cv::Mat eigenvectors;
    cv::Mat meanvalue;
    cv::Mat x;
    bool isPCA;

    friend class cereal::access;
    /**
     * Serialises this class using cereal.
     *
     * @param[in] ar The archive to serialise to (or to serialise from).
     */
    template<class Archive>
    void serialize(Archive& ar)
    {
        ar(weights, meanvalue, x, isPCA);
        if(isPCA){
            ar(eigenvectors);
        }
    }
};


class ldmarkmodel{

public:
	ldmarkmodel();

	void init(std::string faceModel);

    void loadFaceDetModelFile(std::string filePath);

	void sdmProcessOneFaceByDlib(cv::Mat& img, cv::Rect& dlib_faceRects, vector<cv::Point>& landmarks68);

private:

	std::string faceModelPath;

    std::vector<cv::Rect> faceBox;


    std::vector<std::vector<int>> LandmarkIndexs;
    std::vector<int> eyes_index;
    cv::Mat meanShape;
    std::vector<HoGParam> HoGParams;
    bool isNormal;
    std::vector<LinearRegressor> LinearRegressors;
    cv::CascadeClassifier face_cascade;

    cv::Mat estimateHeadPoseMat;
    cv::Mat estimateHeadPoseMat2;
    int *estimateHeadPosePointIndexs;

    friend class cereal::access;
    /**
     * Serialises this class using cereal.
     *
     * @param[in] ar The archive to serialise to (or to serialise from).
     */
    template<class Archive>
    void serialize(Archive& ar)
    {
        ar(LandmarkIndexs, eyes_index, meanShape, HoGParams, isNormal, LinearRegressors);
    }

	void  refineDlibBox(cv::Rect& box);
};

//����ģ��
bool load_ldmarkmodel(std::string filename, ldmarkmodel &model);

#endif


