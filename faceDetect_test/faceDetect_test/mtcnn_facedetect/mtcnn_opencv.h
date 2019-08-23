
#ifndef _MTCNN_OPENCV__H__
#define _MTCNN_OPENCV__H__

//Created by Jack Yu
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
using namespace std;
//using namespace cv;

static const float pnet_stride = 2;
static const float pnet_cell_size = 12;
static const int pnet_max_detect_num = 5000;
//mean & std
static const float mean_val = 127.5f;
static const float std_val = 0.0078125f;
//minibatch size
static const int step_size = 128;


typedef struct FaceBox {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
} FaceBox;
typedef struct FaceInfo {
    float bbox_reg[4];
    float landmark_reg[10];
    float landmark[10];
    FaceBox bbox;
} FaceInfo;



class MTCNN {
public:
    MTCNN();

	int init(const string& proto_model_dir);

    vector<FaceInfo> Detect_mtcnn(cv::Mat& src, const int min_size, const float* threshold, const float factor, const int stage);
//protected:
    vector<FaceInfo> ProposalNet(const cv::Mat& img, int min_size, float threshold, float factor);
    vector<FaceInfo> NextStage(const cv::Mat& image, vector<FaceInfo> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold);
    void BBoxRegression(vector<FaceInfo>& bboxes);
    void BBoxPadSquare(vector<FaceInfo>& bboxes, int width, int height);
    void BBoxPad(vector<FaceInfo>& bboxes, int width, int height);
    void GenerateBBox(cv::Mat* confidence, cv::Mat* reg_box, float scale, float thresh);
    std::vector<FaceInfo> NMS(std::vector<FaceInfo>& bboxes, float thresh, char methodType);
    float IoU(float xmin, float ymin, float xmax, float ymax, float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom = false);



//    std::shared_ptr<dnn::Net> PNet_;
//    std::shared_ptr<dnn::Net> ONet_;
//    std::shared_ptr<dnn::Net> RNet_;
public:
    cv::dnn::Net PNet_;
    cv::dnn::Net RNet_;
    cv::dnn::Net ONet_;

    std::vector<FaceInfo> candidate_boxes_;
    std::vector<FaceInfo> total_boxes_;
};

#endif