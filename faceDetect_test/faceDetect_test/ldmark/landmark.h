//
// Created by LiHongJie on 2018/12/6.
//

#ifndef MTCNN_AS_LANDMARK_H
#define MTCNN_AS_LANDMARK_H
#include "../cnn/include/net.h"
#include <string>
#include <vector>
#include <time.h>
#include <algorithm>
#include <map>
#include <iostream>
#include <math.h>
#include "../util/common.h"
using namespace std;

class Landmark{
public:
    Landmark();
    ~Landmark();
	void init();
    void set_param(int width, int height, int num_threads);
    void detect(ncnn::Mat& img_, ncnn::Mat& pred);
    int width;
    int height;
    int num_points;
    int num_threads;

    int* get_landmarks(){
        if(!landmarks){
            return nullptr;
        }else{
            return landmarks;
        }
    }
private:
    ncnn::Net model;
    //ncnn::Mat img;
    int *landmarks;

};
#endif //MTCNN_AS_LANDMARK_H
