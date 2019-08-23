//
// Created by LiHongJie on 2018/12/6.
//
#include "landmark.h"
#include <stdio.h>
#include <time.h>


Landmark::Landmark()
{
}

void Landmark::init() {
	//std::string param_file = model_path + "/onetm.proto";
	//std::string bin_file = model_path + "/onetm_30w.bin";
	//std::string param_file = model_path + "/onet2_30w.param";
	//std::string bin_file = model_path + "/onet2_30w.bin";
	//std::string param_file = model_path + "/shufflenetv2s2.proto";
	//std::string bin_file = model_path + "/shufflenetv2s2.bin";

	//std::string modelDir = "C:/Users/Administrator/Desktop/faceDetect_test/faceDetect_test/ldmark-model/ncnn";
	std::string modelDir = "./ldmark-model/ncnn";
	std::string param_file = modelDir + "/shufflenetv2_deploy.param";
	std::string bin_file = modelDir + "/shufflenetv2_iter_1500000.bin";

	int size;

	size = model.load_param(param_file.data());
	if (size != 0) {
		printf("load_param return %d", size);
	}

	size = model.load_model(bin_file.data());
	if (size != 0) {
		printf("load_model return:%d", size);
	}

	this->landmarks = nullptr;

}

void Landmark::set_param(int width, int height, int num_threads) {
    this->width = width;
    this->height = height;
    this->num_threads = num_threads;
}

void Landmark:: detect(ncnn::Mat& img_, ncnn::Mat& pred){
    //this->img = img_;
    ncnn::Mat input;
    long t1, t2;
    //t1 = get_current_time();
    if(img_.w != this->width || img_.h != this->height){
        resize_bilinear(img_, input, this->width, this->height);
    }else{
        input = img_;
    }
    //t2 = get_current_time();
    //printf("resize use time:%d ms",(t2-t1)/1000);

    //t1 = get_current_time();
    float mean[] = {127.5,127.5,127.5};
    float norm[] =  { 0.0078125f, 0.0078125f, 0.0078125f };//{1.0/128.0, 1.0/128.0, 1.0/128.0};
    //float mean[] = {0.0, 0.0, 0.0};
    //float norm[] = {1.0/255.0, 1.0/255.0, 1.0/255.0};
    input.substract_mean_normalize(mean, norm);
    //LOGD("input w:%d h:%d c:%d",input.w, input.h, input.c);
    //t2 = get_current_time();
   // LOGD("substract_mean_normalize use time:%d",(t2-t1)/1000);

    //t1 = get_current_time();
    ncnn::Extractor ex = this->model.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(this->num_threads);
    //t2 = get_current_time();
    //LOGD("set Extractor use time: %d", (t2-t1)/1000);

    //t1 = get_current_time();
    int ret = ex.input("data", input);

    ret = ex.extract("landmark_pred", pred);
    //t2 = get_current_time();
    //LOGD("extract use time；%d",(t2-t1)/1000);
}



Landmark::~Landmark() {
    this->model.clear();
}