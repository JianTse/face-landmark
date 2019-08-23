//
// Created by Longqi on 2017/11/18..
//

/*
 * TO DO : change the P-net and update the generat box
 */

#include "spoofOpencv.h"

const int INPUT_DATA_WIDTH = 48;
const int INPUT_DATA_HEIGHT = 48;

const float IMG_MEAN = 127.5f;
const float IMG_INV_STDDEV = 1.f / 128.f;

cvSpoofNetwork::cvSpoofNetwork()
{
}

cvSpoofNetwork::~cvSpoofNetwork(){
}

int cvSpoofNetwork::init(const std::string &model_path)
{
	std::string  protoText = model_path + "/M3.prototxt";
	std::string  caffeModel = model_path + "/M3.caffemodel";
	_net = cv::dnn::readNetFromCaffe(protoText, caffeModel);
	if (_net.empty()) {
		return 0;
	}
	return 1;
}

void cvSpoofNetwork::destroy()
{
}

std::vector<float> cvSpoofNetwork::run(cv::Mat& img, cv::Rect& faceRect)
{
	cv::Mat patch;
	cv::Size windowSize = cv::Size(INPUT_DATA_WIDTH, INPUT_DATA_HEIGHT);
	cv::resize(img(faceRect), patch, windowSize, 0, 0, cv::INTER_AREA);

	// build blob images from the inputs
	auto blobInput =
		cv::dnn::blobFromImage(patch, IMG_INV_STDDEV, cv::Size(),
			cv::Scalar(IMG_MEAN, IMG_MEAN, IMG_MEAN), false);

	_net.setInput(blobInput, "data");

	const std::vector<cv::String> outBlobNames{ "prob" };
	std::vector<cv::Mat> outputBlobs;

	_net.forward(outputBlobs, outBlobNames);
	cv::Mat scoresBlob = outputBlobs[0];

	const float *scores_data = (float *)scoresBlob.data;

	std::vector<float> scores;
	for (int i = 0; i < 4; i++)
	{
		scores.push_back(scores_data[i]);
	}
	return scores;
}