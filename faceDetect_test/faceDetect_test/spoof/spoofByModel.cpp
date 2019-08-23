#include "spoofByModel.h"

const int INPUT_DATA_WIDTH = 48;
const int INPUT_DATA_HEIGHT = 48;

const float IMG_MEAN = 127.5f;
const float IMG_INV_STDDEV = 1.f / 128.f;

SpoofByModel::SpoofByModel()
{
}

SpoofByModel::~SpoofByModel(){
	_ncnnNet.clear();
}

int SpoofByModel::init(const std::string &model_path)
{
	//ncnn
	std::string  param_fn = model_path + "/spoof.param";
	std::string  bin_fn = model_path + "/spoof.bin";
	int ret_param = _ncnnNet.load_param(param_fn.data());
	int ret_bin = _ncnnNet.load_model(bin_fn.data());
	if (ret_param != 0 || ret_bin != 0)
	{
		return 0;
	}		

	//opencv
	std::string  protoText = model_path + "/M3.prototxt";
	std::string  caffeModel = model_path + "/M3-hard.caffemodel";
	_cvNet = cv::dnn::readNetFromCaffe(protoText, caffeModel);
	if (_cvNet.empty()) {
		return 0;
	}

	return 1;
}

void SpoofByModel::destroy()
{
	_ncnnNet.clear();
}

std::vector<float> SpoofByModel::ncnnCheckFace(cv::Mat& img, cv::Rect& faceRect)
{
	cv::Mat patch = img(faceRect).clone();

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(patch.data, ncnn::Mat::PIXEL_BGR, patch.cols, patch.rows, 48, 48);
	const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
	const float norm_vals[3] = { 0.0078125,0.0078125,0.0078125 };
	in.substract_mean_normalize(mean_vals, norm_vals);
	ncnn::Extractor ex = _ncnnNet.create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(4);
	ex.input("data", in);
	ncnn::Mat out;
	ex.extract("prob", out);

	std::vector<float> cls_scores;
	cls_scores.resize(out.w);
	for (int j = 0; j<out.w; j++)
	{
		cls_scores[j] = out[j];
	}
	return cls_scores;
}

std::vector<float> SpoofByModel::cvCheckFace(cv::Mat& img, cv::Rect& faceRect)
{
	cv::Mat patch;
	cv::Size windowSize = cv::Size(INPUT_DATA_WIDTH, INPUT_DATA_HEIGHT);
	cv::resize(img(faceRect), patch, windowSize, 0, 0, cv::INTER_AREA);

	// build blob images from the inputs
	cv::Mat blobInput =
		cv::dnn::blobFromImage(patch, IMG_INV_STDDEV, cv::Size(),
			cv::Scalar(IMG_MEAN, IMG_MEAN, IMG_MEAN), false);

	_cvNet.setInput(blobInput, "data");

	const std::vector<cv::String> outBlobNames{ "prob" };
	std::vector<cv::Mat> outputBlobs;

	_cvNet.forward(outputBlobs, outBlobNames);
	cv::Mat scoresBlob = outputBlobs[0];

	const float *scores_data = (float *)scoresBlob.data;

	std::vector<float> scores;
	for (int i = 0; i < 4; i++)
	{
		scores.push_back(scores_data[i]);
	}
	return scores;
}