#include "shufflenet_ncnn.h"

const int INPUT_DATA_WIDTH = 112;
const int INPUT_DATA_HEIGHT = 112;

const float IMG_MEAN = 127.5f;
const float IMG_INV_STDDEV = 1.f / 128.f;

ShuffleNet_Ldmark::ShuffleNet_Ldmark() 
{
}

ShuffleNet_Ldmark::~ShuffleNet_Ldmark() {
	this->model.clear();
}

void ShuffleNet_Ldmark::init(const std::string& modelDir) {
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

void ShuffleNet_Ldmark::set_param(int width, int height, int num_threads) {
	this->width = width;
	this->height = height;
	this->num_threads = num_threads;
}

void ShuffleNet_Ldmark::detect(ncnn::Mat& img_, ncnn::Mat& pred) {
	//this->img = img_;
	ncnn::Mat input;
	long t1, t2;
	//t1 = get_current_time();
	if (img_.w != this->width || img_.h != this->height) {
		resize_bilinear(img_, input, this->width, this->height);
	}
	else {
		input = img_;
	}
	//t2 = get_current_time();
	//printf("resize use time:%d ms",(t2-t1)/1000);

	//t1 = get_current_time();
	float mean[] = { 127.5,127.5,127.5 };
	float norm[] = { 0.0078125f, 0.0078125f, 0.0078125f };//{1.0/128.0, 1.0/128.0, 1.0/128.0};
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
	//LOGD("extract use time��%d",(t2-t1)/1000);
}

cv::Mat ShuffleNet_Ldmark::cropImage(const cv::Mat &img, cv::Rect r) 
{
	cv::Mat m = cv::Mat::zeros(r.height, r.width, img.type());
	int dx = std::abs(std::min(0, r.x));
	if (dx > 0) {
		r.x = 0;
	}
	r.width -= dx;
	int dy = std::abs(std::min(0, r.y));
	if (dy > 0) {
		r.y = 0;
	}
	r.height -= dy;
	int dw = std::abs(std::min(0, img.cols - 1 - (r.x + r.width)));
	r.width -= dw;
	int dh = std::abs(std::min(0, img.rows - 1 - (r.y + r.height)));
	r.height -= dh;
	if (r.width > 0 && r.height > 0) {
		img(r).copyTo(m(cv::Range(dy, dy + r.height), cv::Range(dx, dx + r.width)));
	}
	return m;
}


std::vector<cv::Point> ShuffleNet_Ldmark::run(const cv::Mat &img, cv::Rect& faceRect)
{
	int left = faceRect.x;
	int right = left + faceRect.width;
	int top = faceRect.y;
	int bottom = top + faceRect.height;

	cv::Mat  faceImg = cropImage(img, faceRect);
	
	//ncnn::Mat img = ncnn::Mat::from_pixels((unsigned char *)imageDate, ncnn::Mat::PIXEL_RGBA2BGR,width, height);
	ncnn::Mat face = ncnn::Mat::from_pixels_resize(faceImg.data, ncnn::Mat::PIXEL_BGR, faceImg.cols, faceImg.rows, 112, 112);
	ncnn::Mat pred;

	std::vector<cv::Point> ldmarks;
	detect(face, pred);
	if (pred.w == 0 || pred.h == 0) {		
		return ldmarks;
	}

	int pointNum = pred.w*pred.h*pred.c;
	int w = right - left;
	int h = bottom - top;

	//LOGD("pred.w:%d, pred.h:%d",pred.w,pred.h);
	for (int i = 0; i<pointNum; i = i + 2) {
		int x = cvRound(pred[i] * w + left);
		int y = cvRound(pred[i + 1] * h + top);
		ldmarks.push_back(cv::Point(x, y));
	}

	return ldmarks;
}
