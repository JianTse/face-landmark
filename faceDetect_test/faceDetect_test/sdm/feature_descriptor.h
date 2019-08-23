#ifndef FEATURE_DESCRIPTOR_H_
#define FEATURE_DESCRIPTOR_H_


#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "cereal.hpp"
#include "string.hpp"
#include "vector.hpp"
#include "binary.hpp"
#include "mat_cerealisation.hpp"

//extern "C" {
    #include "hog.h" // From the VLFeat C library
//}

struct HoGParam
{
    VlHogVariant vlhog_variant;
    int num_cells;
    int cell_size;
    int num_bins;
    float relative_patch_size; // the patch size we'd like in percent of the IED of the current image
    // note: alternatively, we could dynamically vary cell_size. Guess it works if the hog features are somehow normalised.

private:
    friend class cereal::access;
    /**
     * Serialises this class using cereal.
     *
     * @param[in] ar The archive to serialise to (or to serialise from).
     */
    template<class Archive>
    void serialize(Archive& ar)
    {
        ar(vlhog_variant, num_cells, cell_size, num_bins, relative_patch_size);
    }
};


struct HOG_Class {
	HoGParam  param;
	int ww, dd, hh;
	int wwhh;
	VlHog* hog;
	cv::Mat hogArray;
	cv::Mat hogDescriptor;
	cv::Mat normImg;
	cv::Mat norm32Img;
};


cv::Mat CalculateHogDescriptor(cv::Mat &src, cv::Mat shape, std::vector<int> LandmarkIndexs, std::vector<int> eyes_index, HoGParam mHoGParam);
cv::Mat fast_CalculateHogDescriptor(cv::Mat &src, cv::Mat shape, std::vector<int> LandmarkIndexs, std::vector<int> eyes_index, std::vector<HOG_Class>& stageHogs);
cv::Mat CalculateHogDescriptor_xj(cv::Mat &src, cv::Mat shape, std::vector<int> LandmarkIndexs, std::vector<int> eyes_index, std::vector<HOG_Class>& stageHogs);
#endif
