///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt
//
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////


#include "LandmarkDetectorParameters.h"

// System includes
#include <sstream>
#include <iostream>
#include <cstdlib>

#ifndef CONFIG_DIR
#define CONFIG_DIR "~"
#endif

using namespace std;

using namespace LandmarkDetector;

FaceModelParameters::FaceModelParameters()
{
	// initialise the default values
	init();

	//check_model_path();

	// For in the wild fitting these parameters are suitable
	window_sizes_init = vector<int>(4);
	//window_sizes_init[0] = 15; window_sizes_init[1] = 13; window_sizes_init[2] = 11; window_sizes_init[3] = 11;
	//window_sizes_init[0] = 0; window_sizes_init[1] = 13; window_sizes_init[2] = 11; window_sizes_init[3] = 9;
	window_sizes_init[0] = 0; window_sizes_init[1] = 0; window_sizes_init[2] = 11; window_sizes_init[3] = 9;

	window_sizes_current = window_sizes_init;

	sigma = 1.25;
	reg_factor = 35;
	weight_factor = 2.5;
	num_optimisation_iteration = 10;

	// For in-the-wild images use an in-the wild detector				
	curr_face_detector = MTCNN_DETECTOR;

	// Use multi-view hypotheses if in-the-wild setting
	multi_view = true;
}

FaceModelParameters::FaceModelParameters(vector<string> &arguments)
{
	// initialise the default values
	init();

	// First element is reserved for the executable location (useful for finding relative model locs)
	//boost::filesystem::path root = boost::filesystem::path(arguments[0]).parent_path();
	string root = "";

	bool* valid = new bool[arguments.size()];
	valid[0] = true;

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;

		if (arguments[i].compare("-mloc") == 0)
		{
			string model_loc = arguments[i + 1];
			model_location = model_loc;
			valid[i] = false;
			valid[i + 1] = false;
			i++;

		}
		if (arguments[i].compare("-fdloc") ==0)
		{
			string face_detector_loc = arguments[i + 1];
			haar_face_detector_location = face_detector_loc;
			curr_face_detector = HAAR_DETECTOR;
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		if (arguments[i].compare("-sigma") == 0)
		{
			stringstream data(arguments[i + 1]);
			data >> sigma;
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-w_reg") == 0)
		{
			stringstream data(arguments[i + 1]);
			data >> weight_factor;
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-reg") == 0)
		{
			stringstream data(arguments[i + 1]);
			data >> reg_factor;
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-multi_view") == 0)
		{

			stringstream data(arguments[i + 1]);
			int m_view;
			data >> m_view;

			multi_view = (bool)(m_view != 0);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-validate_detections") == 0)
		{
			stringstream data(arguments[i + 1]);
			int v_det;
			data >> v_det;

			validate_detections = (bool)(v_det != 0);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-n_iter") == 0)
		{
			stringstream data(arguments[i + 1]);
			data >> num_optimisation_iteration;

			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-q") == 0)
		{

			quiet_mode = true;

			valid[i] = false;
		}
		else if (arguments[i].compare("-wild") == 0)
		{
			// For in the wild fitting these parameters are suitable
			window_sizes_init = vector<int>(4);
			//window_sizes_init[0] = 15; window_sizes_init[1] = 13; window_sizes_init[2] = 11; window_sizes_init[3] = 11;
			window_sizes_init[0] = 0; window_sizes_init[1] = 13; window_sizes_init[2] = 11; window_sizes_init[3] = 9;

			sigma = 1.25;
			reg_factor = 35;
			weight_factor = 2.5;
			num_optimisation_iteration = 10;

			valid[i] = false;

			// For in-the-wild images use an in-the wild detector				
			curr_face_detector = MTCNN_DETECTOR;

			// Use multi-view hypotheses if in-the-wild setting
			multi_view = true;
		}
	}

	for (int i = (int)arguments.size() - 1; i >= 0; --i)
	{
		if (!valid[i])
		{
			arguments.erase(arguments.begin() + i);
		}
	}

#if   0
	// Make sure model_location is valid
	// First check working directory, then the executable's directory, then the config path set by the build process.
	boost::filesystem::path config_path = boost::filesystem::path(CONFIG_DIR);
	boost::filesystem::path model_path = boost::filesystem::path(model_location);
	if (boost::filesystem::exists(model_path))
	{
		model_location = model_path.string();
	}
	else if (boost::filesystem::exists(root/model_path))
	{
		model_location = (root/model_path).string();
	}
	else if (boost::filesystem::exists(config_path/model_path))
	{
		model_location = (config_path/model_path).string();
	}
	else
	{
		std::cout << "Could not find the landmark detection model to load" << std::endl;
	}

	if (model_path.stem().string().compare("main_ceclm_general") == 0)
	{
		curr_landmark_detector = CECLM_DETECTOR;
		sigma = 1.5f * sigma;
		reg_factor = 0.9f * reg_factor;
	}
	else if (model_path.stem().string().compare("main_clnf_general") == 0)
	{
		curr_landmark_detector = CLNF_DETECTOR;
	}
	else if (model_path.stem().string().compare("main_clm_general") == 0)
	{
		curr_landmark_detector = CLM_DETECTOR;
	}

	// Make sure face detector location is valid
	// First check working directory, then the executable's directory, then the config path set by the build process.
	model_path = boost::filesystem::path(haar_face_detector_location);
	if (boost::filesystem::exists(model_path))
	{
		haar_face_detector_location = model_path.string();
	}
	else if (boost::filesystem::exists(root / model_path))
	{
		haar_face_detector_location = (root / model_path).string();
	}
	else if (boost::filesystem::exists(config_path / model_path))
	{
		haar_face_detector_location = (config_path / model_path).string();
	}
	else
	{
		std::cout << "Could not find the HAAR face detector location" << std::endl;
	}

	// Make sure face detector location is valid
	// First check working directory, then the executable's directory, then the config path set by the build process.
	model_path = boost::filesystem::path(mtcnn_face_detector_location);
	if (boost::filesystem::exists(model_path))
	{
		mtcnn_face_detector_location = model_path.string();
	}
	else if (boost::filesystem::exists(root / model_path))
	{
		mtcnn_face_detector_location = (root / model_path).string();
	}
	else if (boost::filesystem::exists(config_path / model_path))
	{
		mtcnn_face_detector_location = (config_path / model_path).string();
	}
	else
	{
		std::cout << "Could not find the MTCNN face detector location" << std::endl;
	}
	check_model_path(root.string());
#endif
}

void FaceModelParameters::check_model_path(const std::string& root)
{
#if  0
	// Make sure model_location is valid
	// First check working directory, then the executable's directory, then the config path set by the build process.
	boost::filesystem::path config_path = boost::filesystem::path(CONFIG_DIR);
	boost::filesystem::path model_path = boost::filesystem::path(model_location);
	boost::filesystem::path root_path = boost::filesystem::path(root);

	if (boost::filesystem::exists(model_path))
	{
		model_location = model_path.string();
	}
	else if (boost::filesystem::exists(root_path / model_path))
	{
		model_location = (root_path / model_path).string();
	}
	else if (boost::filesystem::exists(config_path / model_path))
	{
		model_location = (config_path / model_path).string();
	}
	else
	{
		std::cout << "Could not find the landmark detection model to load" << std::endl;
	}
#endif
}

void FaceModelParameters::init()
{

	// number of iterations that will be performed at each scale
	num_optimisation_iteration = 5;

	// using an external face checker based on SVM
	validate_detections = true;

	// Using hierarchical refinement by default (can be turned off)
	refine_hierarchical = true;

	// Refining parameters by default
	refine_parameters = true;

	window_sizes_small = vector<int>(4);
	window_sizes_init = vector<int>(4);

	// For fast tracking
	window_sizes_small[0] = 0;
	window_sizes_small[1] = 9;
	window_sizes_small[2] = 7;
	window_sizes_small[3] = 0;

	// Just for initialisation
	window_sizes_init.at(0) = 11;
	window_sizes_init.at(1) = 9;
	window_sizes_init.at(2) = 7;
	window_sizes_init.at(3) = 5;

	face_template_scale = 0.3f;
	// Off by default (as it might lead to some slight inaccuracies in slowly moving faces)
	use_face_template = false;

	// For first frame use the initialisation
	window_sizes_current = window_sizes_init;

	//model_location = "model/main_ceclm_general.txt";
	model_location = "main_clnf_general.txt";
	curr_landmark_detector = CLNF_DETECTOR;   //CLM_DETECTOR, CLNF_DETECTOR, CECLM_DETECTOR

	sigma = 1.5f;
	reg_factor = 25.0f;
	weight_factor = 0.0f; // By default do not use NU-RLMS for videos as it does not work as well for them

	validation_boundary = 0.725f;

	limit_pose = true;
	multi_view = true;

	reinit_video_every = 2;

	// Face detection
	haar_face_detector_location = "classifiers/haarcascade_frontalface_alt.xml";
	mtcnn_face_detector_location = "model/mtcnn_detector/MTCNN_detector.txt";
	quiet_mode = false;

	// By default use MTCNN
	curr_face_detector = MTCNN_DETECTOR;

}

