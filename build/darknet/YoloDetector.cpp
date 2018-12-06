#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include "YoloDetector.h"
#include "../../src/yolo_v2_class.hpp"
extern "C" {
#include "../../src/data.h"
#include "../../src/option_list.h"
}


namespace yolo_detector {

class YoloErrorStateIsUndefinedException : public std::runtime_error
{
public:
	YoloErrorStateIsUndefinedException(const std::string& message):
		std::runtime_error(("YoloErrorStateIsUndefinedException: " + message))
	{}
};

/*! \class YoloDetectorImplementation
* �����-������� ��� darknet(yolo), ���������� ���������� ��������� ��� ������
*								   �� yolo_cpp_dll.dll ������� �������������� ��������
*/
class YoloDetectorImplementation : public IYoloDetectorOwned
{
public:
	// \brief ������������� ���������� �������������: ������, ��������� ����, ������ gpu
	ErrorCode Initialize(const char* config_filepath, const char* weight_filepath, const char *name_list_filepath, int gpu_id, int width_override, int height_override)
	{
		try {
			if (!config_filepath || !name_list_filepath)
			{
				throw std::runtime_error("config_filepath and name_list_filepath must be non-nullptr paths");
			}
			std::ifstream fin_config(config_filepath);
			if (!fin_config.is_open())
				throw std::runtime_error(std::string("Error opening  ") + config_filepath);
			std::string weight_filepath_string;
			if (weight_filepath != nullptr)
			{
				weight_filepath_string = weight_filepath;
			}
			bool try_load_weights = weight_filepath_string.size() > 0;
			if (try_load_weights)
			{
				std::ifstream fin_weight(weight_filepath_string);
				if (!fin_weight.is_open())
					throw std::runtime_error(std::string("Error opening  ") + weight_filepath_string);
			}
			std::ifstream fin_name_list(name_list_filepath);
			if (!fin_name_list.is_open())
				throw std::runtime_error(std::string("Error opening  ") + name_list_filepath);
			m_detector_from_darknet.reset(new Detector( config_filepath, weight_filepath_string, m_net_classes, gpu_id, width_override, height_override));
			list *plist = get_paths(const_cast<char*>(name_list_filepath));
			m_list_names.reserve(plist->size);
			for (node * node_it = plist->front; node_it != nullptr; node_it = node_it->next) 
			{
				m_list_names.push_back(static_cast<const char*>(node_it->val));
			}
			free_list_contents(plist);
			free_list(plist);
			if (m_net_classes != m_list_names.size()) throw std::runtime_error("files from different networks");
			m_weights_loaded = try_load_weights;
		}
		catch (const YoloErrorStateIsUndefinedException& error) {
			//some critical error happened, so the original darknet tried to call exit() for such sitauation,
			//the most typical is "not enough cuda memory".
			//This was converted to C++ exception, but internal darknet state is undefinde (maybe huge resource leaks).
			//So don't try to deallocate m_detector_from_darknet, save error message for logging, and return kCriticalErrorCantContinue
			m_last_error = error.what();
			return kCriticalErrorCantContinue;
		}
		catch (std::exception& error) {
			m_detector_from_darknet.reset();
			m_last_error = error.what();
		}
		catch (...) {
			m_detector_from_darknet.reset();
			m_last_error = "Unknown error";
		}
		if (m_detector_from_darknet == nullptr)
			return kError;

		return kOk;
	}

	int GetInputWidth() override {
		return m_detector_from_darknet->get_net_width();
	};
	int GetInputHeight() override {
		return m_detector_from_darknet->get_net_height();
	}
	int GetInputColorDepth() override {
		return m_detector_from_darknet->get_net_color_depth();
	}

	// \brief ��������� ����������� �������� �� ������������ � ������� ��������� ���� Yolo 
	virtual ErrorCode DetectObjects(const uint8_t* image, int width, int height, int bpl, int bpp, float threshold, bool use_mean)
	{
		if (!m_weights_loaded)
		{
			m_last_error = "net was loaded without weights";
			return kError;
		}
		const int net_channels = m_detector_from_darknet->get_net_color_depth();
		//���������, ��� ����� ������� �� ������� ����������� ����� ����� �� ��� �������� � ���������.
		//������������ ���������� - ��� ������ 4-� ��������� �������� �� ���� 3-� ��������� ���� -
        //��� ���������� ��� BGRx � ��������� ����� �������������.
		if (bpp != net_channels && !(bpp == 4 && net_channels == 3))
		{
			m_last_error = std::string("Incorrect color depth: image depth is ") + std::to_string(bpp)
				+ ", net depth is " + std::to_string(net_channels);
			return kError;
		}
		try
		{
			m_input_image.resize(width*std::abs(height)*net_channels);
			const int height_dir = height >= 0 ? -1 : +1;
			height = std::abs(height);
			const uint8_t* input_lines_ptr = height_dir > 0 ? image : image + bpl*(height - 1);
			bpl = height_dir >= 0 ? bpl : -bpl;
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					float* output_lines_ptr = &m_input_image[(x + width*y)];
					for (int c = net_channels - 1; c >= 0; --c) {
						*output_lines_ptr = static_cast<float>(input_lines_ptr[x*bpp + c]) / 255.f;
						output_lines_ptr += height*width;
					}
				}
				input_lines_ptr += bpl;
			}

			image_t temp_image = {};
			temp_image.c = net_channels;
			temp_image.data = m_input_image.data();
			temp_image.h = height;
			temp_image.w = width;

			m_current_bbox_result = m_detector_from_darknet->detect(temp_image, m_probs, threshold, use_mean);
			m_current_result.resize(m_current_bbox_result.size());
			for (int index = 0; index < m_current_result.size(); ++index)
			{
				m_current_result[index].x = m_current_bbox_result[index].x;
				m_current_result[index].y = m_current_bbox_result[index].y;
				m_current_result[index].w = m_current_bbox_result[index].w;
				m_current_result[index].h = m_current_bbox_result[index].h;
				m_current_result[index].prob = m_current_bbox_result[index].prob;
				m_current_result[index].obj_id = m_current_bbox_result[index].obj_id;
				m_current_result[index].feautures = &m_probs[m_list_names.size()*index];
			}
			return kOk;
		}
		catch (const YoloErrorStateIsUndefinedException& error) {
			//some critical error happened, so the original darknet tried to call exit() for such sitauation,
			//the most typical is "not enough cuda memory".
			//This was converted to C++ exception, but internal darknet state is undefinde (maybe huge resource leaks).
			//So return kCriticalErrorCantContinue
			m_last_error = error.what();
			return kCriticalErrorCantContinue;
		}
		catch (std::exception& error) {
			m_last_error = error.what();
			return kError;
		}
		catch (...) {
			m_last_error = "Unknown error";
			return kError;
		}
	}

	// \brief ���������� ���������� ������������ Yolo ��������
	virtual int GetNumberOfDetections()
	{
		return static_cast<int>(m_current_bbox_result.size());
	}

	// \brief ���������� ���������� features ��� ������� �� ��������������� ��������
	virtual int GetNumberOfFeatures()
	{
		return static_cast<int>(m_list_names.size());
	}

	// \brief ���������� �������������� ������ �� ���� ���������� �� �����������
	virtual Detection* GetAllDetections()
	{
		return m_current_bbox_result.empty() ? nullptr : m_current_result.data();
	}

	// \brief ���������� ���������� ������������� �������
	virtual int GetNumberOfClasses()
	{
		return static_cast<int>(m_list_names.size());
	}

	// \brief ���������� ������������ ������ �� ��� �������
	virtual const char* GetNameOfClass(int class_index)
	{
		return class_index < m_net_classes ? m_list_names[class_index].c_str() : nullptr;
	}

	// \brief ���������� ��� ������ ��������� ��������
	virtual const char* GetLastError() const
	{
		return m_last_error.c_str();
	}

	// \brief ����������� ���������� ����������
	virtual void Release()
	{
		delete this;
	}
private:
	std::shared_ptr<Detector> m_detector_from_darknet;
	// ������ m_input_image ������ ���������� (B-G-R) �����������
	// ��� �������� ����������� ������ ������� ���� ����� B, ����� ���� ����� G, ����� ����� R
	// ����������� �������� ��� top-based
	std::vector<float> m_input_image;
	// ������ ������ ����������� �� ������� ��� ������� �� ������,
	// ������� ���������� � ��������� feature map
	std::vector<float> m_probs;
	std::vector<bbox_t> m_current_bbox_result;
	std::vector<Detection> m_current_result;
	std::vector<std::string> m_list_names;
	int m_net_classes;
	// ������ �������� ��������� ��������� �� ������, ���� ��� ����
	std::string m_last_error;
	bool m_weights_loaded = false;
};

// \brief ������� �������� ��������� yolo: ������� ������ IYoloDetector
YOLODLL_API ErrorCode CreateYoloDetectorWithSizes(const char* config_filepath, const char* weight_filepath, const char *name_list_filepath,
								int gpu_id, int width_override, int height_override, const char* const* params, IYoloDetectorOwned*& i_yolo_detector)
{
	try
	{
		YoloDetectorImplementation* result_interface = new YoloDetectorImplementation();
		i_yolo_detector = nullptr;
		auto result = result_interface->Initialize(config_filepath, weight_filepath, name_list_filepath, gpu_id, width_override, height_override);
		i_yolo_detector = result_interface;
		return result;
	}
	catch (...)
	{
		return kError;
	}
}

}//namespace yolo_detector

extern "C" void report_uncontinuable_error_throw(const char* reason, const char* details)
{
	std::string error_string = reason + std::string(" ") + details;
	std::cerr << error_string << std::endl;
	assert(0);
	throw yolo_detector::YoloErrorStateIsUndefinedException(error_string);
}
