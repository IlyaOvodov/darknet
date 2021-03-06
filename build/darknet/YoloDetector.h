/// @copyright
/// 2018 ELVEES NeoTek JSC. All rights reserved.
/// Closed source software. Actual software is delivered under the license agreement and (or) non-disclosure
/// agreement. All software is copyrighted by ELVEES NeoTek JSC (Russia) and may not be copied, publicly
/// transmitted, modified or distributed without prior written authorization from the copyright holder.
/// @file 
/// @brief ������� �������������� �������� � ������� Yolo 
/// @details ��������� �� ������������ �� ����������� ������� - Detection* detection - ���������������� 
/// ��� ��������� ������� ������ ����������� �������� DetectObjects

#pragma once
#ifndef YOLODLL_API
#ifdef YOLODLL_EXPORTS
#if defined(_MSC_VER)
#define YOLODLL_API __declspec(dllexport) 
#else
#define YOLODLL_API __attribute__((visibility("default")))
#endif
#else
#if defined(_MSC_VER)
#define YOLODLL_API __declspec(dllimport) 
#else
#define YOLODLL_API
#endif
#endif
#endif

#include <stdint.h>

namespace yolo_detector 
{	
/*! \enum ErrorCode
* ���� ������, ����������� ��� ������������� ������� �������������� �������� � ������� Yolo
*/
enum ErrorCode {
	//�������� ����������� ������, � ���������� �������� ������ ������������� � ������������ ������ (�������� ������ ������, �������� cuda � �.�.)
	//���������� ������������ �������� ����������. ��������������, ��� ���������� ������� ����� ������ ������ � ��� ������� std::quick_exit()
	kCriticalErrorCantContinue = -2,
	kError = -1,
	kOk = 0,
	kWarning = 1
};

/*! \struct Detection
* �������� ������������ Yolo ������� � ��������
*/
struct Detection {
	unsigned int x = 0, y = 0, w = 0, h = 0;	// (x,y) - ���������� �������� ������ ����, (w, h) - ������ � ������ �������
	float prob = 0;					// confidence - ����������� ����, ��� ������ ������ ���������
	unsigned int obj_id = -1;		// ����� ������� �� ��������� [0, ����������_������� + 1]
	const float *feautures = nullptr;			// ����������� �� ������� ��� ������ �������
};

/*! \class IYoloDetector
* �����, ����������� ���������� �������������� �������� � ������� Yolo
*/
class IYoloDetector {
public:
	/**
	* @brief ��������� ������ ������������� � DetectObjects �����������
	* 
	* ���� ����������� ����� ������ ������, ��� �������������� � ���������� ������ DetectObjects(...)
	*/
	virtual int GetInputWidth() = 0;
	virtual int GetInputHeight() = 0;
	/**
	* @brief ��������� ��������� ������������� � DetectObjects �����������
	*
	* ���� ����������� ����� ������ ���������� �������, DetectObjects ����������� � �������
	*/
	virtual int GetInputColorDepth() = 0;
	/**
	* @brief ��������� ����������� �������� �� ������������ � ������� ��������� ���� Yolo
	*
	* ������� ����������� � ������� B-G-R/B-G-R-x ��� 1-��������� � ����������� �� ������������ ���������
	* @param [in] image: ������� ����������� � ������� B-G-R ��� 1-���������, 
	*				    ��� �������������� bpl image ������ ��������� �� ������ � ������ ������ �����������
	* @param [in] width: ������ 
	* @param [in] height: ���  bottom-based (BITMAPINFOHEADER..) ����������� ���������� ���������� height > 0
	*					  ���  top-based (cv::Mat..) ����������� ���������� ���������� height < 0
	*					  ���������� ����� � ��������� ������ top-based
	* @param [in] bpl: bytes per line
	* @param [in] bpp: bytes per pixel 
	* @param [in] threshold: ������������ ������� ������������ � confidence = threshold ��� ����
	* confidence ������������� �� ������� Intersection over Union (IoU), ��������������: [0,1]
	* @param [in] use_mean: �������� ����� ��������� ��������� ������� �� ������
	* @return	kOK, ���� �������������� ������ ��� ������
	*			kError, ���� ��������� ������ 
	*/
	virtual ErrorCode DetectObjects(const uint8_t* image, int width, int height, int bpl, int bpp, float threshold, bool use_mean) = 0;
	// \brief ���������� ���������� ������������ Yolo ��������
	virtual int GetNumberOfDetections() = 0;
	// \brief ���������� ���������� features ��� ������� �� ��������������� ��������
	virtual int GetNumberOfFeatures() = 0;
	// \brief ���������� �������������� ������ �� ���� ���������� �� �����������
	virtual Detection* GetAllDetections() = 0;
	// \brief ���������� ���������� ������������� �������
	virtual int GetNumberOfClasses() = 0;
	// \brief ���������� ������������ ������ �� ��� �������
	virtual const char* GetNameOfClass(int class_index) = 0;
	// \brief ��� ������ ��������� ��������
	virtual const char* GetLastError() const = 0;
protected:
	// \brief ���������, ����� ������ ���� ������� delete, ������ �� ���������� �������� ������� ������
	virtual ~IYoloDetector() {}
	//\brief  ����������� ���������� ����������
	virtual void Release() = 0;
};

// ���������� ��������� �� ��������� IYoloDetector ����� ������ ����� ���� ���������
class IYoloDetectorOwned : public IYoloDetector
{
public:
	using IYoloDetector::Release;
};

/**
* @brief ������� �������� ��������� yolo: ������� ������ IYoloDetector
* @param [in] config_filepath ���������������� ���� � ������� ���� yolo
* @param [in] weight_filepath ���� � ������, ��������������� ������
* (���� ��������� ������ ���������� ����������� �� ����� ���������� ������ ������ ��� nullptr, � ���� ������ ������� DetectObjects �������� �� �����)
* @param [in] name_list_filepath ���� � ������� ������� ��� ������ ������� yolo: coco.names, voc.names, 9k.names
* @param [in] gpu_id ���� ������� ��������� gpu, �� ����� ������� �� ����� �� ��� ��������� 0,1,2,3..
* � ������� ������ ��� ������� yolo ��������� �� gpu ��-��������� ������ 0
* @param [in] width_override ���� >0, �� �������������� ������ �������� ���� ���������, �������� � ������������.
* @param [in] height_override ���� >0, �� �������������� ������ �������� ���� ���������, �������� � ������������.
* @param [in] params ���� �� ������������, �� �������� ����������� � �������
* @param [out] i_yolo_detector ��������� �� ����� IYoloDetector, ������� ���������
*                              ������������ ������ � ������������� yolo �������� � ���������
* @return	kOK, ���� ������ ������ IYoloDetector ��������������� ��� ������
*			kError, ���� ��������� ����������� ������ (������ �������������)
*			kWarning, ���� ��������� ������������� ������
*/
extern "C" YOLODLL_API ErrorCode CreateYoloDetectorWithSizes(const char* config_filepath, const char* weight_filepath, const char *name_list_filepath,
							 int gpu_id, int width_override, int height_override, const char* const* params, IYoloDetectorOwned*& i_yolo_detector);

}//namespace yolo_detctor
