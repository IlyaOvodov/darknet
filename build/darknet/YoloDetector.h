/// @copyright
/// 2018 ELVEES NeoTek JSC. All rights reserved.
/// Closed source software. Actual software is delivered under the license agreement and (or) non-disclosure
/// agreement. All software is copyrighted by ELVEES NeoTek JSC (Russia) and may not be copied, publicly
/// transmitted, modified or distributed without prior written authorization from the copyright holder.
/// @file 
/// @brief Функции детектирования объектов с помощью Yolo 
/// @details Указатель на обнаруженные на изображении объекты - Detection* detection - перезаписывается 
/// при обработке каждого нового изображения функцией DetectObjects

#pragma once

#include <stdint.h>

namespace yolo_detector 
{	
/*! \enum ErrorCode
* Коды ошибок, возникающие при использовании функций детектирования объектов с помощью Yolo
*/
enum ErrorCode {
	kError = -1,
	kOk = 0,
	kWarning = 1
};

/*! \struct Detection
* Описание обнаруженной Yolo рамочки с объектом
*/
struct Detection {
	unsigned int x = 0, y = 0, w = 0, h = 0;	// (x,y) - координаты верхнего левого угла, (w, h) - ширина и высота рамочки
	float prob = 0;					// confidence - вероятность того, что объект найден правильно
	unsigned int obj_id = -1;		// класс объекта из диапазона [0, количество_классов + 1]
	const float *feautures = nullptr;			// вероятности по классам для каждой рамочки
};

/*! \class IYoloDetector
* Класс, описывающий результаты детектирования объектов с помощью Yolo
*/
class IYoloDetector {
public:
	/**
	* @brief Выполняет обнаружение объектов на изображеннии с помощью нейронной сети Yolo
	*
	* Ожидает изображения в формате B-G-R
	* @param [in] image: ожидает изображения в формате B-G-R, 
	*				    для положительного bpl image должен указывать на первую в памяти строку изображения
	* @param [in] width: ширина 
	* @param [in] height: Для  bottom-based (BITMAPINFOHEADER..) изображений необходимо передавать height > 0
	*					  Для  top-based (cv::Mat..) изображений необходимо передавать height < 0
	*					  Координаты рамок с объектами всегда top-based
	* @param [in] bpl: bytes per line
	* @param [in] bpp: bytes per pixel 
	* @param [in] threshold: обнаруженные объекты отображаются с confidence = threshold или выше
	* confidence расчитывается по метрике Intersection over Union (IoU), соответственно: [0,1]
	* @param [in] use_mean: включить режим усрднения выходного тензора по кадрам
	* @return	kOK, если детектирование прошло без ошибок
	*			kError, если произошла ошибка 
	*/
	virtual ErrorCode DetectObjects(const uint8_t* image, int width, int height, int bpl, int bpp, float threshold, bool use_mean) = 0;
	// \brief Возвращает количество обнаруженных Yolo объектов
	virtual int GetNumberOfDetections() = 0;
	// \brief Возвращает количество features для каждого из детектированных объектов
	virtual int GetNumberOfFeatures() = 0;
	// \brief Возвращает результирующий вектор со всем детекциями на изображении
	virtual Detection* GetAllDetections() = 0;
	// \brief Возвращает количество детектируемых классов
	virtual int GetNumberOfClasses() = 0;
	// \brief Возвращает наименование класса по его индексу
	virtual const char* GetNameOfClass(int class_index) = 0;
	// \brief Код ошибки последней операции
	virtual const char* GetLastError() const = 0;
protected:
	// \brief Добавлено, чтобы нельзя было вызвать delete, защита от случайного удаления объекта класса
	virtual ~IYoloDetector() {}
	//\brief  Освобождает реализацию интерфейса
	virtual void Release() = 0;
};

// Освободить указатель на интерфейс IYoloDetector можно только через этот интерфейс
class IYoloDetectorOwned : public IYoloDetector
{
public:
	using IYoloDetector::Release;
};

/**
* @brief Функция создания детектора yolo: объекта класса IYoloDetector
* @param [in] config_filepath конфигурационный файл с моделью сети yolo
* @param [in] weight_filepath файл с весами, соответствующий модели
* @param [in] name_list_filepath файл с именами классов для разных моделей yolo: coco.names, voc.names, 9k.names
* @param [in] gpu_id если имеется несколько gpu, то можно указать на каком из них запускать 0,1,2,3..
* в обычном случае для запуска yolo детектора на gpu по-умолчанию ставят 0
* @param [in] params пока не используется, но возможно понадобится в будущем
* @param [out] i_yolo_detector указатель на класс IYoloDetector, который позволяет
*                              осуществлять доступ к обнаруженными yolo рамочкам с объектами
* @return	kOK, если объект класса IYoloDetector инициализирован без ошибок
*			kError, если произошла критическая ошибка (ошибка инициализации)
*			kWarning, если произошла некритическая ошибка
*/
ErrorCode CreateYoloDetector(const char* config_filepath, const char* weight_filepath, const char *name_list_filepath,
							 int gpu_id, const char* const* params, IYoloDetectorOwned*& i_yolo_detector);

}//namespace yolo_detctor