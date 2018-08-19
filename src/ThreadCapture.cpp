#include "ThreadCapture.h"

#include <stdio.h>
#include <string>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/types_c.h"
#include "opencv2/core/version.hpp"

extern "C" {
#define _TIMESPEC_DEFINED
#include <pthread.h>
}

class ThreadReader {
public:
	ThreadReader(CvCapture * cap);
	~ThreadReader();
	IplImage* GetImage();

	cv::VideoCapture &cpp_cap;
	pthread_t grab_thread;
	pthread_mutex_t in_s_mutex_grab = 0;
	bool grab_ready = false;
	pthread_t retrieve_thread;
	pthread_mutex_t in_s_mutex_retreive = 0;
	cv::Mat frame;
	IplImage* img = nullptr;
};

void* CreateTreadCaptureContext(CvCapture * cap)
{
	return (void*)(new ThreadReader(cap));
}

IplImage* GetImage(void* reader) {
	static ThreadReader* r = reinterpret_cast<ThreadReader*>(reader);
	return r->GetImage();
};

void ReleaseTreadCaptureContext(void* reader) { // (TreadCaptureContext* )
	delete (ThreadReader*)(reader);
}


void *grab_continuous(void *ptr)
{
	ThreadReader* context = reinterpret_cast<ThreadReader*>(ptr);
	while (1) {
		cvWaitKey(1);
		if (!context->cpp_cap.isOpened())
			return 0;
		pthread_mutex_lock(&context->in_s_mutex_grab);
		context->cpp_cap.grab();
		context->grab_ready = true;
		pthread_mutex_unlock(&context->in_s_mutex_grab);
	}
	return 0;
}


void* get_webcam_frame_retrieve_continuous(void *ptr) {
	ThreadReader* context = reinterpret_cast<ThreadReader*>(ptr);
	try {
		while (1) {
			cvWaitKey(1);
			if (context->cpp_cap.isOpened())
			{
				pthread_mutex_lock(&context->in_s_mutex_grab);
				if (!context->grab_ready) {
					pthread_mutex_unlock(&context->in_s_mutex_grab);
					continue;
				}
				pthread_mutex_lock(&context->in_s_mutex_retreive);
				context->cpp_cap.retrieve(context->frame);
				context->grab_ready = false;
				pthread_mutex_unlock(&context->in_s_mutex_retreive);
				pthread_mutex_unlock(&context->in_s_mutex_grab);
			}
			else {
				std::cout << " Video-stream stoped &&&&&&&&&&&&&&&&&&! \n";
				return 0;
			}
			//Sleep(1);
		}
	}
	catch (...) {
		std::cout << " Video-stream stoped! \n";
	}
	return 0;
}


ThreadReader::ThreadReader(CvCapture * cap) :
	cpp_cap(*(cv::VideoCapture *)(cap))
{
	pthread_mutex_init(&in_s_mutex_grab, NULL);
	pthread_mutex_init(&in_s_mutex_retreive, NULL);

	if (pthread_create(&grab_thread, 0, grab_continuous, (void *)this))
		std::cout << "Thread creation failed";
	if (pthread_create(&retrieve_thread, 0, get_webcam_frame_retrieve_continuous, (void *)this))
		std::cout << "Thread creation failed 2";

}

ThreadReader::~ThreadReader()
{
	pthread_cancel(grab_thread);
	pthread_cancel(retrieve_thread);

	pthread_mutex_destroy(&in_s_mutex_grab);
	pthread_mutex_destroy(&in_s_mutex_retreive);

	if (img) {
		cvReleaseImage(&img);
		img = NULL;
	}
}

IplImage* ThreadReader::GetImage()
{
	IplImage* res = nullptr;
	pthread_mutex_lock(&in_s_mutex_retreive);
	if (frame.data) {
		IplImage tmp = frame;
		res = cvCloneImage(&tmp);
		frame = cv::Mat();
	}
	pthread_mutex_unlock(&in_s_mutex_retreive);
	return res;
}
