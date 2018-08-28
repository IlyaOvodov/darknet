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
#include <pthread.h>
#ifdef WIN32
#include <time.h>
#include <winsock.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif
}

double kGrabTimeThreshold = 3.;

namespace {
	double get_wall_time()
	{
		struct timeval time;
		if (gettimeofday(&time, NULL)) {
			return 0;
		}
		return (double)time.tv_sec + (double)time.tv_usec * .000001;
	}
} // namespace

class ThreadReader {
public:
	ThreadReader(CvCapture * cap);
	~ThreadReader();
	IplImage* GetImage();
	void Init(CvCapture * cap);
	void Clear();

	cv::VideoCapture* cpp_cap;
	pthread_t grab_thread;
	pthread_mutex_t in_s_mutex_grab = 0;
	bool grab_ready = false;
	pthread_t retrieve_thread;
	pthread_mutex_t in_s_mutex_retreive = 0;
	cv::Mat frame;
	pthread_mutex_t check_time_mutex_ = 0;
	bool need_restart_ = false;
	double last_grab_time_ = 0;
};

void* ThreadCaptureCreate(CvCapture * cap)
{
	return (void*)(new ThreadReader(cap));
}

IplImage* ThreadCaptureGetImage(void* context_ptr) {
	static ThreadReader* r = reinterpret_cast<ThreadReader*>(context_ptr);
	return r->GetImage();
};

int  ThreadCaptureNeedsReset(void* context_ptr) {
	static ThreadReader* context = reinterpret_cast<ThreadReader*>(context_ptr);
	pthread_mutex_lock(&context->check_time_mutex_);
	double dt = get_wall_time() - context->last_grab_time_;
	pthread_mutex_unlock(&context->check_time_mutex_);
	return context->need_restart_ || dt > kGrabTimeThreshold;
}

void ThreadCaptureClear(void* context_ptr) {
	exit(-1); // GVNC не доделал м€гкую перезагрузку
	static ThreadReader* r = reinterpret_cast<ThreadReader*>(context_ptr);
	r->Clear();
}
void ThreadCaptureInit(void* context_ptr, CvCapture * acap) {
	static ThreadReader* r = reinterpret_cast<ThreadReader*>(context_ptr);
	r->Init(acap);
}

void ThreadCaptureRelease(void* reader) { // (TreadCaptureContext* )
	delete (ThreadReader*)(reader);
}

void *grab_continuous(void *ptr)
{
	ThreadReader* context = reinterpret_cast<ThreadReader*>(ptr);
	while (1) {
		try {
			cvWaitKey(1);
			if (context->need_restart_ || !context->cpp_cap->isOpened())
			{
				std::cout << " grab_continuous FAILED isOpened! \n";
				context->need_restart_ = true;
				continue;
			}
			pthread_mutex_lock(&context->in_s_mutex_grab);
			context->cpp_cap->grab();
			context->grab_ready = true;
			pthread_mutex_unlock(&context->in_s_mutex_grab);
			pthread_mutex_lock(&context->check_time_mutex_);
			context->last_grab_time_ = get_wall_time();
			pthread_mutex_unlock(&context->check_time_mutex_);
		}
		catch (const std::exception& e) {
			std::cout << " grab_continuous ERROR: " << e.what() << "\n";
			context->need_restart_ = true;
		}
		catch (...) {
			std::cout << " grab_continuous ERROR: Unknown error.\n";
			context->need_restart_ = true;
		}
	}
	return 0;
}


void* get_webcam_frame_retrieve_continuous(void *ptr) {
	ThreadReader* context = reinterpret_cast<ThreadReader*>(ptr);
	while (1) {
		try {
			cvWaitKey(1);
			if (!context->need_restart_ && context->cpp_cap->isOpened())
			{
				pthread_mutex_lock(&context->in_s_mutex_grab);
				if (!context->grab_ready) {
					pthread_mutex_unlock(&context->in_s_mutex_grab);
					continue;
				}
				pthread_mutex_lock(&context->in_s_mutex_retreive);
				context->cpp_cap->retrieve(context->frame);
				context->grab_ready = false;
				pthread_mutex_unlock(&context->in_s_mutex_retreive);
				pthread_mutex_unlock(&context->in_s_mutex_grab);
			}
			else {
				std::cout << " get_webcam_frame_retrieve_continuous FAILED isOpened! \n";
				context->need_restart_ = true;
			}
		}
		catch (const std::exception& e) {
			std::cout << " get_webcam_frame_retrieve_continuous ERROR: " << e.what() << "\n";
			context->need_restart_ = true;
		}
		catch (...) {
			std::cout << " get_webcam_frame_retrieve_continuous ERROR: Unknown error.\n";
			context->need_restart_ = true;
		}
	}
	return 0;
}

void ThreadReader::Init(CvCapture * cap)
{
	cpp_cap = (cv::VideoCapture *)(cap);
	pthread_mutex_init(&in_s_mutex_grab, NULL);
	pthread_mutex_init(&in_s_mutex_retreive, NULL);
	pthread_mutex_init(&check_time_mutex_, NULL);

	last_grab_time_ = get_wall_time();

	if (pthread_create(&grab_thread, 0, grab_continuous, (void *)this))
		std::cout << "Thread creation failed";
	if (pthread_create(&retrieve_thread, 0, get_webcam_frame_retrieve_continuous, (void *)this))
		std::cout << "Thread creation failed 2";
	need_restart_ = false;
}

ThreadReader::ThreadReader(CvCapture * cap)
{
	Init(cap);
}

void ThreadReader::Clear()
{
	need_restart_ = true;
	pthread_cancel(grab_thread);
	pthread_join(grab_thread, 0);

	pthread_mutex_unlock(&in_s_mutex_grab);
	pthread_mutex_unlock(&in_s_mutex_retreive);
	pthread_mutex_unlock(&check_time_mutex_);
	pthread_cancel(retrieve_thread);
	// wait for cancellation
	pthread_join(retrieve_thread, 0);

	pthread_mutex_destroy(&in_s_mutex_grab);
	pthread_mutex_destroy(&in_s_mutex_retreive);
	pthread_mutex_destroy(&check_time_mutex_);
	cpp_cap = 0;
}

ThreadReader::~ThreadReader()
{
	Clear();
}

IplImage* ThreadReader::GetImage()
{
	IplImage* res = nullptr;
	if (need_restart_)
		return res;
	pthread_mutex_lock(&in_s_mutex_retreive);
	if (frame.data) {
		IplImage tmp = frame;
		res = cvCloneImage(&tmp);
		frame = cv::Mat();
	}
	pthread_mutex_unlock(&in_s_mutex_retreive);
	return res;
}
