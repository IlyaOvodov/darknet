#pragma once
#include "opencv2/highgui/highgui_c.h"

#ifdef __cplusplus
extern "C" {
#endif

void* ThreadCaptureCreate(CvCapture * acap); //-> TreadCaptureContext*
IplImage* ThreadCaptureGetImage(void* context_ptr); // (TreadCaptureContext* )
int  ThreadCaptureNeedsReset(void* context_ptr);
void ThreadCaptureClear(void* context_ptr);
void ThreadCaptureInit(void* context_ptr, CvCapture * acap);
void ThreadCaptureRelease(void * context_ptr); // (TreadCaptureContext* )

#ifdef __cplusplus
}
#endif
