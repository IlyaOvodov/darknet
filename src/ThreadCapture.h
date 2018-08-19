#pragma once
#include "opencv2/highgui/highgui_c.h"

#ifdef __cplusplus
extern "C" {
#endif

void* CreateTreadCaptureContext(CvCapture * acap); //TreadCaptureContext*
IplImage* GetImage(void* reader); // (TreadCaptureContext* )
void ReleaseTreadCaptureContext(void *); // (TreadCaptureContext* )

#ifdef __cplusplus
}
#endif
