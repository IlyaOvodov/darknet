#pragma once
#ifndef BarcodeDemo_H
#define BarcodeDemo_H

#include "opencv2/core/types_c.h"

#ifdef __cplusplus
extern "C" {
#endif

	IplImage* RestoreImage(const IplImage* input_image, const CvRect input_roi,
		double x_ang, double y_ang, double z_ang, double h2w_ratio);


#ifdef __cplusplus
}
#endif

#endif // BarcodeDemo_H