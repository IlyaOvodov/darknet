#pragma once
#ifndef DetectInscription_H
#define DetectInscription_H

#include "opencv2/core/types_c.h"


#ifdef __cplusplus
class BarcodesDecoder;
extern "C" {
#else
#define BarcodesDecoder void
#endif

#include "image.h"

BarcodesDecoder* CreateBarcodesDecoder(char *datacfg, char *cfgfile1, char *weightfile1, float thresh, int demo_images);

void ReleaseBarcodesDecoder(BarcodesDecoder* bd);

void DetectBarcodes(BarcodesDecoder* bd, image im_small, image im_full, IplImage* im_demo, int show_images, int save_images);

#ifdef __cplusplus
}
#endif

#endif // DetectInscription_H