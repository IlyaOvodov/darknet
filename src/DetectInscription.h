#pragma once
#ifndef DetectInscription_H
#define DetectInscription_H

#include "opencv2/core/types_c.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "image.h"

void detect_barcodes(char *datacfg, char *cfgfile1, char *weightfile1, image im, float thresh, int dont_show, int save_images);

#ifdef __cplusplus
}
#endif

#endif // DetectInscription_H