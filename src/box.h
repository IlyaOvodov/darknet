#ifndef BOX_H
#define BOX_H

#ifdef YOLODLL_EXPORTS
#if defined(_MSC_VER)
#define YOLODLL_API __declspec(dllexport) 
#else
#define YOLODLL_API __attribute__((visibility("default")))
#endif
#else
#if defined(_MSC_VER)
#define YOLODLL_API
#else
#define YOLODLL_API
#endif
#endif

typedef struct{
    float x, y, w, h;
} box;

typedef struct{
    float dx, dy, dw, dh;
} dbox;

typedef struct detection {
	box bbox;
	int classes;
	float *prob; // prob only for probable (prob>thresh) classes, otherwise = 0
	float *prob_raw; // prob as detected, for all classes
	float *mask;
	float objectness;
	int sort_class;
} detection;

typedef struct detection_with_class {
	detection det;
	// The most probable class id: the best class index in this->prob.
	// Is filled temporary when processing results, otherwise not initialized
	int best_class;
} detection_with_class;

box float_to_box(float *f);
float box_iou(box a, box b);
float box_rmse(box a, box b);
dbox diou(box a, box b);
void do_nms(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort_v2(box *boxes, float **probs, int total, int classes, float thresh);
YOLODLL_API void do_nms_sort(detection *dets, int total, int classes, float thresh);
YOLODLL_API void do_nms_obj(detection *dets, int total, int classes, float thresh);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

// Creates array of detections with prob > thresh and fills best_class for them
// Return number of selected detections in *selected_detections_num
detection_with_class* get_actual_detections(detection *dets, int dets_num, float thresh, int* selected_detections_num);
// compare to sort detection** by bbox.x
int compare_by_lefts(const void *a_ptr, const void *b_ptr);
// compare to sort detection** by best_class probability 
int compare_by_probs(const void *a_ptr, const void *b_ptr);
#endif
