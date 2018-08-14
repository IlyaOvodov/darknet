#include <pthread.h>
#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#ifdef WIN32
#include <time.h>
#include <winsock.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif

#define FRAMES 3

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/version.hpp"

#include "DetectInscription.h"

#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio_c.h"
#endif
#include "http_stream.h"
image get_image_from_stream(CvCapture *cap);


static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network net;
static image in_s ;
pthread_mutex_t in_s_mutex;
static image det_s;
static CvCapture * cap;
static int cpp_video_capture = 0;
static float fps = 0;
static float demo_thresh = 0;
static int demo_ext_output = 0;

static float *predictions[FRAMES];
static int demo_index = 0;
static image images[FRAMES];
static IplImage* ipl_images[FRAMES];
static float *avg;

static BarcodesDecoder* bd = 0;

void draw_detections_cv(IplImage* show_img, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes);
void draw_detections_cv_v3(IplImage* show_img, detection *dets, int num, float thresh, char **names, image **alphabet, int classes, int ext_output);
void show_image_cv_ipl(IplImage *disp, const char *name);
image get_image_from_stream_resize(CvCapture *cap, int w, int h, int c, IplImage** in_img, int cpp_video_capture, int dont_close);
IplImage* in_img;
IplImage* det_img;
IplImage* show_img;

static int flag_exit;
static double read_time;

double get_wall_time()
{
	struct timeval time;
	if (gettimeofday(&time, NULL)) {
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void *fetch_in_thread(void *ptr)
{
    //in = get_image_from_stream(cap);
	int dont_close_stream = 1;	// set 1 if your IP-camera periodically turns off and turns on video-stream
	in_s = get_image_from_stream_resize(cap, net.w, net.h, net.c, &in_img, cpp_video_capture, dont_close_stream);
	pthread_mutex_lock(&in_s_mutex);
	if (!in_s.data) {
        //error("Stream closed.");
		printf("Stream closed.\n");
		flag_exit = 1;
		return EXIT_FAILURE;
    }
    //in_s = resize_image(in, net.w, net.h);
	pthread_mutex_unlock(&in_s_mutex);

    return 0;
}

void *fetch_in_thread_continuous(void *ptr)
{
	//in = get_image_from_stream(cap);
	int dont_close_stream = 1;	// set 1 if your IP-camera periodically turns off and turns on video-stream
	while (1) {
		image in_s_tmp;
		IplImage* in_img_tmp;
		double t1 = get_wall_time();
		in_s_tmp = get_image_from_stream_resize(cap, net.w, net.h, net.c, &in_img_tmp, cpp_video_capture, dont_close_stream);
		//if (!in_s_tmp.data) {
		//	//error("Stream closed.");
		//	printf("Stream closed.\n");
		//	flag_exit = 1;
		//	return EXIT_FAILURE;
		//}
		pthread_mutex_lock(&in_s_mutex);
		if (in_s_tmp.data) {
			if (in_s.data){
				free_image(in_s); // drop it, because previous on is not processed yet
				cvReleaseImage(&in_img);
			}
			// last image taken
			in_s = in_s_tmp;
			in_img = in_img_tmp;
			//in_s = resize_image(in, net.w, net.h);
			double t2 = get_wall_time();
			read_time = t2 - t1;
		}
		pthread_mutex_unlock(&in_s_mutex);
	}
	return 0;
}

void *detect_in_thread(void *ptr)
{
    float nms = .45;	// 0.4F

    layer l = net.layers[net.n-1];
    float *X = det_s.data;
    float *prediction = network_predict(net, X);

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, FRAMES, l.outputs, avg);
    l.output = avg;

    free_image(det_s);
	/*
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, 1, 1, demo_thresh, probs, 0, boxes, 0);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);
	*/
	int letter = 0;
	int nboxes = 0;
	detection *dets = get_network_boxes(&net, det_s.w, det_s.h, demo_thresh, demo_thresh, 0, 1, &nboxes, letter);
	//if (nms) do_nms_obj(dets, nboxes, l.classes, nms);	// bad results
	if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
	

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");

    //images[demo_index] = det;
    //det = images[(demo_index + FRAMES/2 + 1)%FRAMES];
	ipl_images[demo_index] = det_img;
	det_img = ipl_images[(demo_index + FRAMES / 2 + 1) % FRAMES];
    demo_index = (demo_index + 1)%FRAMES;
	    
	//draw_detections(det, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
	draw_detections_cv_v3(det_img, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes, demo_ext_output);
	//draw_detections_cv(det_img, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
	free_detections(dets, nboxes);

	return 0;
}


void *detect_in_thread_bar(void *ptr)
{
	if (!det_s.data)
		return;

	ipl_images[demo_index] = det_img;
	IplImage* det_img_to_draw = ipl_images[(demo_index + FRAMES / 2 + 1) % FRAMES];
	demo_index = (demo_index + 1) % FRAMES;

	if (det_img_to_draw) {
		//image det_img_im = ipl_to_image(det_img);
		//if (det_img_im.c > 1)
		//	rgbgr_image(det_img_im);

		//DetectBarcodes(bd, det_s, det_img_im, det_img_to_draw, 0 /*show_images*/, 0 /*save_images*/);

		//free_image(det_img_im);
	}

	printf("\033[2J");
	printf("\033[1;1H");
	printf("\nFPS:%.1f\n", fps);
	pthread_mutex_lock(&in_s_mutex);
	double read_time2 = read_time;
	pthread_mutex_unlock(&in_s_mutex);
	printf("\nt: %.4f\n", read_time2);
	printf("Objects:\n\n");

	det_img = det_img_to_draw;

	free_image(det_s);
	det_s.data = 0;
	return 0;
}


void demo(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, char **names, int classes,
	int frame_skip, char *prefix, char *out_filename, int http_stream_port, int dont_show, int ext_output)
{
    //skip = frame_skip;
    image **alphabet = load_alphabet();
    int delay = frame_skip;
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
	demo_ext_output = ext_output;
    printf("Demo\n");
    net = parse_network_cfg_custom(cfgfile, 1);	// set batch=1
    if(weightfile){
        load_weights(&net, weightfile);
    }
    //set_batch_network(&net, 1);
	fuse_conv_batchnorm(net);
    srand(2222222);

    if(filename){
        printf("video file: %s\n", filename);
//#ifdef CV_VERSION_EPOCH	// OpenCV 2.x
//		cap = cvCaptureFromFile(filename);
//#else					// OpenCV 3.x
		cpp_video_capture = 1;
		cap = get_capture_video_stream(filename);
//#endif
    }else{
		printf("Webcam index: %d\n", cam_index);
//#ifdef CV_VERSION_EPOCH	// OpenCV 2.x
//        cap = cvCaptureFromCAM(cam_index);
//#else					// OpenCV 3.x
		cpp_video_capture = 1;
		cap = get_capture_webcam(cam_index);
//#endif
    }

	if (!cap) {
#ifdef WIN32
		printf("Check that you have copied file opencv_ffmpeg340_64.dll to the same directory where is darknet.exe \n");
#endif
		error("Couldn't connect to webcam.\n");
	}

    layer l = net.layers[net.n-1];
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) images[j] = make_image(1,1,3);

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

	flag_exit = 0;

    pthread_t fetch_thread;
    pthread_t detect_thread;

    fetch_in_thread(0);
	det_img = in_img;
    det_s = in_s;

    fetch_in_thread(0);
    detect_in_thread(0);
	det_img = in_img;
    det_s = in_s;

    for(j = 0; j < FRAMES/2; ++j){
        fetch_in_thread(0);
        detect_in_thread(0);
		det_img = in_img;
        det_s = in_s;
    }

    int count = 0;
    if(!prefix && !dont_show){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        cvMoveWindow("Demo", 0, 0);
        cvResizeWindow("Demo", 1352, 1013);
    }

	CvVideoWriter* output_video_writer = NULL;    // cv::VideoWriter output_video;
	if (out_filename && !flag_exit)
	{
		CvSize size;
		size.width = det_img->width, size.height = det_img->height;

		//const char* output_name = "test_dnn_out.avi";
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('H', '2', '6', '4'), 25, size, 1);
		output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('D', 'I', 'V', 'X'), 25, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('M', 'J', 'P', 'G'), 25, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('M', 'P', '4', 'V'), 25, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('M', 'P', '4', '2'), 25, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('X', 'V', 'I', 'D'), 25, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('W', 'M', 'V', '2'), 25, size, 1);
	}

    double before = get_wall_time();

    while(1){
        ++count;
        if(1){
            if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
            if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

            if(!prefix){
				if (!dont_show) {
					show_image_cv_ipl(show_img, "Demo");
					int c = cvWaitKey(1);
					if (c == 10) {
						if (frame_skip == 0) frame_skip = 60;
						else if (frame_skip == 4) frame_skip = 0;
						else if (frame_skip == 60) frame_skip = 4;
						else frame_skip = 0;
					}
				}
            }else{
                char buff[256];
                sprintf(buff, "%s_%08d.jpg", prefix, count);
				cvSaveImage(buff, show_img, 0);
                //save_image(disp, buff);
            }

			// if you run it with param -http_port 8090  then open URL in your web-browser: http://localhost:8090
			if (http_stream_port > 0 && show_img) {
				//int port = 8090;
				int port = http_stream_port;
				int timeout = 200;
				int jpeg_quality = 30;	// 1 - 100
				send_mjpeg(show_img, port, timeout, jpeg_quality);
			}

			// save video file
			if (output_video_writer && show_img) {
				cvWriteFrame(output_video_writer, show_img);
				printf("\n cvWriteFrame \n");
			}

			cvReleaseImage(&show_img);

            pthread_join(fetch_thread, 0);
            pthread_join(detect_thread, 0);

			if (flag_exit == 1) break;

            if(delay == 0){
				show_img = det_img;
            }
			det_img = in_img;
            det_s = in_s;
        }else {
            fetch_in_thread(0);
			det_img = in_img;
            det_s = in_s;
            detect_in_thread(0);

			show_img = det_img;
			if (!dont_show) {
				show_image_cv_ipl(show_img, "Demo");
				cvWaitKey(1);
			}
			cvReleaseImage(&show_img);
        }
        --delay;
        if(delay < 0){
            delay = frame_skip;

            double after = get_wall_time();
            float curr = 1./(after - before);
            fps = curr;
            before = after;
        }
    }
	printf("input video stream closed. \n");
	if (output_video_writer) {
		cvReleaseVideoWriter(&output_video_writer);
		printf("output_video_writer closed. \n");
	}

	// free memory
	cvReleaseImage(&show_img);
	cvReleaseImage(&in_img);
	free_image(in_s);

	free(avg);
	for (j = 0; j < FRAMES; ++j) free(predictions[j]);
	for (j = 0; j < FRAMES; ++j) free_image(images[j]);

	for (j = 0; j < l.w*l.h*l.n; ++j) free(probs[j]);
	free(boxes);
	free(probs);

	free_ptrs(names, net.layers[net.n - 1].classes);

	int i;
	const int nsize = 8;
	for (j = 0; j < nsize; ++j) {
		for (i = 32; i < 127; ++i) {
			free_image(alphabet[j][i]);
		}
		free(alphabet[j]);
	}
	free(alphabet);

	free_network(net);
}

void barcodes_detector123(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, int dont_show)
{
	image im = load_image(filename, 0, 0, 3);

//	BarcodesDecoder* bd = CreateBarcodesDecoder(datacfg, cfgfile, weightfile, thresh, dont_show, 1 /*save_images*/);
//	DetectBarcodes(bd, im, dont_show, 1 /*save_images*/);
//	ReleaseBarcodesDecoder(bd);

	free_image(im);
}



void demobar(char *datacfg, char *cfgfile, char *weightfile, const char *filename, float thresh, int cam_index,
	int frame_skip, char *prefix, char *out_filename, int http_stream_port, int dont_show)
{
	pthread_mutex_init(&in_s_mutex, NULL);
	//GVNC bd = CreateBarcodesDecoder(datacfg, cfgfile, weightfile, thresh, 1 /*demo_images*/);

	int delay = frame_skip;

	net = parse_network_cfg_custom(cfgfile, 1);	// net нужна только для размеров в fetch_in_thread

	srand(2222222);

	if (filename) {
		printf("video file: %s\n", filename);
		//#ifdef CV_VERSION_EPOCH	// OpenCV 2.x
		//		cap = cvCaptureFromFile(filename);
		//#else					// OpenCV 3.x
		cpp_video_capture = 1;
		cap = get_capture_video_stream(filename);
		//#endif
	}
	else {
		printf("Webcam index: %d\n", cam_index);
		//#ifdef CV_VERSION_EPOCH	// OpenCV 2.x
		//        cap = cvCaptureFromCAM(cam_index);
		//#else					// OpenCV 3.x
		cpp_video_capture = 1;
		cap = get_capture_webcam(cam_index);
		//#endif
	}

	if (!cap) {
#ifdef WIN32
		printf("Check that you have copied file opencv_ffmpeg340_64.dll to the same directory where is darknet.exe \n");
#endif
		error("Couldn't connect to webcam.\n");
	}

	int j;

	for (j = 0; j < FRAMES; ++j) images[j] = make_image(1, 1, 3);

	flag_exit = 0;

	pthread_t fetch_thread;
	pthread_t detect_thread;

	fetch_in_thread(0);
	det_img = in_img;
	det_s = in_s;

	fetch_in_thread(0);
	detect_in_thread_bar(0);
	det_img = in_img;
	det_s = in_s;

	for (j = 0; j < FRAMES / 2; ++j) {
		fetch_in_thread(0);
		detect_in_thread_bar(0);
		det_img = in_img;
		det_s = in_s;
	}

	int count = 0;
	if (!prefix && !dont_show) {
		cvNamedWindow("Demo", CV_WINDOW_NORMAL);
		cvMoveWindow("Demo", 0, 0);
		cvResizeWindow("Demo", 600, 400);

		cvNamedWindow("Demo2", CV_WINDOW_NORMAL);
		cvMoveWindow("Demo2", 100, 100);
		cvResizeWindow("Demo2", 600, 400);
	}

	CvVideoWriter* output_video_writer = NULL;    // cv::VideoWriter output_video;
	if (out_filename && !flag_exit)
	{
		CvSize size;
		size.width = det_img->width, size.height = det_img->height;

		//const char* output_name = "test_dnn_out.avi";
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('H', '2', '6', '4'), 25, size, 1);
		output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('D', 'I', 'V', 'X'), 25, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('M', 'J', 'P', 'G'), 25, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('M', 'P', '4', 'V'), 25, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('M', 'P', '4', '2'), 25, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('X', 'V', 'I', 'D'), 25, size, 1);
		//output_video_writer = cvCreateVideoWriter(out_filename, CV_FOURCC('W', 'M', 'V', '2'), 25, size, 1);
	}

	double before = get_wall_time();

	const int read_in_thread = 1;
	const int continuous_read = 1;
	if (read_in_thread && continuous_read) {
		in_s.data = 0; // to enable continuous read
		in_img = 0;
		if (pthread_create(&fetch_thread, 0, fetch_in_thread_continuous, 0)) error("Thread creation failed");
	}
	while (1) {
		++count;
		if (read_in_thread) {
			if (!continuous_read)
				if (pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
			if (pthread_create(&detect_thread, 0, detect_in_thread_bar, 0)) error("Thread creation failed");

			if (!prefix) {
				if (!dont_show) {
					if (show_img)
						show_image_cv_ipl(show_img, "Demo");
					int c = cvWaitKey(1);
					if (c == 10) {
						if (frame_skip == 0) frame_skip = 60;
						else if (frame_skip == 4) frame_skip = 0;
						else if (frame_skip == 60) frame_skip = 4;
						else frame_skip = 0;
					}
				}
			}
			else {
				char buff[256];
				sprintf(buff, "%s_%08d.jpg", prefix, count);
				if (show_img)
					cvSaveImage(buff, show_img, 0);
				//save_image(disp, buff);
			}

			// if you run it with param -http_port 8090  then open URL in your web-browser: http://localhost:8090
			if (http_stream_port > 0 && show_img) {
				//int port = 8090;
				int port = http_stream_port;
				int timeout = 200;
				int jpeg_quality = 30;	// 1 - 100
				send_mjpeg(show_img, port, timeout, jpeg_quality);
			}

			// save video file
			if (output_video_writer && show_img) {
				cvWriteFrame(output_video_writer, show_img);
				printf("\n cvWriteFrame \n");
			}

			if (show_img) {
				cvReleaseImage(&show_img);
				show_img = 0;
			}

			if(!continuous_read)
				pthread_join(fetch_thread, 0);
			pthread_join(detect_thread, 0);

			if (flag_exit == 1) break;

			if (delay == 0 && det_img) {
				show_img = det_img;
				det_img = 0;
			}
			pthread_mutex_lock(&in_s_mutex);
			if (in_s.data) { // successfully read
				det_img = in_img;
				det_s = in_s;
				in_s.data = 0;
				in_img = 0;
				if (!dont_show)
					show_image_cv_ipl(det_img, "Demo2");
			}
			pthread_mutex_unlock(&in_s_mutex);
		}
		else {
			fetch_in_thread(0);
			det_img = in_img;
			det_s = in_s;
			detect_in_thread_bar(0);

			show_img = det_img;
			if (!dont_show) {
				show_image_cv_ipl(show_img, "Demo");
				cvWaitKey(1);
			}
			cvReleaseImage(&show_img);
		}
		--delay;
		if (delay < 0) {
			delay = frame_skip;

			double after = get_wall_time();
			float curr = 1. / (after - before);
			fps = curr;
			before = after;
		}
	}
	printf("input video stream closed. \n");
	if (output_video_writer) {
		cvReleaseVideoWriter(&output_video_writer);
		printf("output_video_writer closed. \n");
	}

	// free memory
	pthread_mutex_destroy(&in_s_mutex);
	cvReleaseImage(&show_img);
	cvReleaseImage(&in_img);
	free_image(in_s);

	for (j = 0; j < FRAMES; ++j) free_image(images[j]);

	free_network(net);
	ReleaseBarcodesDecoder(bd);
}

#else
void demo(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, char **names, int classes,
	int frame_skip, char *prefix, char *out_filename, int http_stream_port, int dont_show, int ext_output)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
void demobar(char *datacfg, char *cfgfile, char *weightfile, const char *filename, float thresh, int cam_index,
	int frame_skip, char *prefix, char *out_filename, int http_stream_port, int dont_show);
{
	fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

