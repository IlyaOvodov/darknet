
#include "DetectInscription.h"

#include <vector>
#include <algorithm>

#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
extern "C" {
#include "utils.h"
#include "parser.h"
#include "option_list.h"
}
#include "box.h"
#include "demo.h"
#include "image.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/core_c.h"
//#include "opencv2/core/core.hpp"
#include "opencv2/core/version.hpp"
#include "opencv2/imgproc/imgproc_c.h"

#include "BarcodeDemo.h"
#endif



class BacodesDecoder
{
public:
	BacodesDecoder(char *datacfg, char *cfgfile1, char *weightfile1, float thresh, bool demo_images);
	~BacodesDecoder();
	void DetectBarcodes(image im);
	void ShowImages();
	void SaveImages();
private:
	void FreeSavedImages();

	float thresh_ = 0;
	bool demo_images_ = false;

	network net1_ = {};
	network net2_ = {};
	double h2w_ratio_ = 0;
	char **names_ = 0;
	image **alphabet_ = 0;

	image sized1_ = {0};
	image sized2_ = { 0 };
};

BacodesDecoder::BacodesDecoder(char *datacfg, char *cfgfile1, char *weightfile1, float thresh, bool demo_images)
	: thresh_(thresh), demo_images_(demo_images)
{
	thresh_ = 0.1f; //GVNC

	list *options = read_data_cfg(datacfg);
	char *cfgfile2 = option_find_str(options, "cfgfile2", "cfg2.cfg");
	char *weightfile2 = option_find_str(options, "weightfile2", "weightfile2.weights");

	char *name_list = option_find_str(options, "names", "barcodes.names");
	names_ = get_labels(name_list);
	alphabet_ = load_alphabet();

	net1_ = parse_network_cfg_custom(cfgfile1, 1); // set batch=1
	load_weights(&net1_, weightfile1);

	net2_ = parse_network_cfg_custom(cfgfile2, 1); // set batch=1
	load_weights(&net2_, weightfile2);
	h2w_ratio_ = static_cast<double>(net2_.h) / net2_.w;

}

BacodesDecoder::~BacodesDecoder()
{
	FreeSavedImages();

	const int nsize = 8;
	for (int j = 0; j < nsize; ++j) {
		for (int i = 32; i < 127; ++i) {
			free_image(alphabet_[j][i]);
		}
		free(alphabet_[j]);
	}
	free(alphabet_);
	free_ptrs(reinterpret_cast<void**>(names_), net2_.layers[net2_.n - 1].classes);
}

void BacodesDecoder::FreeSavedImages()
{
	free_image(sized1_);
	free_image(sized2_);
}

void BacodesDecoder::ShowImages()
{
	if (sized1_.data)
		show_image(sized1_, "predictions1");
	if (sized2_.data)
		show_image(sized2_, "predictions2");
}

void BacodesDecoder::SaveImages()
{
	if (sized1_.data)
		save_image(sized1_, "predictions1");
	if (sized2_.data)
		save_image(sized2_, "predictions2");
}


struct DetectionResult
{
	bool is_good = false;
	float k_top = 0;
	float k_bottom = 0;
	std::vector<int> tops;
	std::vector<int> middles;
	std::vector<int> bottoms;
	std::vector<int> outliers;
	std::string strings[4]; // top, mid, bottom, outliers
};

//template <class T> T sqr(T x) { return x*x; }

// расстояние от (x,y) до прямой (ln_x,ln_y)+kx с учетом знака (<0 для точек над прямой, т.е. y точки < y прямой)
float DistToLine(float ln_x, float ln_y, float ln_k, float x, float y)
{
	const float dx = ln_x - x;
	const float dy = ln_y - y;
	const float d = (ln_k*dx - dy)/sqrt(1+ ln_k*ln_k);
	return d;
}

template <class T> T median_val(std::vector<T> v)
{
	const auto median_it = v.begin() + v.size() / 2;
	std::nth_element(v.begin(), median_it, v.end());
	return *median_it;
}

DetectionResult FindGroupsByRansac(detection_with_class* selected_detections, int selected_detections_num, char **names)
{
	const size_t kTrialsNum1 = 5;
	const size_t kTrialsNum2 = 10;
	const float kSzThr = 0.25;
	const float kSzThr2 = 0.5;
	std::vector<int> inliers1;
	std::vector<int> outliers1;
	std::vector<int> inliers2;
	std::vector<int> outliers2;
	std::vector<int> inliers3;
	std::vector<float> buf;
	DetectionResult best_res;
	srand(123456);

	// отбираем max (n1, n2, n3), равенстве min(max(med_d, 1..3)
	float best_d = std::numeric_limits<float>::max();

	if (selected_detections_num == 0)
		return best_res;
	// median size
	buf.clear();
	for (int i = 0; i < selected_detections_num; ++i)
	{
		const detection_with_class& det = selected_detections[i];
		buf.push_back(std::max(det.det.bbox.w, det.det.bbox.h));
	}
	const float med_size = median_val(buf);

	const size_t trials1_num = std::min<size_t>(kTrialsNum1, selected_detections_num/2);
	for (size_t i1 = 0; i1 < trials1_num; ++i1)
	{
		const detection_with_class& det1 = selected_detections[i1]; // первую точку берем слева
		const size_t left_points_num = selected_detections_num - i1 - 1;
		const size_t trials2_num = std::min(kTrialsNum2, left_points_num);
		for (size_t i2 = 0; i2 < trials2_num; ++i2)
		{
			const size_t ind2 = i1 + 1 + rand() % left_points_num;
			const detection_with_class& det2 = selected_detections[ind2]; // первую точку берем слева
			const float k1 = (det2.det.bbox.y - det1.det.bbox.y) / (det2.det.bbox.x - det1.det.bbox.x);
			inliers1.clear();
			outliers1.clear();
			buf.clear();
			for (int i = 0; i < selected_detections_num; ++i)
			{
				const detection_with_class& det = selected_detections[i];
				const float d = DistToLine(det1.det.bbox.x, det1.det.bbox.y, k1, det.det.bbox.x, det.det.bbox.y);
				if (abs(d) < med_size*kSzThr)
				{
					inliers1.push_back(i);
					buf.push_back(abs(d));
				}
				else if (abs(d) > med_size*kSzThr2)
				{
					outliers1.push_back(i);
				}
			}
			const float med_d1 = median_val(buf);

			// Вторая строка
			const size_t trials3_num = std::min(kTrialsNum1, outliers1.size());
			for (size_t i3 = 0; i3 < trials3_num; ++i3)
			{
				const detection_with_class& det3 = selected_detections[outliers1[i3]]; // первую точку берем слева
				const float d = DistToLine(det1.det.bbox.x, det1.det.bbox.y, k1, det3.det.bbox.x, det3.det.bbox.y);
				const bool first_is_top = d > 0; // т.к. координаты сверху вниз

				if (inliers1.size() < (first_is_top ? best_res.tops.size() : best_res.bottoms.size()))
					continue; // точно плохой вариант

				const size_t left_points_num2 = outliers1.size() - i3 - 1;
				const size_t trials4_num = std::min(kTrialsNum2, left_points_num2);
				for (size_t i4 = 0; i4 < trials4_num; ++i4)
				{
					const size_t ind4 = i3 + 1 + rand() % left_points_num2;
					const detection_with_class& det4 = selected_detections[outliers1[ind4]]; // первую точку берем слева
					const float k2 = (det4.det.bbox.y - det3.det.bbox.y) / (det4.det.bbox.x - det3.det.bbox.x);
					inliers2.clear();
					outliers2.clear();
					buf.clear();
					for (int i = 0; i < outliers1.size(); ++i)
					{
						const detection_with_class& det = selected_detections[outliers1[i]];
						const float d = DistToLine(det3.det.bbox.x, det3.det.bbox.y, k2, det.det.bbox.x, det.det.bbox.y);
						if (abs(d) < med_size*kSzThr)
						{
							inliers2.push_back(outliers1[i]);
							buf.push_back(abs(d));
						}
						else if(abs(d) > med_size*kSzThr2)
						{
							outliers2.push_back(outliers1[i]);
						}
					}
					const float med_d2 = median_val(buf);

					if (inliers2.size() < (first_is_top ? best_res.bottoms.size() : best_res.tops.size()))
						continue; // точно плохой вариант

					// Уже проверено, что size 1 и 2 не меньше лучшего. Проверяем, что результат лучше лучшего
					if (inliers1.size() > (first_is_top ? best_res.tops.size() : best_res.bottoms.size()) ||
						inliers2.size() > (first_is_top ? best_res.bottoms.size() : best_res.tops.size()) ||
						std::max(med_d1, med_d2) > best_d)
					{
						best_res.is_good = true;
						best_res.k_top = first_is_top ? k1 : k2;
						best_res.k_bottom = first_is_top ? k2 : k1;
						best_res.tops = first_is_top ? inliers1 : inliers2;
						best_res.bottoms = first_is_top ? inliers2 : inliers1;
						best_res.middles.clear();
						best_res.outliers.clear();
						best_d = std::max(med_d1, med_d2);

						// уточнить middles
						const detection_with_class& det_top = selected_detections[best_res.tops[0]];
						const detection_with_class& det_bottom = selected_detections[best_res.bottoms[0]];
						const float d_top_bottom = DistToLine(det_top.det.bbox.x, det_top.det.bbox.y, k1, det_bottom.det.bbox.x, det_bottom.det.bbox.y);
						for (int i = 0; i < outliers2.size(); ++i)
						{
							const detection_with_class& det = selected_detections[outliers2[i]];
							const float d1 = DistToLine(det_top.det.bbox.x, det_top.det.bbox.y, k1, det.det.bbox.x, det.det.bbox.y);
							bool is_good = true;
							if (d1 < d_top_bottom*0.25) // отклонение в большую сторону не проверяем, т.к. есть вторая проверка ниже
								is_good = false;
							// минус, т.к. расстояние в норме отрицательное
							const float d2 = -DistToLine(det_bottom.det.bbox.x, det_bottom.det.bbox.y, k2, det.det.bbox.x, det.det.bbox.y);
							if (d2 < d_top_bottom*0.25)
								is_good = false;
							if (is_good)
								best_res.middles.push_back(outliers2[i]);
							else
								best_res.outliers.push_back(outliers2[i]);
						} //for (int i
					} //if (inliers1
				} //for (size_t i4
			} //for (size_t i3
		} //for (size_t i2
	} //for (size_t i1


	// fill names
	if (names)
		for (int i = 0; i < 4; ++i)
		{
			const std::vector<int>& idx_vect = i==0 ? best_res.tops : i==1 ? best_res.middles : i == 2 ? best_res.bottoms : best_res.outliers;
			for (int idx : idx_vect)
			{
				const int best_class = selected_detections[idx].best_class;
				best_res.strings[i] += std::string(names[best_class]) + " ";
			}
		}

	return best_res;
}

template<class T>
int Round(T x) { return static_cast<int>(x + T(0.5)); }

void PrintResults(const DetectionResult& res, detection_with_class* selected_detections, char **names)
{
	printf("Results:\n  ");
	printf("%s\n", res.strings[0].c_str());
	printf("            %s\n", res.strings[1].c_str());
	printf("%s\n", res.strings[2].c_str());
	if (!res.strings[3].empty())
		printf("  outliers: %s\n", res.strings[3].c_str());
	printf("\nk: %4.2f %4.2f \n", res.k_top, res.k_bottom);
}

void BacodesDecoder::DetectBarcodes(image im)
{
	const float nms1 = 0.1f;
	const float nms2 = 0.2f;
	int letterbox = 0;

	FreeSavedImages();

	image sized_im = letterbox ? letterbox_image(im, net1_.w, net1_.h) : resize_image(im, net1_.w, net1_.h);

	float *X = sized_im.data;
	network_predict(net1_, X);
	int nboxes = 0;
	detection *dets = get_network_boxes(&net1_, im.w, im.h, thresh_, 0 /*hier_thresh*/, 0, 1, &nboxes, letterbox);
	do_nms_obj(dets, nboxes, net1_.layers[net1_.n - 1].classes, nms1);

	for (int idet = 0; idet < nboxes; ++idet) {
		bool is_ok = false;
		for (int i_cl = 0; i_cl < dets[idet].classes; ++i_cl)
			if (dets[idet].prob[i_cl] > thresh_ && dets[idet].extra_features_num >= 3)
				is_ok = true;
		if (is_ok) {
			CvRect input_roi = { Round(dets[idet].bbox.x*im.w - dets[idet].bbox.w*im.w / 2),
				Round(dets[idet].bbox.y*im.h - dets[idet].bbox.h*im.h / 2), Round(dets[idet].bbox.w*im.w), Round(dets[idet].bbox.h*im.h) };
			IplImage* input_image = image_to_ipl(im);

			// 1я попытка
			IplImage* restored_mat = RestoreImage(input_image, input_roi,
				dets[idet].extra_features[0], dets[idet].extra_features[1], dets[idet].extra_features[2], h2w_ratio_);
			image found1 = ipl_to_image(restored_mat);

			sized1_ = resize_image(found1, net2_.w, net2_.h);
			float *X = sized1_.data;
			network_predict(net2_, X);
			int nboxes = 0;
			detection *dets1 = get_network_boxes(&net2_, sized1_.w, sized1_.h, thresh_, 0 /*hier_thresh*/, 0, 1, &nboxes, 0 /*letterbox*/ );
			// Здесь не делаем nms, т.к. перекошенные области могут сильно перекрываться.

			int selected_detections_num;
			detection_with_class* selected_detections = get_actual_detections(dets1, nboxes, thresh_, &selected_detections_num);
			// text output
			qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_lefts);
			for (int i = 0; i < selected_detections_num; ++i)
			{
				box& b = selected_detections[i].det.bbox;
				b.x *= net2_.w;
				b.y *= net2_.h;
				b.w *= net2_.w;
				b.h *= net2_.h;
			}
			if (demo_images_)
				draw_detections_v3(sized1_, dets1, nboxes, thresh_, names_, alphabet_, net2_.layers[net2_.n - 1].classes, 1 /*ext_output*/);

			DetectionResult res = FindGroupsByRansac(selected_detections, selected_detections_num, names_);

			if (res.is_good)
			{
				if (demo_images_)
					PrintResults(res, selected_detections, names_);

				// 2я попытка
				const float k = res.tops.size() > res.bottoms.size() ? res.k_top : res.k_bottom;
				float gamma = dets[idet].extra_features[2] + atanf(k) * 2/CV_PI; // -> (-1..1)
				gamma = (gamma > 1) ? (gamma - 2) : (gamma < -1 ? (gamma + 2) : gamma);
				IplImage* restored_mat2 = RestoreImage(input_image, input_roi,
					dets[idet].extra_features[0], dets[idet].extra_features[1], gamma, h2w_ratio_);
				image found2 = ipl_to_image(restored_mat2);

				sized2_ = resize_image(found2, net2_.w, net2_.h);
				float *X = sized2_.data;
				network_predict(net2_, X);
				int nboxes = 0;
				detection *dets2 = get_network_boxes(&net2_, sized2_.w, sized2_.h, thresh_, 0 /*hier_thresh*/, 0, 1, &nboxes, 0 /*letterbox*/);
				do_nms_obj(dets2, nboxes, net2_.layers[net2_.n - 1].classes, nms2);

				int selected_detections_num;
				detection_with_class* selected_detections = get_actual_detections(dets2, nboxes, thresh_, &selected_detections_num);
				// text output
				qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_lefts);
				for (int i = 0; i < selected_detections_num; ++i)
				{
					box& b = selected_detections[i].det.bbox;
					b.x *= net2_.w;
					b.y *= net2_.h;
					b.w *= net2_.w;
					b.h *= net2_.h;
				}

				DetectionResult res = FindGroupsByRansac(selected_detections, selected_detections_num, names_);

				if (demo_images_)
					if (res.is_good)
						PrintResults(res, selected_detections, names_);
				if (demo_images_)
					draw_detections_v3(sized2_, dets2, nboxes, thresh_, names_, alphabet_, net2_.layers[net2_.n - 1].classes, 1 /*ext_output*/);
				free(selected_detections);
				free_image(found2);
				cvReleaseImage(&restored_mat2);
			}
			free(selected_detections);
			free_image(found1);
			cvReleaseImage(&restored_mat);
			cvReleaseImage(&input_image);
		}
	}
	free_image(sized_im);
}


void detect_barcodes(char *datacfg, char *cfgfile1, char *weightfile1, image im, float thresh, int dont_show, int save_images) {
	BacodesDecoder decoder(datacfg, cfgfile1, weightfile1, thresh, !dont_show || save_images);
	decoder.DetectBarcodes(im);
	if (save_images)
		decoder.SaveImages();
	if (!dont_show)
		decoder.ShowImages();
}
