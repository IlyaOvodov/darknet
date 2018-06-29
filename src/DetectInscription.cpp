
#include "DetectInscription.h"

#include <vector>
#include <algorithm>
#include <ctype.h>
#include <fstream>

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

std::string out_root = "E:\\iovodov\\Barcodes\\RealImagesExport\\";

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
	detection_with_class* det_ptr = 0;
};


class BarcodesDecoder
{
public:
	BarcodesDecoder(char *datacfg, char *cfgfile1, char *weightfile1, float thresh, int demo_images);
	~BarcodesDecoder();
	void DetectBarcodes(image im_small, image im_full, IplImage* im_demo);
	void ShowImages(int images);
	void SaveImages(int images);
private:
	void FreeSavedImages();
	void ToFile(std::fstream& f, const detection_with_class& det, int w, int h);

	float thresh_ = 0;
	float thresh1_ = 0.1f;
	int demo_images_ = 0;

	network net1_ = {};
	network net2_ = {};
	double h2w_ratio_ = 0;
	char **names_ = 0;
	image **alphabet_ = 0;

	image sized1_ = {0};
	image sized2_ = { 0 };

	std::vector<std::pair<box, DetectionResult>> saved_results_;
	int no_ = 0;
};

BarcodesDecoder::BarcodesDecoder(char *datacfg, char *cfgfile1, char *weightfile1, float thresh, int demo_images)
	: thresh_(thresh), demo_images_(demo_images)
{
	thresh_ = 0.3f; //GVNC

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

BarcodesDecoder::~BarcodesDecoder()
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

void BarcodesDecoder::FreeSavedImages()
{
	free_image(sized1_);
	sized1_.data = 0;
	free_image(sized2_);
	sized2_.data = 0;
}

void BarcodesDecoder::ShowImages(int images)
{
	if (sized1_.data && (images & 2))
		show_image(sized1_, "predictions1");
	if (sized2_.data && (images & 2))
		show_image(sized2_, "predictions2");
}

void BarcodesDecoder::SaveImages(int images)
{
	if (sized1_.data && (images & 2))
		save_image(sized1_, "predictions1");
	if (sized2_.data && (images & 4))
		save_image(sized2_, "predictions2");
}


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
		buf.push_back(std::max(det.det->bbox.w, det.det->bbox.h));
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
			const float k1 = (det2.det->bbox.y - det1.det->bbox.y) / (det2.det->bbox.x - det1.det->bbox.x);
			inliers1.clear();
			outliers1.clear();
			buf.clear();
			for (int i = 0; i < selected_detections_num; ++i)
			{
				const detection_with_class& det = selected_detections[i];
				const float d = DistToLine(det1.det->bbox.x, det1.det->bbox.y, k1, det.det->bbox.x, det.det->bbox.y);
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
				const float d = DistToLine(det1.det->bbox.x, det1.det->bbox.y, k1, det3.det->bbox.x, det3.det->bbox.y);
				const bool first_is_top = d > 0; // т.к. координаты сверху вниз

				if (inliers1.size() < (first_is_top ? best_res.tops.size() : best_res.bottoms.size()))
					continue; // точно плохой вариант

				const size_t left_points_num2 = outliers1.size() - i3 - 1;
				const size_t trials4_num = std::min(kTrialsNum2, left_points_num2);
				for (size_t i4 = 0; i4 < trials4_num; ++i4)
				{
					const size_t ind4 = i3 + 1 + rand() % left_points_num2;
					const detection_with_class& det4 = selected_detections[outliers1[ind4]]; // первую точку берем слева
					const float k2 = (det4.det->bbox.y - det3.det->bbox.y) / (det4.det->bbox.x - det3.det->bbox.x);
					inliers2.clear();
					outliers2.clear();
					buf.clear();
					for (int i = 0; i < outliers1.size(); ++i)
					{
						const detection_with_class& det = selected_detections[outliers1[i]];
						const float d = DistToLine(det3.det->bbox.x, det3.det->bbox.y, k2, det.det->bbox.x, det.det->bbox.y);
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
					if (buf.empty())
						continue;
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
						const float d_top_bottom = DistToLine(det_top.det->bbox.x, det_top.det->bbox.y, k1, det_bottom.det->bbox.x, det_bottom.det->bbox.y);
						for (int i = 0; i < outliers2.size(); ++i)
						{
							const detection_with_class& det = selected_detections[outliers2[i]];
							const float d1 = DistToLine(det_top.det->bbox.x, det_top.det->bbox.y, k1, det.det->bbox.x, det.det->bbox.y);
							bool is_good = true;
							if (d1 < d_top_bottom*0.25) // отклонение в большую сторону не проверяем, т.к. есть вторая проверка ниже
								is_good = false;
							// минус, т.к. расстояние в норме отрицательное
							const float d2 = -DistToLine(det_bottom.det->bbox.x, det_bottom.det->bbox.y, k2, det.det->bbox.x, det.det->bbox.y);
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
				best_res.strings[i] += std::string(names[best_class]);
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

bool CheckResultValidity(const DetectionResult& res)
{
	bool b = res.strings[0].length() == 9
		&& res.strings[1].length() == 3
		&& res.strings[2].length() == 20
		&& res.strings[0].c_str()[2] == '-'
		&& res.strings[0].c_str()[6] == '-'
		&& res.strings[1] == res.strings[0].substr(3, 3)
		&& res.strings[2].c_str()[0] == res.strings[2].c_str()[2]
		&& res.strings[2].c_str()[4] == res.strings[2].c_str()[6]
		&& res.strings[2].c_str()[8] == res.strings[2].c_str()[10]
		&& res.strings[2].c_str()[12] == res.strings[2].c_str()[14]
		&& res.strings[2].c_str()[16] == res.strings[2].c_str()[18]
		&& isdigit(res.strings[0].c_str()[0])
		&& isdigit(res.strings[0].c_str()[1])
		&& isdigit(res.strings[0].c_str()[3])
		&& isdigit(res.strings[0].c_str()[4])
		&& isdigit(res.strings[0].c_str()[5])
		&& isalpha(res.strings[0].c_str()[7])
		&& isdigit(res.strings[0].c_str()[8])
		&& isdigit(res.strings[1].c_str()[0])
		&& isdigit(res.strings[1].c_str()[1])
		&& isdigit(res.strings[1].c_str()[2])
		;

	return b;
}

void DrawDetections(IplImage* im_demo, const DetectionResult& res, int res_index)
{
	bool is_correct = CheckResultValidity(res);
	if (!is_correct)
		return;
	for (int i=0; i<3; ++i)
	{
		int left = Round(im_demo->width * .005);
		int right = Round(im_demo->width * .155);
		//int width = im_demo->height * .006;
		int font_size = Round(im_demo->height * .001f);
		int rect_h = Round(10 + 25 * font_size);
		int base_y = Round(im_demo->width * .005 + rect_h + rect_h * (4*res_index + i));//im_demo->height * .001;

		CvScalar color = cvScalar(224, 224, 224, 0);

		CvPoint pt_text, pt_text_bg1, pt_text_bg2;
		pt_text.x = left;
		pt_text.y = base_y-5;
		pt_text_bg1.x = left;
		pt_text_bg1.y = base_y - rect_h;
		pt_text_bg2.x = right;
		pt_text_bg2.y = base_y;
		//cvRectangle(im_demo, pt_text_bg1, pt_text_bg2, color, width, 8, 0);
		cvRectangle(im_demo, pt_text_bg1, pt_text_bg2, color, CV_FILLED, 8, 0);	// filled

		CvScalar font_color = is_correct ? cvScalar(0,0,0,0) : cvScalar(127, 127, 127, 0);
		CvFont font;
		cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, font_size, font_size, 0, font_size * 3, 8);
		std::string s = res.strings[i];
		if (i == 2 && is_correct)
		{
			int x = left;
			int step = 38 * font_size;
			std::string cs;
			for (int p = 0; p < 5; ++p)
			{
				cs += s.substr(p * 4, 1);
				char ch = s.substr(p * 4, 1).c_str()[0];
				color = ch=='R' ? cvScalar(127, 127, 255, 0)
					: ch == 'G' ? cvScalar(127, 255, 127, 0)
					: ch == 'B' ? cvScalar(255, 127, 127, 0)
					: ch == 'Y' ? cvScalar(127, 255, 255, 0)
					: ch == 'W' ? cvScalar(255, 255, 255, 0)
					: cvScalar(127, 127, 127, 0);

				cvRectangle(im_demo, cvPoint(x, base_y - rect_h), cvPoint(x+step, base_y), color, CV_FILLED, 8, 0);	// filled
				x += step *3/2;
			}
			s = s.substr(1, 1) + s.substr(3, 1)
				+ " " + s.substr(5, 1) + s.substr(7, 1)
				+ " " + s.substr(9, 1) + s.substr(11, 1)
				+ " " + s.substr(13, 1) + s.substr(15, 1)
				+ " " + s.substr(17, 1) + s.substr(19, 1);
				
				//+ "   " + cs;
			cvPutText(im_demo, s.c_str(), pt_text, &font, font_color);
		}
		else
		{
			if (i == 1)
				s = "         " + s;
			cvPutText(im_demo, s.c_str(), pt_text, &font, font_color);
		}
	}
}

void BarcodesDecoder::ToFile(std::fstream& f, const detection_with_class& d, int w, int h)
{
	f << names_[d.best_class] << " " << d.det->bbox.x/w << " " << d.det->bbox.y / h << " " << d.det->bbox.w / w << " " << d.det->bbox.h / h
		<< " " << d.det->extra_features[0] << " " << d.det->extra_features[1] << " " << d.det->extra_features[2] << '\n';
}

void BarcodesDecoder::DetectBarcodes(image im_small, image im_full, IplImage* im_demo)
{
	const float nms1 = 0.1f;
	const float nms2 = 0.2f;
	int letterbox = 0;

	FreeSavedImages();

	image sized_im = letterbox ? letterbox_image(im_small, net1_.w, net1_.h) : resize_image(im_small, net1_.w, net1_.h);

	float *X = sized_im.data;
	network_predict(net1_, X);
	int nboxes = 0;
	detection *dets = get_network_boxes(&net1_, im_small.w, im_small.h, thresh_, 0 /*hier_thresh*/, 0, 1, &nboxes, letterbox);
	do_nms_obj(dets, nboxes, net1_.layers[net1_.n - 1].classes, nms1);
	int selected_detections_num;
	detection_with_class* sdets = get_actual_detections(dets, nboxes, thresh_, &selected_detections_num);

	for (auto it = saved_results_.begin(); it != saved_results_.end(); )
	{
		it->second.det_ptr = 0;
		bool found = false;
		for (int d = 0; d < selected_detections_num; ++d)
		{
			if (box_iou(sdets[d].det->bbox, it->first) > 0.5)
			{
				found = true;
				break;
			}
		}
		if (found)
			++it;
		else
			it = saved_results_.erase(it);
	}

	int res_index = 0;
	for (int idet = 0; idet < selected_detections_num; ++idet) {
		CvRect input_roi = { Round(sdets[idet].det->bbox.x*im_full.w - sdets[idet].det->bbox.w*im_full.w / 2),
			Round(sdets[idet].det->bbox.y*im_full.h - sdets[idet].det->bbox.h*im_full.h / 2), Round(sdets[idet].det->bbox.w*im_full.w), Round(sdets[idet].det->bbox.h*im_full.h) };
		IplImage* input_image = image_to_ipl(im_full);

		// 1я попытка
		IplImage* restored_mat = RestoreImage(input_image, input_roi,
			sdets[idet].det->extra_features[0], sdets[idet].det->extra_features[1], sdets[idet].det->extra_features[2], h2w_ratio_);
		image found1 = ipl_to_image(restored_mat);

		sized1_ = resize_image(found1, net2_.w, net2_.h);
		float *X = sized1_.data;
		network_predict(net2_, X);
		int nboxes1 = 0;
		detection *dets1 = get_network_boxes(&net2_, sized1_.w, sized1_.h, thresh1_, 0, 0, 1, &nboxes1, 0 );
		// Здесь не делаем nms, т.к. перекошенные области могут сильно перекрываться.

		int selected_detections_num1;
		detection_with_class* selected_detections1 = get_actual_detections(dets1, nboxes1, thresh1_, &selected_detections_num1);
		// text output
		qsort(selected_detections1, selected_detections_num1, sizeof(*selected_detections1), compare_by_lefts);
		for (int i = 0; i < selected_detections_num1; ++i)
		{
			box& b = selected_detections1[i].det->bbox;
			b.x *= net2_.w;
			b.y *= net2_.h;
			b.w *= net2_.w;
			b.h *= net2_.h;
		}
		if (demo_images_ & 2)
			draw_detections_v3(sized1_, dets1, nboxes1, thresh1_, names_, alphabet_, net2_.layers[net2_.n - 1].classes, 1);

		DetectionResult res = FindGroupsByRansac(selected_detections1, selected_detections_num1, names_);

		if (res.is_good)
		{
			if (demo_images_ & 2)
				PrintResults(res, selected_detections1, names_);

			// 2я попытка
			const float k = res.tops.size() > res.bottoms.size() ? res.k_top : res.k_bottom;
			float gamma = sdets[idet].det->extra_features[2] + atanf(k) * 2/float(CV_PI); // -> (-1..1)
			gamma = (gamma > 1) ? (gamma - 2) : (gamma < -1 ? (gamma + 2) : gamma);
			IplImage* restored_mat2 = RestoreImage(input_image, input_roi,
				sdets[idet].det->extra_features[0], sdets[idet].det->extra_features[1], gamma, h2w_ratio_);
			image found2 = ipl_to_image(restored_mat2);

			sized2_ = resize_image(found2, net2_.w, net2_.h);
			float *X = sized2_.data;
			network_predict(net2_, X);
			int nboxes2 = 0;
			detection *dets2 = get_network_boxes(&net2_, sized2_.w, sized2_.h, thresh_, 0, 0, 1, &nboxes2, 0);
			do_nms_obj(dets2, nboxes2, net2_.layers[net2_.n - 1].classes, nms2);

			int selected_detections_num2;
			detection_with_class* selected_detections2 = get_actual_detections(dets2, nboxes2, thresh_, &selected_detections_num2);
			// text output
			qsort(selected_detections2, selected_detections_num2, sizeof(*selected_detections2), compare_by_lefts);
			for (int i = 0; i < selected_detections_num2; ++i)
			{
				box& b = selected_detections2[i].det->bbox;
				b.x *= net2_.w;
				b.y *= net2_.h;
				b.w *= net2_.w;
				b.h *= net2_.h;
			}

			DetectionResult res = FindGroupsByRansac(selected_detections2, selected_detections_num2, names_);

			if (demo_images_)
				if (res.is_good)
					PrintResults(res, selected_detections2, names_);
			if (demo_images_ & (4 | 1))
				draw_detections_v3(sized2_, dets2, nboxes2, thresh_, names_, alphabet_, net2_.layers[net2_.n - 1].classes, 1);

			// Если распознали плохо, вытаскаиваем из сохраненных результатов, если хорошо - сохраняем
			if (CheckResultValidity(res))
			{
				for (auto it = saved_results_.begin(); it != saved_results_.end(); )
				{
					if (box_iou(sdets[idet].det->bbox, it->first) > 0.5)
						it = saved_results_.erase(it);
					else
						++it;
				}
				res.det_ptr = &sdets[idet];
				saved_results_.push_back(std::make_pair(sdets[idet].det->bbox, res));

				if (sized2_.data)
				{
					char bbb[100];
					sprintf(bbb, "%s%05d %s %s", out_root.c_str(), no_++, res.strings[0].c_str(), res.strings[2].c_str());
					save_image(sized2_, bbb);
					std::fstream f(std::string(bbb) + ".txt", std::ios::out);
					for (int i : res.tops)
						ToFile(f, selected_detections2[i], net2_.w, net2_.h);
					for (int i : res.middles)
						ToFile(f, selected_detections2[i], net2_.w, net2_.h);
					for (int i : res.bottoms)
						ToFile(f, selected_detections2[i], net2_.w, net2_.h);
				}


			}
			else
			{
				for (auto it = saved_results_.begin(); it != saved_results_.end(); ++it)
				{
					if (box_iou(sdets[idet].det->bbox, it->first) > 0.5)
					{
						res = it->second;
						break;
					}
				}
			}

			free(selected_detections2);
			free_image(found2);
			cvReleaseImage(&restored_mat2);
		}
		else
		{
			// prohibit draw of dets if !res.is_good
			for (int i = 0; i < net1_.layers[net1_.n - 1].classes; ++i)
				sdets[idet].det->prob[i] = 0;
		}

		free(selected_detections1);
		free_image(found1);
		cvReleaseImage(&restored_mat);
		cvReleaseImage(&input_image);
	} //for (int idet = 0

	if (demo_images_ & 1)
	{
		std::sort(saved_results_.begin(), saved_results_.end(),
				[](const std::pair<box, DetectionResult> a, std::pair<box, const DetectionResult> b)->bool
				{
					return a.first.x < b.first.x;
				}
			);

		for (auto& res: saved_results_)
			DrawDetections(im_demo, res.second, res_index++);
		draw_detections_cv_v3(im_demo, dets, nboxes, thresh_, names_, alphabet_, net1_.layers[net1_.n - 1].classes, 1 /*ext_output*/);
	}

	free(sdets);
	free_image(sized_im);
}

BarcodesDecoder* CreateBarcodesDecoder(char *datacfg, char *cfgfile1, char *weightfile1, float thresh, int demo_images)
{
	return new BarcodesDecoder(datacfg, cfgfile1, weightfile1, thresh, demo_images);
}

void ReleaseBarcodesDecoder(BarcodesDecoder* bd)
{
	delete bd;
}

void DetectBarcodes(BarcodesDecoder* bd, image im_small, image im_full, IplImage* im_demo, int show_images, int save_images) {
	bd->DetectBarcodes(im_small, im_full, im_demo);
	//bd->SaveImages(save_images);
	//bd->ShowImages(show_images);
}
