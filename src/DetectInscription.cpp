
#include "DetectInscription.h"

#include <vector>
#include <string>
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
#include "levenstein.hpp"
#endif

using std::string;
using std::vector;

// Hardcoded constants
std::string kOutRoot = "D:\\Programming\\BarcodesDemoDump\\";

const size_t kDictionaryFitThr = 8;

const bool kCheckLastLineValidity = true;

const int kMaxLifetime = 10;
const float kIouThr = 0.5;
const float kThreshold0 = 0.1f; //0.3
const float kThreshold1 = 0.1f;

const float kThreshold2 = 0.1f;

const int kResultDistThr = kCheckLastLineValidity ? 5 : 3;

class CheckDictionary
{
public:
	CheckDictionary() {
		AddStr("01-708-T9 G8G7B9B3G1G3Y9Y1R1R6");
		AddStr("02-821-R9 W1W8G5G3Y9Y4W7W6G1G6");
		AddStr("13-776-W0 G9G9R9R4W3W0G9G1W9W6");
		AddStr("14-073-F1 W6W2Y9Y6B7B5B5B6Y8Y3");
		AddStr("25-681-M6 Y0Y3R1R6Y4Y0R9R1G3G5");
		AddStr("56-330-A2 W7W5B6B1R3R2G4G2R1R7");
		AddStr("85-009-D5 W6W0Y8Y5Y3Y1G5G7B5B7");
		AddStr("87-497-E7 R1R7Y6Y9G0G8B3B0R3R7");
		AddStr("87-939-P7 B4B0R5R6W8W9G4G0R0R8");

		std::ifstream f("ls.txt");
		for (string s; std::getline(f, s); )
		{
			if (strings_.size() > 50)
				break;
			AddStr(s.substr(6, 30));
		}
		
		//size_t d = std::numeric_limits<size_t>::max();
		//for (int i = 0; i < strings_.size(); ++i)
		//	for (int j = i + 1; j < strings_.size(); ++j)
		//	{
		//		size_t di = levenshtein_distance(strings_[i].first, strings_[j].first) + levenshtein_distance(strings_[i].second, strings_[j].second);
		//		if (di < d)
		//			d = di;
		//	}
		//printf("%d", d);
	}

	struct FindRes
	{
		size_t d = std::numeric_limits<size_t>::max();
		size_t next_d = std::numeric_limits<size_t>::max();
		string s0 = "";
		string s2 = "";
	};

	FindRes FindBestFit(const string& s0, const string& s2)
	{
		FindRes best;
		for (const auto& si: strings_)
		{
			size_t di = levenshtein_distance(si.first, s0) + levenshtein_distance(si.second, s2);
			if (di < best.d)
			{
				best.next_d = best.d;
				best.d = di;
				best.s0 = si.first;
				best.s2 = si.second;
			}
			else if (di < best.next_d)
			{
				best.next_d = di;
			}
		}
		return best;
	}

private:
	void AddStr(const string& s) {
		strings_.push_back(
			std::make_pair(s.substr(0, 9), s.substr(10))
		);
	}

	vector<std::pair<string, string>> strings_;
};


struct DetectionResult
{
	bool is_good = false;
	float k_top = 0;
	float k_bottom = 0;
	std::vector<int> tops;
	std::vector<int> middles;
	std::vector<int> bottoms;
	std::vector<int> outliers;
	std::vector<int> classes[4];
	std::vector<float> probs[4];
	//std::vector<float> coords[4];
	std::string strings[4]; // top, mid, bottom, outliers
};

struct AggrDetectionResult
{
	AggrDetectionResult(const DetectionResult& r) :aggr(r), current(r) {}

	DetectionResult aggr;
	DetectionResult current;
	int lifetime = 0;
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
	typedef std::vector<std::pair<box, AggrDetectionResult>> SavedResultsVect;

	SavedResultsVect::iterator FindBestFit(box bbox, const DetectionResult& res);
	void UpdateFit(SavedResultsVect::iterator fit, box bbox, const DetectionResult& res);
	void FreeSavedImages();
	void ToFile(std::fstream& f, const detection_with_class& det, int w, int h);

	float thresh_ = kThreshold0;
	float thresh1_ = kThreshold1;
	float thresh2_ = kThreshold2;
	int demo_images_ = 0;

	network net1_ = {};
	network net2_ = {};
	double h2w_ratio_ = 0;
	char **names_ = 0;
	image **alphabet_ = 0;

	image sized1_ = {0};
	image sized2_ = { 0 };

	std::vector<box> only_bbox_detections_; // –амки, от которых детектировалс€ только bbox

	SavedResultsVect saved_results_;
	int no_ = 0;

	CheckDictionary dict_;
};

BarcodesDecoder::BarcodesDecoder(char *datacfg, char *cfgfile1, char *weightfile1, float thresh, int demo_images)
	: thresh_(thresh), demo_images_(demo_images)
{
	thresh_ = kThreshold0; //GVNC

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

// рассто€ние от (x,y) до пр€мой (ln_x,ln_y)+kx с учетом знака (<0 дл€ точек над пр€мой, т.е. y точки < y пр€мой)
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

			// ¬тора€ строка
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

					// ”же проверено, что size 1 и 2 не меньше лучшего. ѕровер€ем, что результат лучше лучшего
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
							if (d1 < d_top_bottom*0.25) // отклонение в большую сторону не провер€ем, т.к. есть втора€ проверка ниже
								is_good = false;
							// минус, т.к. рассто€ние в норме отрицательное
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
				best_res.classes[i].push_back(best_class);
				best_res.probs[i].push_back(selected_detections[idx].det->prob_raw[best_class]);
				//best_res.coords[i].push_back(selected_detections[idx].det->bbox.x);
				best_res.strings[i] += std::string(names[best_class]);
			}
		}

	return best_res;
}


template <class Vector, class Iterator>
void ReplaceVector(Vector dst, Iterator dst_start, Iterator dst_end, Iterator src_start, Iterator src_end) {
	dst_start = dst.erase(dst_start, dst_end);
	if (dst.empty())
		dst_start = dst.end();
	dst.insert(dst_start, src_start, src_end);
}


DetectionResult ImproveResult(DetectionResult res) // передача не по ссылке, мен€ем
{
	/*
	std::ofstream f("T:\\qwe.txt", std::ios::app);

	char buf[200];
	sprintf(buf, "%22s\t%22s\t%22s\t%22s\n", res.strings[0].c_str(), res.strings[1].c_str(), res.strings[2].c_str(), res.strings[3].c_str());
	f << buf;

	//f << res.strings[0].c_str() << "\n";
	//f << "            " << res.strings[1].c_str() << "\n";
	//f << res.strings[2].c_str() << "\n";
	//if (!res.strings[3].empty())
	//	f << "  outliers: " << res.strings[3].c_str() << "\n" << "\n";
	*/
	int dash1 = -1;
	int dash2 = -1;
	for (int i = 0; i < res.strings[0].length(); ++i) {
		if (res.strings[0][i] == '-') {
			if (dash1 < 0)
				dash1 = i;
			else if (dash2 == 0)
				dash2 = i;
			else
				dash1 = dash2 = -1;
		}
	}
	int str0_mid_len = (dash1 >= 0 && dash2 >= 0) ? (dash2 - dash1 - 1) : -1;
	if (str0_mid_len == 3 && res.strings[1].length() == 3)
	{
		for (int i = 0; i < res.strings[1].length(); ++i) {
			if (res.probs[0][dash1 + 1 + i] > res.probs[1][i]) {
				res.probs[1][i] = res.probs[0][dash1 + 1 + i];
				res.classes[1][i] = res.classes[0][dash1 + 1 + i];
				res.strings[1][i] = res.strings[0][dash1 + 1 + i];
			}
			else {
				res.probs[0][dash1 + 1 + i] = res.probs[1][i];
				res.classes[0][dash1 + 1 + i] = res.classes[1][i];
				res.strings[0][dash1 + 1 + i] = res.strings[1][i];
			}
		}
	}
	else if (str0_mid_len >= 0 && str0_mid_len > res.strings[1].length()) {
		ReplaceVector(res.probs[1], res.probs[1].begin(), res.probs[1].end(), res.probs[0].begin() + dash1 + 1, res.probs[0].begin() + dash2);
		ReplaceVector(res.classes[1], res.classes[1].begin(), res.classes[1].end(), res.classes[0].begin() + dash1 + 1, res.classes[0].begin() + dash2);
		ReplaceVector(res.strings[1], res.strings[1].begin(), res.strings[1].end(), res.strings[0].begin() + dash1 + 1, res.strings[0].begin() + dash2);
	}
	else if (str0_mid_len >= 0 && str0_mid_len < res.strings[1].length()) {
		ReplaceVector(res.probs[0], res.probs[0].begin() + dash1 + 1, res.probs[0].begin() + dash2, res.probs[1].begin(), res.probs[1].end());
		ReplaceVector(res.classes[0], res.classes[0].begin() + dash1 + 1, res.classes[0].begin() + dash2, res.classes[1].begin(), res.classes[1].end());
		ReplaceVector(res.strings[0], res.strings[0].begin() + dash1 + 1, res.strings[0].begin() + dash2, res.strings[1].begin(), res.strings[1].end());
	}

	return res;
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

bool CheckResultValidity(const DetectionResult& res, bool check_last_line)
{
	bool b = res.strings[0].length() == 9
		&& res.strings[1].length() == 3
		&& res.strings[0].c_str()[2] == '-'
		&& res.strings[0].c_str()[6] == '-'
		&& res.strings[1] == res.strings[0].substr(3, 3)
		
		&& ( res.strings[2].length() == 20
		&& res.strings[2].c_str()[0] == res.strings[2].c_str()[2]
		&& res.strings[2].c_str()[4] == res.strings[2].c_str()[6]
		&& res.strings[2].c_str()[8] == res.strings[2].c_str()[10]
		&& res.strings[2].c_str()[12] == res.strings[2].c_str()[14]
		&& res.strings[2].c_str()[16] == res.strings[2].c_str()[18]
		|| !check_last_line)
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

void DrawDetections(IplImage* im_demo, const AggrDetectionResult& res, int res_index)
{
	double draw_k = 2;
	bool is_correct = CheckResultValidity(res.aggr, kCheckLastLineValidity);
	if (!is_correct)
		return;
	for (int i=0; i<3; ++i)
	{
		int left = Round(im_demo->width * .005 * draw_k);
		//int width = im_demo->height * .006;
		int font_size = Round(im_demo->height * .001f * draw_k);
		int rect_h = Round(10 * draw_k + 25 * font_size);
		int base_y = Round(im_demo->width * .005 * draw_k + rect_h + rect_h * (4*res_index + i));//im_demo->height * .001;
		int right = left + Round(10 * draw_k + 25 * font_size * 11);

		CvScalar color = cvScalar(196, 255, 255, 0);

		CvPoint pt_text, pt_text_bg1, pt_text_bg2;
		pt_text.x = left;
		pt_text.y = static_cast<int>(base_y - 5 * draw_k);
		pt_text_bg1.x = left;
		pt_text_bg1.y = base_y - rect_h;
		pt_text_bg2.x = right;
		pt_text_bg2.y = base_y;
		//cvRectangle(im_demo, pt_text_bg1, pt_text_bg2, color, width, 8, 0);
		cvRectangle(im_demo, pt_text_bg1, pt_text_bg2, color, CV_FILLED, 8, 0);	// filled

		CvScalar font_color = is_correct ? cvScalar(0,0,0,0) : cvScalar(127, 127, 127, 0);
		CvFont font;
		cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, font_size, font_size, 0, font_size * 3, 8);
		std::string s = res.aggr.strings[i];
		if (i == 2 && CheckResultValidity(res.aggr, true))
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


double ResultQuality(const DetectionResult& res)
{
	double q = CheckResultValidity(res, kCheckLastLineValidity) ? 1000. : 0;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < res.probs[i].size(); ++j)
			q += res.probs[i][j];
	return q;
}

double ResultLineQuality(const DetectionResult& res, int line, int line_template_lenght)
{
	double q = (res.probs[line].size() == line_template_lenght) ? 1000. : 0;
	for (int j = 0; j < res.probs[line].size(); ++j)
		q += res.probs[line][j];
	return q;
}

double ResultsDistance(const DetectionResult& res1, const DetectionResult& res2)
{
	return static_cast<double>(levenshtein_distance(res1.strings[0], res2.strings[0]) + levenshtein_distance(res1.strings[2], res2.strings[2]));
}

BarcodesDecoder::SavedResultsVect::iterator BarcodesDecoder::FindBestFit(box bbox, const DetectionResult& res)
{
	auto best = saved_results_.end();
	double best_dist = 1000000;
	for (auto it = saved_results_.begin(); it != saved_results_.end(); ++it)
	{
		double dist = ResultsDistance(res, it->second.aggr);
		if (dist < best_dist)
		{
			best_dist = dist;
			best = it;
		}
	}
	if (best_dist < kResultDistThr)
		return best;
	else
		return saved_results_.end();
}

void BarcodesDecoder::UpdateFit(SavedResultsVect::iterator fit, box bbox, const DetectionResult& res)
{
	fit->second.current = res;
	fit->second.lifetime = 0;
	if (ResultLineQuality(res, 0, 9) > ResultLineQuality(fit->second.aggr, 0, 9)) {
		fit->second.aggr.classes[0] = res.classes[0];
		fit->second.aggr.probs[0] = res.probs[0];
		fit->second.aggr.strings[0] = res.strings[0];
	}
	if (ResultLineQuality(res, 1, 3) > ResultLineQuality(fit->second.aggr, 1, 3)) {
		fit->second.aggr.classes[1] = res.classes[1];
		fit->second.aggr.probs[1] = res.probs[1];
		fit->second.aggr.strings[1] = res.strings[1];
	}
	if (ResultLineQuality(res, 2, 10) > ResultLineQuality(fit->second.aggr, 2, 10)) {
		fit->second.aggr.classes[2] = res.classes[2];
		fit->second.aggr.probs[2] = res.probs[2];
		fit->second.aggr.strings[2] = res.strings[2];
	}
}


void UpdateFromDict(const CheckDictionary::FindRes& dict_res, DetectionResult& res)
{
	res.strings[0] = dict_res.s0;
	res.strings[1] = dict_res.s0.substr(3, 3);
	res.strings[2] = dict_res.s2;
	for (int i = 0; i < 3; ++i)
	{
		res.probs[i].clear();
		res.probs[i].resize(res.strings[i].size(), 1.f);
	}
	res.is_good = true;
}


void BarcodesDecoder::DetectBarcodes(image im_small, image im_full, IplImage* im_demo)
{
	const int ext_output = -1;

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

	for (auto it = saved_results_.begin(); it != saved_results_.end(); ++it)
	{
		--it->second.lifetime;
	}

	// ÷икл по результатам детекции самих рамок
	only_bbox_detections_.clear();
	for (int idet = 0; idet < selected_detections_num; ++idet) {
		CvRect input_roi = { Round(sdets[idet].det->bbox.x*im_full.w - sdets[idet].det->bbox.w*im_full.w / 2),
			Round(sdets[idet].det->bbox.y*im_full.h - sdets[idet].det->bbox.h*im_full.h / 2), Round(sdets[idet].det->bbox.w*im_full.w), Round(sdets[idet].det->bbox.h*im_full.h) };
		IplImage* input_image = image_to_ipl(im_full);

		// 1€ попытка
		IplImage* restored_mat = RestoreImage(input_image, input_roi,
			sdets[idet].det->extra_features[0], sdets[idet].det->extra_features[1], sdets[idet].det->extra_features[2], h2w_ratio_);
		image found1 = ipl_to_image(restored_mat);

		sized1_ = resize_image(found1, net2_.w, net2_.h);
		float *X = sized1_.data;
		network_predict(net2_, X);
		int nboxes1 = 0;
		detection *dets1 = get_network_boxes(&net2_, sized1_.w, sized1_.h, thresh1_, 0, 0, 1, &nboxes1, 0 );
		// «десь не делаем nms, т.к. перекошенные области могут сильно перекрыватьс€.

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
			draw_detections_v3(sized1_, dets1, nboxes1, thresh1_, names_, alphabet_, net2_.layers[net2_.n - 1].classes, ext_output);

		DetectionResult res1 = FindGroupsByRansac(selected_detections1, selected_detections_num1, names_);

		auto dict_fit = dict_.FindBestFit(res1.strings[0], res1.strings[2]);
		printf("1111111111111111111111111111111111111 %zd %zd %s %s\n", dict_fit.d, dict_fit.next_d, dict_fit.s0.c_str(), dict_fit.s2.c_str());

		if (res1.is_good) // найдено хоть какое-то вразумительное соответствие по 1й попытке поворота
		{
			if (demo_images_ & 2)
				PrintResults(res1, selected_detections1, names_);

			// 2€ попытка
			const float k = res1.tops.size() > res1.bottoms.size() ? res1.k_top : res1.k_bottom;
			float gamma = sdets[idet].det->extra_features[2] + atanf(k) * 2/float(CV_PI); // -> (-1..1)
			gamma = (gamma > 1) ? (gamma - 2) : (gamma < -1 ? (gamma + 2) : gamma);
			IplImage* restored_mat2 = RestoreImage(input_image, input_roi,
				sdets[idet].det->extra_features[0], sdets[idet].det->extra_features[1], gamma, h2w_ratio_);
			image found2 = ipl_to_image(restored_mat2);

			sized2_ = resize_image(found2, net2_.w, net2_.h);
			float *X = sized2_.data;
			network_predict(net2_, X);
			int nboxes2 = 0;
			detection *dets2 = get_network_boxes(&net2_, sized2_.w, sized2_.h, thresh2_, 0, 0, 1, &nboxes2, 0);
			do_nms_obj(dets2, nboxes2, net2_.layers[net2_.n - 1].classes, nms2);

			int selected_detections_num2;
			detection_with_class* selected_detections2 = get_actual_detections(dets2, nboxes2, thresh2_, &selected_detections_num2);
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

			DetectionResult res2 = FindGroupsByRansac(selected_detections2, selected_detections_num2, names_);
			res2 = ImproveResult(res2);
			auto dict_fit = dict_.FindBestFit(res2.strings[0], res2.strings[2]);
			printf("2222222222222222222222222222222222222 %zd %zd %s %s\n", dict_fit.d, dict_fit.next_d, dict_fit.s0.c_str(), dict_fit.s2.c_str());
			if (dict_fit.d <= kResultDistThr || dict_fit.d + kResultDistThr < dict_fit.next_d)
			{
				UpdateFromDict(dict_fit, res2);
			}

			if (demo_images_)
				if (res2.is_good)
					PrintResults(res2, selected_detections2, names_);
			if (demo_images_ & (4 | 1))
				draw_detections_v3(sized2_, dets2, nboxes2, thresh2_, names_, alphabet_, net2_.layers[net2_.n - 1].classes, ext_output);

			// ≈сли распознали плохо, вытаскаиваем из сохраненных результатов, если хорошо - сохран€ем
			if (res2.is_good) // результат найден и вразумительный
			{
				auto saved_it = FindBestFit(sdets[idet].det->bbox, res2);
				if (saved_it != saved_results_.end())
				{
					UpdateFit(saved_it, sdets[idet].det->bbox, res2);
				}
				else
				{
					AggrDetectionResult r(res2);
					saved_results_.push_back(std::make_pair(sdets[idet].det->bbox, r));
				}

				if (0) { // сохранение картинок дл€ обучени€ч
					if (sized2_.data)
					{
						char bbb[100];
						sprintf(bbb, "%s%05d %s %s", kOutRoot.c_str(), no_++, res2.strings[0].c_str(), res2.strings[2].c_str());
						save_image(sized2_, bbb);
						std::fstream f(std::string(bbb) + ".txt", std::ios::out);
						for (int i : res2.tops)
							ToFile(f, selected_detections2[i], net2_.w, net2_.h);
						for (int i : res2.middles)
							ToFile(f, selected_detections2[i], net2_.w, net2_.h);
						for (int i : res2.bottoms)
							ToFile(f, selected_detections2[i], net2_.w, net2_.h);
					}
				}
			}
			else // результат не соответствует шаблону или вообще кривой
			{
				only_bbox_detections_.push_back(sdets[idet].det->bbox);
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
		}	//  if найдено хоть какое-то вразумительное соответствие по 1й попытке поворота

		free(selected_detections1);
		free_image(found1);
		cvReleaseImage(&restored_mat);
		cvReleaseImage(&input_image);
	} //for (int idet = 0  // ÷икл по результатам детекции самих рамок

	// те рамки, от которых затектировались только положени€,
	// сопоставл€ем с теми имеющимис€, которые ни с чем не сопоставились
	for (auto it = saved_results_.begin(); it != saved_results_.end(); ++it)
	{
		if (it->second.lifetime == 0)
			continue; // уже нашли нормальную пару
		auto best_it_bbox = only_bbox_detections_.end();
		float best_iou = kIouThr;
		for (auto it_bbox = only_bbox_detections_.begin(); it_bbox != only_bbox_detections_.end(); ++it_bbox)
		{
			float iou = box_iou(*it_bbox, it->first);
			if (iou < best_iou)
			{
				best_it_bbox = it_bbox;
				best_iou = iou;
			}
		}
		if (best_it_bbox != only_bbox_detections_.end())
		{
			it->first = *best_it_bbox;
			it->second.lifetime = 0;
			only_bbox_detections_.erase(best_it_bbox);
		}
	}

	// ”дал€ем, что не нашло пары и отжило свой век
	for (auto it = saved_results_.begin(); it != saved_results_.end(); )
	{
		if (it->second.lifetime >= -kMaxLifetime)
			++it;
		else
			it = saved_results_.erase(it);
	}

	if (demo_images_ & 1)
	{
		std::sort(saved_results_.begin(), saved_results_.end(),
				[](const std::pair<box, AggrDetectionResult> a, std::pair<box, const AggrDetectionResult> b)->bool
				{
					return a.first.x < b.first.x;
				}
			);

		int res_index = 0;
		for (auto& res: saved_results_)
			if (CheckResultValidity(res.second.aggr, kCheckLastLineValidity))
				DrawDetections(im_demo, res.second, res_index++);
		draw_detections_cv_v3(im_demo, dets, nboxes, thresh_, names_, alphabet_, net1_.layers[net1_.n - 1].classes, ext_output);
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
