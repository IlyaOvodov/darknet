#include "BarcodeDemo.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
//#include <boost/filesystem.hpp> 
#include "opencv2/core/types_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "AlgoLibrary/al_utilities.h"
//#include "SrsException/srs_exception.h"

//using boost::filesystem::path;
//using AlgoLibrary::Round;

const int kRelativeDistance = 10; // ѕредполагаемое удаление, дл€ перспективы
const float kSizeMargin = 0.4f; // ”величение размера, чтобы гарантировано не отрезалась часть восстановленного изображени€
		// —делано таким (0.4), чтобы картинка походила на ту, на которой распозщнавали символы. –еально достаточно 0.1

class ImageRotator
{
public:
	ImageRotator() {};
	//private:
	/*
	* @brief «аполн€ет матрицу трансформации дл€ cv::warpPerspective()
	* @param [IN] input_image - исходное изображение дл€ поворота
	* @param [IN] alpha, beta, gamma - угол поворота в градусах по оси X, Y, Z соответственно
	* @param [IN] dx, dy, dz - смещение по оси X, Y, Z соответственно
	* @param [IN] dist - рассто€ние в пиксел€х до объекта в 3-мерном пространстве (по оси z)
	* @param [IN] target_scale - приблизительный коэффициент масштабировани€ изображени€ после преобразовани€ (определ€ет фокусное рассто€ние f = target_scale*dist)
	* @param [IN] target_size - размер изображени€, в которое делаем преобразование (чтобы центрировать преобразованную картинку в нем)
	*/
	void FillTransformationMatrix(const cv::Point input_center,
		double alpha, double beta, double gamma,
		double dx, double dy, double dz,
		double dist, double target_scale, cv::Point target_center);

	void FillReverseTransformationMatrix(const cv::Point input_center,
		double alpha, double beta, double gamma,
		double dx, double dy, double dz,
		double dist, double target_scale, cv::Point target_center);

	/*
	* @brief ѕоворачивает входное изображение
	* @param [IN] input_image - исходное изображение дл€ поворота
	* @param [IN] alpha, beta, gamma, dx, dy, dz - см. выше
	*/
	cv::Mat RotateImageBack(const cv::Mat &input_image, const cv::Rect input_roi,
		//const std::vector<cv::Rect>& input_rects,
		//std::vector<cv::Rect>& output_rects,
		//std::vector<cv::Point>& border_points,
		double alpha, double beta, double gamma);

private:
	cv::Mat m_2D_to_3D_matrix;

	//Affine transformations in 3D space
	cv::Mat m_x_axis_rotation_matrix;
	cv::Mat m_y_axis_rotation_matrix;
	cv::Mat m_z_axis_rotation_matrix;

	// Rotation matrix (m_x_axis_rotation_matrix*m_y_axis_rotation_matrix*m_z_axis_rotation_matrix)
	cv::Mat m_rotation_matrix;

	cv::Mat m_translation_matrix;
	cv::Mat m_projection_matrix;
	cv::Mat m_3D_to_2D_matrix;
	cv::Mat m_resulting_transform_matrix;
};

ImageRotator image_rotator;


template<class T>
int Round(T x) { return static_cast<int>(x + T(0.5)); }

void ImageRotator::FillTransformationMatrix(const cv::Point input_center,
	double alpha, double beta, double gamma,
	double dx, double dy, double dz,
	double dist, double target_scale, cv::Point target_center)
{
	alpha = (alpha)*CV_PI / 2.; //-> 0..1 -> 0..Pi/2
	beta = (beta)*CV_PI / 2.;
	gamma = (gamma)*CV_PI / 2.;

	m_2D_to_3D_matrix = (cv::Mat_<double>(4, 3) <<
		1, 0, -input_center.x, // параллельно центруем картинку
		0, 1, -input_center.y,
		0, 0, 0,
		0, 0, 1);

	// Affine transformations in 3D space
	m_x_axis_rotation_matrix = (cv::Mat_<double>(4, 4) <<
		1, 0, 0, 0,
		0, cos(alpha), -sin(alpha), 0,
		0, sin(alpha), cos(alpha), 0,
		0, 0, 0, 1);
	m_y_axis_rotation_matrix = (cv::Mat_<double>(4, 4) <<
		cos(beta), 0, -sin(beta), 0,
		0, 1, 0, 0,
		sin(beta), 0, cos(beta), 0,
		0, 0, 0, 1);
	m_z_axis_rotation_matrix = (cv::Mat_<double>(4, 4) <<
		cos(gamma), -sin(gamma), 0, 0,
		sin(gamma), cos(gamma), 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1);
	// Rotation matrix (m_x_axis_rotation_matrix*m_y_axis_rotation_matrix*m_z_axis_rotation_matrix)
	m_rotation_matrix = m_x_axis_rotation_matrix * m_y_axis_rotation_matrix * m_z_axis_rotation_matrix;

	m_translation_matrix = (cv::Mat_<double>(4, 4) <<
		1, 0, 0, dx,
		0, 1, 0, dy,
		0, 0, 1, dz + dist, //  параллельно отодвигаем на рассто€ние dist
		0, 0, 0, 1);

	const double f = target_scale*dist;
	m_3D_to_2D_matrix = (cv::Mat_<double>(3, 4) <<
		f, 0, target_center.x, 0, // ќдновременно сдвигаем из центра в требуемое положение
		0, f, target_center.y, 0,
		0, 0, 1, 0);

	m_resulting_transform_matrix = m_3D_to_2D_matrix * m_translation_matrix * m_rotation_matrix * m_2D_to_3D_matrix;
}


template<class PointType>
PointType TransformPoint(const cv::Mat& trans_mat, PointType point)
{
	PointType result;
	double denom = trans_mat.at<double>(2, 0)*point.x + trans_mat.at<double>(2, 1)*point.y + trans_mat.at<double>(2, 2);
	result.x = (trans_mat.at<double>(0, 0)*point.x + trans_mat.at<double>(0, 1)*point.y + trans_mat.at<double>(0, 2)) / denom;
	result.y = (trans_mat.at<double>(1, 0)*point.x + trans_mat.at<double>(1, 1)*point.y + trans_mat.at<double>(1, 2)) / denom;
	return result;
}

cv::Mat ImageRotator::RotateImageBack(const cv::Mat &input_image, const cv::Rect input_roi,
	//const std::vector<cv::Rect>& input_rects,
	//std::vector<cv::Rect>& output_rects,
	//std::vector<cv::Point>& border_points,
	double alpha, double beta, double gamma)
{
	int width_margin = Round(input_roi.width * kSizeMargin);
	int height_margin = Round(input_roi.height * kSizeMargin);
	cv::Rect ext_input_roi(input_roi.x - width_margin / 2, input_roi.y - height_margin / 2, input_roi.width + width_margin, input_roi.height + height_margin);
	int original_wh = fabs(gamma) > 0.5 ? ext_input_roi.height : ext_input_roi.width;
	cv::Size original_size(original_wh, original_wh * 192 / 288);
	cv::Point original_center(original_size.width / 2, original_size.height / 2);

	cv::Point transformed_center(ext_input_roi.width / 2, ext_input_roi.height / 2);
	if (ext_input_roi.x < 0) {
		transformed_center.x += ext_input_roi.x;
		ext_input_roi.width += ext_input_roi.x;
		ext_input_roi.x = 0;
	}
	if (ext_input_roi.y < 0) {
		transformed_center.y += ext_input_roi.y;
		ext_input_roi.height += ext_input_roi.y;
		ext_input_roi.y = 0;
	}
	ext_input_roi.width = std::min(ext_input_roi.width, input_image.cols - ext_input_roi.x);
	ext_input_roi.height = std::min(ext_input_roi.height, input_image.rows - ext_input_roi.y);

	cv::Mat output_image;
	double dist = kRelativeDistance*original_size.width; // ѕримерное рассто€ние до этикетки в единицах ее ширины

	// преобразование, создавшее input_image 
	FillTransformationMatrix(original_center, alpha, beta, gamma, 0, 0, 0, dist, 1., transformed_center);
	m_resulting_transform_matrix = m_resulting_transform_matrix.inv();
	cv::warpPerspective(input_image(ext_input_roi), output_image, m_resulting_transform_matrix, original_size, cv::INTER_LANCZOS4);

	//const cv::Rect& whole_rect = input_rects.back();
	//border_points.push_back(TransformPoint(m_resulting_transform_matrix, cv::Point(whole_rect.x, whole_rect.y)));
	//border_points.push_back(TransformPoint(m_resulting_transform_matrix, cv::Point(whole_rect.x + whole_rect.width, whole_rect.y)));
	//border_points.push_back(TransformPoint(m_resulting_transform_matrix, cv::Point(whole_rect.x + whole_rect.width, whole_rect.y + whole_rect.height)));
	//border_points.push_back(TransformPoint(m_resulting_transform_matrix, cv::Point(whole_rect.x, whole_rect.y + whole_rect.height)));

	//output_rects.resize(input_rects.size());
	//for (int i = 0; i < input_rects.size(); ++i) {
	//	cv::Point2d p1 = TransformPoint(m_resulting_transform_matrix, cv::Point2d(input_rects[i].x, input_rects[i].y));
	//	cv::Point2d p2 = TransformPoint(m_resulting_transform_matrix, cv::Point2d(input_rects[i].x + input_rects[i].width + 1, input_rects[i].y));
	//	cv::Point2d p3 = TransformPoint(m_resulting_transform_matrix, cv::Point2d(input_rects[i].x, input_rects[i].y + input_rects[i].height + 1));
	//	cv::Point2d p4 = TransformPoint(m_resulting_transform_matrix, cv::Point2d(input_rects[i].x + input_rects[i].width + 1, input_rects[i].y + input_rects[i].height + 1));
	//	double x_min = std::min(std::min(p1.x, p2.x), std::min(p3.x, p4.x));
	//	double x_max = std::max(std::max(p1.x, p2.x), std::max(p3.x, p4.x));
	//	double y_min = std::min(std::min(p1.y, p2.y), std::min(p3.y, p4.y));
	//	double y_max = std::max(std::max(p1.y, p2.y), std::max(p3.y, p4.y));
	//	output_rects[i].x = Round(x_min);
	//	output_rects[i].y = Round(y_min);
	//	output_rects[i].width = Round(x_max - x_min);
	//	output_rects[i].height = Round(y_max - y_min);
	//}
	return output_image;
}


IplImage* RestoreImage(const CvMat* input_image, const CvRect input_roi,
	double x_ang, double y_ang, double z_ang)
{
	cv::Mat input_image_mat = cv::cvarrToMat(input_image);
	cv::Mat out_img = image_rotator.RotateImageBack(input_image_mat, input_roi, x_ang, y_ang, z_ang);
	IplImage* res = cvCloneImage(&(IplImage)out_img);
	return res;
}


