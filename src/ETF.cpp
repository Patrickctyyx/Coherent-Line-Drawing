#include "ETF.h"
# define M_PI 3.14159265358979323846

using namespace cv;


ETF::ETF() {
	Size s(300, 300);

	Init(s);
}

ETF::ETF(Size s) {
	Init(s);
}

void ETF::Init(Size s) {
	// CV_32FC3 是 Mat 的一种类型
	// 表示 32 位浮点，通道数为 3
	flowField = Mat::zeros(s, CV_32FC3);
	refinedETF = Mat::zeros(s, CV_32FC3);
	gradientMag = Mat::zeros(s, CV_32FC3);
}

/**
 * Generate initial ETF 
 * by taking perpendicular vectors(counter-clockwise) from gradient map
 */
void ETF::initial_ETF(string file, Size s) {
	// 把各个参数的大小改变成原图像的大小
	// 这些参数原本就有一个默认的大小
	resizeMat(s);

	Mat src = imread(file, 1);
	Mat src_n;
	Mat grad;
	// 将内容归一化到一个范围
	// 第三四个参数是范围归一化模式的最小值和最大值
	// NORM_MINMAX 是范围归一化
	normalize(src, src_n, 0.0, 1.0, NORM_MINMAX, CV_32FC1);
	//GaussianBlur(src_n, src_n, Size(51, 51), 0, 0);

	// Generate grad_x and grad_y
	Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
	// 第二个参数是输出
	// 第三四个参数是 x，y 上的差分结束
	// 第五个参数是 Sobel 核的大小
	Sobel(src_n, grad_x, CV_32FC1, 1, 0, 5);
	Sobel(src_n, grad_y, CV_32FC1, 0, 1, 5);

	//Compute gradient
	// 计算计算输入矩阵 x 和 y 对应的每个像素平方求和后开根号
	// 第三个参数是计算结果
	magnitude(grad_x, grad_y, gradientMag);
	normalize(gradientMag, gradientMag, 0.0, 1.0, NORM_MINMAX);

	flowField = Mat::zeros(src.size(), CV_32FC3);

// 这行是用来并行运行
#pragma omp parallel for
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			// Vec3f 是 3 通道 float 类型的 vector
			// mat.at<Vec3f>(y, x) 是访问图像的一种方式，处理像素
			// at 获取指定位置元素的值
			Vec3f u = grad_x.at<Vec3f>(i, j);
			Vec3f v = grad_y.at<Vec3f>(i, j);

			// 三个参数是三个通道的值
			flowField.at<Vec3f>(i, j) = normalize(Vec3f(v.val[0], u.val[0], 0));
		}
	}

	// 将向量旋转 90 度
	rotateFlow(flowField, flowField, 90);
}


// 平滑初始的 ETF
void ETF::refine_ETF(int kernel) {
#pragma omp parallel for
	for (int r = 0; r < flowField.rows; r++) {
		for (int c = 0; c < flowField.cols; c++) {
			computeNewVector(c, r, kernel);
		}
	}

	flowField = refinedETF.clone();
}

/*
 * Paper's Eq(1)
 */
void ETF::computeNewVector(int x, int y, const int kernel) {
	const Vec3f t_cur_x = flowField.at<Vec3f>(y, x);
	Vec3f t_new = Vec3f(0, 0, 0);

	for (int r = y - kernel; r <= y + kernel; r++) {
		for (int c = x - kernel; c <= x + kernel; c++) {
			if (r < 0 || r >= refinedETF.rows || c < 0 || c >= refinedETF.cols) continue;

			const Vec3f t_cur_y = flowField.at<Vec3f>(r, c);
			float phi = computePhi(t_cur_x, t_cur_y);
			float w_s = computeWs(Point2f(x, y), Point2f(c, r), kernel);
			float w_m = computeWm(norm(gradientMag.at<Vec3f>(y, x)), norm(gradientMag.at<float>(r, c)));
			float w_d = computeWd(t_cur_x, t_cur_y);
			t_new += phi*t_cur_y*w_s*w_m*w_d;
		}
	}
	refinedETF.at<Vec3f>(y, x) = normalize(t_new);
}

/*
 * Paper's Eq(5)
 */
float ETF::computePhi(cv::Vec3f x, cv::Vec3f y) {
	return x.dot(y) > 0 ? 1 : -1;
}

/*
 * Paper's Eq(2)
 */
float ETF::computeWs(cv::Point2f x, cv::Point2f y, int r) {
	return norm(x - y) < r ? 1 : 0;
}

/*
 * Paper's Eq(3)
 */
float ETF::computeWm(float gradmag_x, float gradmag_y) {
	float wm = (1 + tanh(gradmag_y - gradmag_x)) / 2;
	return wm;
}

/*
 * Paper's Eq(4)
 */
float ETF::computeWd(cv::Vec3f x, cv::Vec3f y) {
	return abs(x.dot(y));
}

// 将向量按照 θ 旋转
void ETF::rotateFlow(Mat& src, Mat& dst, float theta) {
	theta = theta / 180.0 * M_PI;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3f v = src.at<cv::Vec3f>(i, j);
			float rx = v[0] * cos(theta) - v[1] * sin(theta);
			float ry = v[1] * cos(theta) + v[0] * sin(theta);
			dst.at<cv::Vec3f>(i, j) = Vec3f(rx, ry, 0.0);
		}
	}

}

void ETF::resizeMat(Size s) {
	# CV_INTER_LINEAR 表示双线性插值
	# 按照第三个参数来决定放大后的大小
	# 如果第三个参数没指定数值，则那两个数字是长宽放大的倍数
	resize(flowField, flowField, s, 0, 0, CV_INTER_LINEAR);
	resize(refinedETF, refinedETF, s, 0, 0, CV_INTER_LINEAR);
	resize(gradientMag, gradientMag, s, 0, 0, CV_INTER_LINEAR);
}


