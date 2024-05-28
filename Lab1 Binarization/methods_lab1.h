#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;


void MouseHandler(int event, int x, int y, int flags, void* param) {
	if (event != EVENT_LBUTTONDBLCLK)
		return;
	if (param == NULL)
		return;
	((vector < Vec2i > *) param)->push_back(Vec2i(x, y));
}

void entropy(Mat& I, Mat& Iout, Mat& el) {
	// Check input image data
	if (I.channels() != 1 || I.type() != CV_8U)
		return;

	// Convert to image with border
	Mat Icopy;
	copyMakeBorder(I, Icopy,
		int((el.rows - 1) / 2),
		int(el.rows / 2),
		int((el.cols - 1) / 2),
		int(el.cols / 2), BORDER_REPLICATE);

	// Initialize output image
	Iout = Mat::zeros(I.rows, I.cols, CV_32F);

	// Initialize local histogram
	double hist[256];
	for (int i = 0; i < 256; i++)
		hist[i] = 0;

	// Calculate element size
	int count = 0;
	for (int i = 0; i < el.rows; i++)
		for (int j = 0; j < el.cols; j++)
			if (el.at<uchar>(i, j))
				count++;

	// For each image pixel
	for (int y = 0; y < I.rows; y++)
		for (int x = 0; x < I.cols; x++) {
			// Calculate local histogram
			for (int i = 0; i < el.rows; i++)
				for (int j = 0; j < el.cols; j++)
					if (el.at<uchar>(i, j))
						hist[Icopy.at<uchar>(y + i, x + j)] += 1;

			// Calculate entropy
			double val = 0;
			for (int i = 0; i < 256; i++)
				if (hist[i] > 0) {
					val -= hist[i] / count * log2(hist[i] / count);
					hist[i] = 0;
				}
			Iout.at<float>(y, x) = float(val);
		}
}

void bwareaopen(const Mat& A, Mat& C, int dim, int conn = 8) {
	if (A.channels() != 1 || A.type() != CV_8U)
		return;

	// Find all connected components
	Mat labels, stats, centers;
	int num = connectedComponentsWithStats(A, labels, stats, centers, conn);

	// Clone image
	C = A.clone();

	// Check size of all connected components
	vector<int> td;
	for (int i = 0; i < num; i++)
		if (stats.at<int>(i, CC_STAT_AREA) < dim)
			td.push_back(i);

	// Remove small areas
	if (td.size() > 0) {
		if (A.type() == CV_8U) {
			for (int i = 0; i < C.rows; i++)
				for (int j = 0; j < C.cols; j++)
					for (int k = 0; k < td.size(); k++)
						if (labels.at<int>(i, j) == td[k]) {
							C.at<uchar>(i, j) = 0;
							continue;
						}
		}
		else {
			for (int i = 0; i < C.rows; i++)
				for (int j = 0; j < C.cols; j++)
					for (int k = 0; k < td.size(); k++)
						if (labels.at<int>(i, j) == td[k]) {
							C.at<float>(i, j) = 0;
							continue;
						}
		}
	}
}

void imfillholes(Mat& I, Mat& Iout) {
	// Check input image data
	if (I.channels() != 1 || I.type() != CV_8U)
		return;

	Mat mask = I.clone();

	// Fill mask from all horizontal borders
	for (int i = 0; i < I.cols; i++) {
		if (mask.at<uchar>(0, i) == 0)
			floodFill(mask, Point(i, 0), Scalar(255), NULL, Scalar(10), Scalar(10));
		if (mask.at<uchar>(I.rows - 1, i) == 0)
			floodFill(mask, Point(i, I.rows - 1), Scalar(255), NULL, Scalar(10), Scalar(10));
	}

	// Fill mask from all vertical borders
	for (int i = 0; i < I.rows; i++) {
		if (mask.at<uchar>(i, 0) == 0)
			floodFill(mask, Point(0, i), Scalar(255), NULL, Scalar(10), Scalar(10));
		if (mask.at<uchar>(i, I.cols - 1) == 0)
			floodFill(mask, Point(I.cols - 1, i), Scalar(255), NULL, Scalar(10), Scalar(10));
	}

	// Use the mask to create an image
	Iout = I.clone();
	Iout.setTo(Scalar(255), mask == 0);
}


Mat histogram(Mat& image, Mat& mask) {
	{
		//Mat image = imread("pic4seg3.jpg", IMREAD_GRAYSCALE);
		if (image.empty()) {
			cerr << "Failed to read image." << endl;
			return image;
		}

		Mat hist;
		int histSize = 256;
		float range[] = { 0, 256 };
		const float* histRange = { range };
		bool uniform = true;
		bool accumulate = false;
		calcHist(&image, 1, nullptr, mask, hist, 1, &histSize, &histRange, uniform, accumulate);
		int totalPixels = countNonZero(mask);
		hist /= totalPixels;

		//cout << hist << endl;


		//int histWidth = 512;
		//int histHeight = 400;
		//int binWidth = cvRound((double)histWidth / histSize);
		//Mat histImage(histHeight, histWidth, CV_8UC3, Scalar(0, 0, 0));
		//normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		//for (int i = 1; i < histSize; i++) {
		//	line(histImage, Point(binWidth * (i - 1), histHeight - cvRound(hist.at<float>(i - 1))),
		//		Point(binWidth * (i), histHeight - cvRound(hist.at<float>(i))),
		//		Scalar(255, 255, 255), 2, 8, 0);
		//}

		//imshow("Histogram", histImage);
		//waitKey(0);

		return hist;
	}
}
// Calculate mean of the image histogram
double calculateMean(const Mat& hist) {
	double mean = 0;
	for (int i = 0; i < hist.rows; i++) {
		//cout << i << " " << hist.at<float>(i) << endl;
		mean += i * hist.at<float>(i);
	}
	return mean;
}

// Calculate variance of the image histogram
double calculateVariance(const Mat& hist, double mean) {
	double variance = 0;
	for (int i = 0; i < hist.rows; i++) {
		variance += pow(i - mean, 2) * hist.at<float>(i);
	}
	return variance;
}



// Calculate entropy of the image histogram
double calculateEntropy(const Mat& hist) {
	double entropy = 0;
	for (int i = 0; i < hist.rows; i++) {
		double p = hist.at<float>(i);
		if (p > 0) {
			entropy -= p * log2(p);
		}
	}
	return entropy;
}

// Calculate uniformity of the image histogram
double calculateUniformity(const Mat& hist) {
	double uniformity = 0;
	for (int i = 0; i < hist.rows; i++) {
		uniformity += pow(hist.at<float>(i), 2);
	}
	return uniformity;
}

// Calculate relative smoothness
double calculateRelativeSmoothness(double variance) {
	double relativeSmoothness = 1 - 1 / (1 + variance);
	return relativeSmoothness;
}

double calculateCentralMoment(const Mat& hist, double mean, int n) {
	double moment = 0;
	for (int i = 0; i < hist.rows; ++i) {
		double zi = i;
		moment += pow(zi - mean, n) * hist.at<float>(i);
	}
	return moment;
}

void PrintParameters(Mat& texture1, Mat& Mask2, Mat& texture2, Mat& Mask1) {
	Mat hist1 = histogram(texture1, Mask2);
	double m1 = calculateMean(hist1);
	double s1 = calculateVariance(hist1, m1);
	double R1 = calculateRelativeSmoothness(s1);
	double u1 = calculateCentralMoment(hist1, m1, 3);
	double U1 = calculateUniformity(hist1);
	double En1 = calculateEntropy(hist1);
	printf("texture|   m   |   s   |   R   |   ¦Ì3 |   U   |   E   |\n");
	printf("--------------------------------------------------------\n");
	printf("sky    |%-7.3f|%-7.3f|%-7.3f|%-7.3f|%-7.3f|%-7.3f|\n", m1, s1, R1, u1, U1, En1);


	Mat hist2 = histogram(texture2, Mask1);
	double m2 = calculateMean(hist2);
	double s2 = calculateVariance(hist2, m2);
	double R2 = calculateRelativeSmoothness(s2);
	double u2 = calculateCentralMoment(hist2, m2, 3);
	double U2 = calculateUniformity(hist2);
	double En2 = calculateEntropy(hist2);
	printf("--------------------------------------------------------\n");
	printf("rock   |%-7.3f|%-7.3f|%-7.3f|%-7.3f|%-7.3f|%-7.3f|\n", m2, s2, R2, u2, U2, En2);
}