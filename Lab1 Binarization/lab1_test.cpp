#include "methods_lab1.h"

void binarization() {
	// Create an OpenCV Mat object to store image data
	Mat I;
	// Read the image from file "pic.tiff" in grayscale mode and store it in Mat object I
	I = imread("pic4bi.tiff",
		IMREAD_GRAYSCALE);
	// Set the threshold value to 127
	double t = 127;
	// Create a new Mat object to store the thresholded image data
	Mat Inew;
	// Apply thresholding to the input image I using the specified threshold value t
// The thresholding type is binary, where pixels above the threshold are set to 255 and pixels below are set to 0
	threshold(I, Inew, t, 255, THRESH_BINARY);
	imshow("binarization", Inew);
}

void double_threshold() {
	// Create an OpenCV Mat object to store image data
	Mat I;
	// Read the image from file "pic.tiff" in color mode and store it in Mat object I
	I = imread("pic4bi.tiff", IMREAD_COLOR);
	// Set the threshold values
	double t1 = 127;
	double t2 = 200;
	// Create Mat objects to store the grayscale version of the image and the thresholded image
	Mat Igray, Inew;
	// Convert the color image I to grayscale and store the result in Igray
	cvtColor(I, Igray, COLOR_BGR2GRAY);
	// Apply inverse thresholding to Igray using threshold value t2
	// Pixels with intensity above t2 are set to 0, otherwise, they retain their original intensity
	threshold(Igray, Inew, t2, 255, THRESH_TOZERO_INV);
	// Apply thresholding to the resulting image Inew using threshold value t1
	// Pixels with intensity below t1 are set to 0, otherwise, they are set to 255
	threshold(Inew, Inew, t1, 255, THRESH_BINARY);
	imshow("double_threshold", Inew);
}

void OTSUmethod() {
	// Create an OpenCV Mat object to store image data
	Mat I;
	// Read the image from file "pic.tiff" in color mode and store it in Mat object I
	I = imread("pic4bi.tiff", IMREAD_COLOR);
	// Set the threshold values
	double t1 = 127;
	double t2 = 200;
	// Create Mat objects to store the grayscale version of the image and the thresholded image
	Mat Igray, Inew;
	// Convert the color image I to grayscale and store the result in Igray
	cvtColor(I, Igray, COLOR_BGR2GRAY);
	// Apply Otsu's thresholding method to the grayscale image Igray
	threshold(Igray, Inew, 0, 255, THRESH_OTSU);
	imshow("OTSUmethod", Inew);
}

void AdaptiveMethod() {
	// Create an OpenCV Mat object to store image data
	Mat I;
	// Read the image from file "pic.tiff" in color mode and store it in Mat object I
	I = imread("pic4bi.tiff", IMREAD_COLOR);
	// Set the threshold values
	double t1 = 127;
	double t2 = 200;
	// Create Mat objects to store the grayscale version of the image and the thresholded image
	Mat Igray, Inew;
	// Convert the color image I to grayscale and store the result in Igray
	cvtColor(I, Igray, COLOR_BGR2GRAY);
	// Apply adaptive thresholding to the grayscale image Igray using a Gaussian-weighted sum of neighborhood values
	adaptiveThreshold(Igray, Inew, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
	imshow("AdaptiveMethod", Inew);
}

// Function to calculate W(I) based on the provided formula
double calculateWeber(double intensity) {
	if (intensity >= 0 && intensity <= 88) {
		return 20 - (12 * intensity) / 88;
	}
	else if (intensity > 88 && intensity <= 138) {
		return 0.002 * pow((intensity - 88), 2);
	}
	else if (intensity > 138 && intensity <= 255) {
		return 7 * (intensity - 138) / 117 + 13;
	}
	else {
		return 0; // For out of range intensities
	}
}

void Webersegmentation() {
	// Convert input image to grayscale
	Mat I = imread("pic4seg1_0.tiff");
	//Mat I = imread("pic4seg1_1.tiff");
	//Mat I = imread("pic4seg1_2.png");

	Mat Igray;
	cvtColor(I, Igray, COLOR_RGB2GRAY);

	// Initialize segmentation result matrices
	Mat segmentationResult = Mat::zeros(Igray.size(), Igray.type());
	Mat segmentationResultGray = Mat::zeros(Igray.size(), Igray.type());

	// Initialize n for class numbering
	int n = 1;

	// Iterate until all pixels are ned
	while (countNonZero(segmentationResult) != segmentationResult.total()) {
		// Find the minimum intensity value in the unassigned region
		double min_intensity;
		minMaxLoc(Igray, &min_intensity, nullptr, nullptr, nullptr, segmentationResult == 0);

		// Calculate contrast for the current minimum intensity
		int contrast = calculateWeber(min_intensity);

		// Increment n for the next class
		n++;

		// Create a mask for pixels within the intensity range [min_intensity, min_intensity + contrast]
		Mat mask = (Igray >= min_intensity) & (Igray <= min_intensity + contrast);

		// Assign n to pixels in the mask region in segmentation result matrices
		segmentationResult.setTo(n, mask);
		segmentationResultGray.setTo(min_intensity, mask);
	}

	// Decrement n to match the number of classes
	n -= 1;

	// Decrement segmentation result to start ns from 0
	segmentationResult -= 1;

	// Apply colormap for visualization
	Mat coloredSegmentation;
	applyColorMap((segmentationResult * 255.0 / (n + 1)), coloredSegmentation, COLORMAP_JET);

	// Display original image, grayscale image, and segmented images
	imshow("Origin", I);
	imshow("Grayscale", Igray);
	imshow("Weber Segmentation JET", coloredSegmentation);
	imshow("Weber Segmentation", segmentationResultGray);


	// Save images
	//imwrite("Grayscale2.png", Igray);
	//imwrite("WeberSegmentationJET2.png", coloredSegmentation);
	//imwrite("WeberSegmentation2.png", segmentationResultGray);
}

void skinSegmentation() {
	// Load the image
	Mat I = imread("pic4seg1_0.tiff");
	//Mat I = imread("pic4seg1_1.tiff");
	//Mat I = imread("pic4seg1_2.png");

	// Create a blank mask image with the same size as the input image
	Mat skinMask = Mat::zeros(I.rows, I.cols, CV_8U);

	// Iterate over the pixels of the image
	for (int y = 0; y < I.rows; y++) {
		for (int x = 0; x < I.cols; x++) {
			// Get the RGB values of the pixel
			Vec3b intensity = I.at<Vec3b>(y, x);
			int R = intensity[2];
			int G = intensity[1];
			int B = intensity[0];

			// Check if the current pixel is a skin pixel
			bool isSkin = (R > 95 && G > 40 && B > 20 && (max(R, max(G, B)) - min(R, min(G, B))) > 15 && abs(R - G) > 15 && R > G && R > B)
				|| (R > 220 && G > 210 && B > 170 && abs(R - G) <= 15 && G > B && R > B)
				|| (static_cast<float>(R) / (R + G + B) > 0.185 && static_cast<float>(R * B) / pow((R + G + B), 2) > 0.107 && static_cast<float>(R * G) / pow((R + G + B), 2) > 0.112);

			// Invert the mask for skin pixels
			if (isSkin) {
				skinMask.at<uchar>(y, x) = 255;
			}
		}
	}
	// Create a result image by masking the original image with the skin mask
	Mat result;
	I.copyTo(result, skinMask);

	// Set non-skin pixels in the result image to black
	result.setTo(Scalar(0, 0, 0), ~skinMask);


	imshow("Skin Detection", result);
	waitKey(0);
	//imwrite("SkinSegmentation2.png", result);
}


// This function performs segmentation based on the CIELab color space.
void CIELabSegmentaion() {
	Mat I = imread("pic4seg2.tiff");
	// Convert the input image from BGR to CIELab color space
	Mat Ilab0;
	cvtColor(I, Ilab0, COLOR_BGR2Lab);

	// Split the image channels into BGR and CIELab components
	vector<Mat> Ibgr, Ilab;
	split(I, Ibgr);
	split(Ilab0, Ilab);

	// Display the input image and prompt the user to select three sample areas using the mouse
	//imshow(" Image ", I);
	//vector<Vec2i> sampleAreas;
	//setMouseCallback(" Image ", MouseHandler, &sampleAreas);
	//while (sampleAreas.size() < 3)
	//	waitKey(20);
	//setMouseCallback(" Image ", NULL);

	//using selecrROI() method
	Rect roi = selectROI("Image", I, false);
	int l = roi.tl().x;
	int t = roi.tl().y;
	int r = roi.br().x;
	int b = roi.br().y;
	Vec2i tl(l, t);
	Vec2i tr(r, t);
	Vec2i bl(l, b);
	Vec2i br(r, b);
	vector<Vec2i> sampleAreas = { tl, tr, bl, br };
	cout << sampleAreas.size() << endl;
	waitKey(0);

	// Compute the mean Lab values and corresponding BGR values for each sample area
	vector<Vec2d> colorMarks;
	vector<Vec3b> colorMarksBGR;
	for (int i = 0; i < sampleAreas.size(); i++) {
		Mat mask = Mat::zeros(Ilab[0].rows, Ilab[0].cols, CV_8U);
		circle(mask, sampleAreas[i], 10, Scalar(255), -1);
		//rectangle(mask, sampleAreas[i], Scalar(255));
		//imshow("mask" + to_string(i), mask);
		Scalar a = mean(Ilab[1], mask);
		Scalar b = mean(Ilab[2], mask);
		colorMarks.push_back(Vec2d(a[0], b[0]));
		Scalar B = mean(Ibgr[0], mask);
		Scalar G = mean(Ibgr[1], mask);
		Scalar R = mean(Ibgr[2], mask);
		colorMarksBGR.push_back(Vec3b((uchar)B[0], (uchar)G[0], (uchar)R[0]));
	}

	// Compute the Euclidean distance between each pixel and each sample area in the Lab color space
	vector<Mat> distance;
	for (int i = 0; i < colorMarks.size(); i++) {
		Mat tmp, tmp2;
		subtract(Ilab[1], colorMarks[i][0], tmp, noArray(), CV_64F);
		multiply(tmp, tmp, tmp);
		subtract(Ilab[2], colorMarks[i][1], tmp2, noArray(), CV_64F);
		multiply(tmp2, tmp2, tmp2);
		sqrt(tmp + tmp2, tmp);
		distance.push_back(tmp);
	}

	// Determine the minimum distance for each pixel and assign labels accordingly
	Mat distance_min = distance[0].clone();
	for (int i = 1; i < distance.size(); i++)
		min(distance_min, distance[i], distance_min);

	Mat labels = Mat::zeros(Ilab[0].rows, Ilab[0].cols, CV_8U);
	for (int i = 0; i < colorMarks.size(); i++) {
		Mat mask = distance[i] == distance_min;
		labels.setTo(Scalar(i), mask);
	}

	// Create segmented frames based on the assigned labels and display them
	vector<Mat>  segmentedFrames;
	for (int i = 0; i < colorMarks.size(); i++) {
		Mat Itmp = Mat::zeros(I.rows, I.cols, I.type());
		I.copyTo(Itmp, labels == i);
		imshow("segmentedFrame" + to_string(i), Itmp);
		//imwrite("segmentedFrame" + to_string(i) + ".png", Itmp);
		segmentedFrames.push_back(Itmp);
	}

	// Create a plot to visualize the segmented image with marked colors
	Mat Iplot(256, 256, CV_8UC3, Scalar(255, 255, 255));
	for (int i = 0; i < colorMarks.size(); i++) {
		Mat Itmp = Mat::zeros(I.rows, I.cols, I.type());
		Mat mask = labels == i;
		for (int x = 0; x < mask.cols; x++)
			for (int y = 0; y < mask.rows; y++)
				if (mask.at<uchar>(y, x) != 0)
					Iplot.at<Vec3b>(Ilab[1].at<uchar>(y, x), Ilab[2].at<uchar>(y, x)) = colorMarksBGR[i];
	}

	// Display the final segmented image
	imshow("Segmented Image", Iplot);
	//imwrite("Segmented Image.png", Iplot);


	waitKey(0);
}

// Function to perform k-means clustering on an input image
void k_means() {
	// Convert input image to Lab color space
	Mat I = imread("pic4seg2.tiff");
	Mat Ilab0;
	cvtColor(I, Ilab0, COLOR_BGR2Lab);

	// Split Lab channels
	vector <Mat> Ilab;
	split(Ilab0, Ilab);

	// Extract the 'a' and 'b' channels and merge them
	Mat ab;
	merge(&(Ilab[1]), 2, ab);

	// Reshape the 'ab' matrix for k-means clustering
	ab = ab.reshape(0, 1);
	ab.convertTo(ab, CV_32F);

	// Set the number of clusters
	int k = 3;

	// Perform k-means clustering
	Mat labels;
	kmeans(ab, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 10, KMEANS_RANDOM_CENTERS);

	// Reshape the labels matrix to match the size of the 'a' channel
	labels = labels.reshape(0, Ilab[0].rows);

	// Create segmented frames for each cluster
	vector<Mat> segmentedFrames;
	for (int i = 0; i < k; i++) {
		// Create a mask for the current cluster
		Mat Itmp = Mat::zeros(I.rows, I.cols, I.type());
		Mat mask = labels == i;

		// Copy the corresponding pixels from the input image to the segmented frame
		I.copyTo(Itmp, mask);

		// Store the segmented frame
		segmentedFrames.push_back(Itmp);
	}

	for (int i = 0; i < segmentedFrames.size(); i++) {
		imshow("segmentedFrames" + to_string(i), segmentedFrames[i]);
		//imwrite("segmentedFrames" + to_string(i) + ".png", segmentedFrames[i]);
	}
}

void textureSegmentation() {
	// Read grayscale image
	Mat I;
	I = imread("pic4seg3.jpg ", IMREAD_GRAYSCALE);

	// Define structuring element for morphological operations
	Mat el = getStructuringElement(MORPH_RECT, Size(9, 9));

	// Calculate entropy of the input image
	Mat E, Eim;
	entropy(I, E, el);

	// Normalize entropy values
	double Emin, Emax;
	minMaxLoc(E, &Emin, &Emax);
	Eim = (E - Emin) / (Emax - Emin);
	Eim.convertTo(Eim, CV_8U, 255);

	// Binarize the normalized entropy image using Otsu's thresholding
	Mat BW1;
	threshold(Eim, BW1, 0, 255, THRESH_OTSU);

	// Perform morphological operations and fill holes in the binary image
	Mat BWao, closeBWao, Mask1;
	bwareaopen(BW1, BWao, 2000);
	Mat nhood = getStructuringElement(MORPH_RECT, Size(9, 9));
	morphologyEx(BWao, closeBWao, MORPH_CLOSE, nhood);
	imfillholes(closeBWao, Mask1);

	// Extract contours from the filled mask
	vector<vector<Point>> contours;
	findContours(Mask1, contours, RETR_TREE, CHAIN_APPROX_NONE);

	// Create a boundary image using the contours
	Mat boundary = Mat::zeros(Mask1.rows, Mask1.cols, CV_8UC1);
	drawContours(boundary, contours, -1, 255, 1);

	// Create a segmented image based on the boundary
	Mat segmentResults = I.clone();
	segmentResults.setTo(Scalar(255), boundary != 0);

	// Generate a texture image by masking the original image with the boundary mask
	Mat I2 = I.clone();
	I2.setTo(0, Mask1 != 0);

	// Repeat the texture analysis process for the masked image I2
	Mat E2, Eim2;
	entropy(I2, E2, el);
	double Emin2, Emax2;
	minMaxLoc(E2, &Emin2, &Emax2);
	Eim2 = (E2 - Emin2) / (Emax2 - Emin2);
	Eim2.convertTo(Eim2, CV_8U, 255);
	Mat BW2;
	threshold(Eim2, BW2, 0, 255, THRESH_OTSU);
	Mat BW2ao, closeBW2ao, Mask2;
	bwareaopen(BW2, BW2ao, 2000);
	morphologyEx(BW2ao, closeBW2ao, MORPH_CLOSE, nhood);
	imfillholes(closeBW2ao, Mask2);

	// Extract contours and create a boundary for the second texture image
	vector<vector<Point>> contours2;
	findContours(Mask2, contours2, RETR_TREE, CHAIN_APPROX_NONE);
	Mat boundary2 = Mat::zeros(Mask2.rows, Mask2.cols, CV_8UC1);
	drawContours(boundary2, contours2, -1, 255, 1);

	// Create the second segmented image based on the boundary
	Mat segmentResults2 = I2.clone();
	segmentResults2.setTo(255, boundary2 != 0);

	// Generate texture images by masking the original image with the inverse of the boundaries
	Mat texture1 = I.clone();
	texture1.setTo(0, Mask2 == 0);
	Mat texture2 = I.clone();
	texture2.setTo(0, Mask1 == 0);

	// Display original grayscale image and the generated texture images
	imshow("originGray", I);
	//imshow("Entropy", Eim);
	//imwrite("BW1", BW1);
	//imshow("BWao", BWao);
	//imshow("closeBWao", closeBWao);
	//imshow("texture1", texture1);
	//imshow("texture2", texture2);
	//imshow("segmentResults", segmentResults);
	//imshow("segmentResults2", segmentResults2);

	waitKey(0);

	//imwrite("originGray.png", I);
	//imwrite("Entropy.png", Eim);
	//imwrite("BW1.png", BW1);
	//imwrite("BWao.png", BWao);
	//imwrite("closeBWao.png", closeBWao);
	//imwrite("texture1.png", texture1);
	//imwrite("texture2.png", texture2);
	//imwrite("segmentResults.png", segmentResults);
	//imwrite("segmentResults2.png", segmentResults2);
	PrintParameters(texture1, Mask2, texture2, Mask1);
}


int main() {
	//binarization();
	//double_threshold();
	//OTSUmethod();
	//AdaptiveMethod();
	//Webersegmentation();
	//skinSegmentation();
	CIELabSegmentaion();
	//k_means();
	//textureSegmentation();

	waitKey(0);
	destroyAllWindows();
	return 0;
}
