#include <iostream>
#include <opencv2/opencv.hpp>

void detectFacePic(const std::string& filename) {
	// Load the input image
	cv::Mat I = cv::imread(filename, cv::IMREAD_COLOR);
	if (I.empty()) {
		std::cerr << "Error: Could not load image" << std::endl;
		return;
	}

	// Convert the image to grayscale
	cv::Mat Igray;
	cv::cvtColor(I, Igray, cv::COLOR_BGR2GRAY);

	// Load face cascade
	cv::CascadeClassifier face_detector;
	//cv::String face_cascade_fn = cv::samples::findFile("haarcascades/haarcascade_frontalface_default.xml");
	cv::String face_cascade_fn = cv::samples::findFile("haarcascades/haarcascade_frontalface_alt.xml");//beter performance than default
	if (!face_detector.load(face_cascade_fn)) {
		std::cerr << "Error: Could not load face cascade" << std::endl;
		return;
	}

	// Detect faces
	std::vector<cv::Rect> faces;
	face_detector.detectMultiScale(Igray, faces, 1.07, 3);
	cv::Mat IoutEyes = I.clone();
	cv::Mat IoutNoses = I.clone();

	std::cout << faces.size() << std::endl;
	for (const auto& face : faces) {
		cv::rectangle(IoutEyes, face, cv::Scalar(0, 255, 255), 1);
		cv::rectangle(IoutNoses, face, cv::Scalar(0, 255, 255), 1);

	}

	// Load eyes cascade
	cv::CascadeClassifier eye_detector;
	cv::String eye_cascade_fn = cv::samples::findFile("haarcascades/haarcascade_eye.xml");
	if (!eye_detector.load(eye_cascade_fn)) {
		std::cerr << "Error: Could not load eyes cascade" << std::endl;
		return;
	}

	// Load nose cascade
	cv::CascadeClassifier nose_detector;
	cv::String nose_cascade_fn = cv::samples::findFile("haarcascades/haarcascade_nose.xml");
	if (!nose_detector.load(nose_cascade_fn)) {
		std::cerr << "Error: Could not load nose cascade" << std::endl;
		return;
	}

	// For each face use it as a ROI and detect eyes and nose
	for (const auto& face : faces) {
		// Detect eyes in the upper 2/3 of the face
		cv::Mat Iface_top = Igray(cv::Rect(face.x, face.y, face.width, face.height * 2 / 3));
		std::vector<cv::Rect> eyes;
		eye_detector.detectMultiScale(Iface_top, eyes, 1.05);
		for (const auto& eye : eyes) {
			cv::Rect eye_rect(face.x + eye.x, face.y + eye.y, eye.width, eye.height);
			cv::rectangle(IoutEyes, eye_rect, cv::Scalar(147, 20, 255), 1);
		}

		// Detect nose in the lower 2/3 of the face
		cv::Mat Iface_bottom = Igray(cv::Rect(face.x, face.y + face.height / 3, face.width, face.height * 2 / 3));
		std::vector<cv::Rect> nose;
		nose_detector.detectMultiScale(Iface_bottom, nose, 1.05);
		for (const auto& n : nose) {
			cv::Rect nose_rect(face.x + n.x, face.y + face.height / 3 + n.y, n.width, n.height);
			cv::rectangle(IoutNoses, nose_rect, cv::Scalar(255, 0, 0), 1);
		}
	}

	// Display the results
	cv::imshow("Detected Faces, Eyes", IoutEyes);
	cv::imshow("Detected Faces, Noses", IoutNoses);

}

void detectFaceVideo(const std::string& filename)
{
	// Load the pre-recorded video
	cv::VideoCapture cap(filename);
	if (!cap.isOpened()) {
		std::cerr << "Error: Could not open video file" << std::endl;
		return;
	}

	// Load face cascade
	cv::CascadeClassifier face_detector;
	cv::String face_cascade_fn = cv::samples::findFile("haarcascades/haarcascade_frontalface_alt.xml");
	if (!face_detector.load(face_cascade_fn)) {
		std::cerr << "Error: Could not load face cascade" << std::endl;
		return;
	}

	// Load eyes cascade
	cv::CascadeClassifier eye_detector;
	cv::String eye_cascade_fn = cv::samples::findFile("haarcascades/haarcascade_eye.xml");
	if (!eye_detector.load(eye_cascade_fn)) {
		std::cerr << "Error: Could not load eyes cascade" << std::endl;
		return;
	}

	// Load nose cascade
	cv::CascadeClassifier nose_detector;
	cv::String nose_cascade_fn = cv::samples::findFile("haarcascades/haarcascade_nose.xml");
	if (!nose_detector.load(nose_cascade_fn)) {
		std::cerr << "Error: Could not load nose cascade" << std::endl;
		return;
	}

	// Read frames from the video
	cv::Mat frame;
	while (cap.read(frame)) {
		// Convert the frame to grayscale
		cv::Mat gray;
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

		// Detect faces
		std::vector<cv::Rect> faces;
		face_detector.detectMultiScale(gray, faces, 1.07, 3);
		std::cout << faces.size() << std::endl;
		for (const auto& face : faces) {
			cv::rectangle(frame, face, cv::Scalar(0, 255, 255), 2);

			// Detect eyes in the upper 2/3 of the face
			cv::Mat Iface_top = gray(cv::Rect(face.x, face.y, face.width, face.height * 2 / 3));
			std::vector<cv::Rect> eyes;
			eye_detector.detectMultiScale(Iface_top, eyes, 1.05);
			for (const auto& eye : eyes) {
				cv::Rect eye_rect(face.x + eye.x, face.y + eye.y, eye.width, eye.height);
				cv::rectangle(frame, eye_rect, cv::Scalar(147, 20, 255), 2);
			}

			// Detect nose in the lower 2/3 of the face
			cv::Mat Iface_bottom = gray(cv::Rect(face.x, face.y + face.height / 3, face.width, face.height * 2 / 3));
			std::vector<cv::Rect> nose;
			nose_detector.detectMultiScale(Iface_bottom, nose, 1.05);
			for (const auto& n : nose) {
				cv::Rect nose_rect(face.x + n.x, face.y + face.height / 3 + n.y, n.width, n.height);
				cv::rectangle(frame, nose_rect, cv::Scalar(255, 0, 0), 2);
			}
		}

		// Display the frame
		cv::imshow("Face Detection", frame);

		// Break the loop if 'q' is pressed
		if (cv::waitKey(30) == 'q') {
			break;
		}
	}
}

void detectFaceWebCam()
{
	// Open the default camera
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cerr << "Error: Could not open camera" << std::endl;
		return;
	}

	// Load face cascade
	cv::CascadeClassifier face_detector;
	cv::String face_cascade_fn = cv::samples::findFile("haarcascades/haarcascade_frontalface_alt.xml");
	if (!face_detector.load(face_cascade_fn)) {
		std::cerr << "Error: Could not load face cascade" << std::endl;
		return;
	}

	// Load eyes cascade
	cv::CascadeClassifier eye_detector;
	cv::String eye_cascade_fn = cv::samples::findFile("haarcascades/haarcascade_eye.xml");
	if (!eye_detector.load(eye_cascade_fn)) {
		std::cerr << "Error: Could not load eyes cascade" << std::endl;
		return;
	}

	// Load nose cascade
	cv::CascadeClassifier nose_detector;
	cv::String nose_cascade_fn = cv::samples::findFile("haarcascades/haarcascade_nose.xml");
	if (!nose_detector.load(nose_cascade_fn)) {
		std::cerr << "Error: Could not load nose cascade" << std::endl;
		return;
	}

	cv::Mat frame;
	while (true) {
		// Capture frame-by-frame
		cap >> frame;
		if (frame.empty()) {
			std::cerr << "Error: Could not grab frame" << std::endl;
			break;
		}

		// Convert the frame to grayscale
		cv::Mat gray;
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

		// Detect faces
		std::vector<cv::Rect> faces;
		face_detector.detectMultiScale(gray, faces, 1.07, 3);
		for (const auto& face : faces) {
			cv::rectangle(frame, face, cv::Scalar(0, 255, 255), 2);

			// Detect eyes in the upper 2/3 of the face
			cv::Mat Iface_top = gray(cv::Rect(face.x, face.y, face.width, face.height * 2 / 3));
			std::vector<cv::Rect> eyes;
			eye_detector.detectMultiScale(Iface_top, eyes, 1.05);
			for (const auto& eye : eyes) {
				cv::Rect eye_rect(face.x + eye.x, face.y + eye.y, eye.width, eye.height);
				cv::rectangle(frame, eye_rect, cv::Scalar(147, 20, 255), 2);
			}

			// Detect nose in the lower 2/3 of the face
			cv::Mat Iface_bottom = gray(cv::Rect(face.x, face.y + face.height / 3, face.width, face.height * 2 / 3));
			std::vector<cv::Rect> nose;
			nose_detector.detectMultiScale(Iface_bottom, nose, 1.05);
			for (const auto& n : nose) {
				cv::Rect nose_rect(face.x + n.x, face.y + face.height / 3 + n.y, n.width, n.height);
				cv::rectangle(frame, nose_rect, cv::Scalar(255, 0, 0), 2);
			}
		}

		// Display the resulting frame
		cv::imshow("Face Detection", frame);

		// Break the loop if 'q' is pressed
		if (cv::waitKey(30) == 'q') {
			break;
		}
	}
}

int main()
{
	//detectFacePic("pic4FD.jpg");
	//detectFacePic("pic4FD1.jpg");
	detectFaceVideo("video4FD1.mp4");
	//detectFaceWebCam();



	cv::waitKey(0);

	return 0;
}
