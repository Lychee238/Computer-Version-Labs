#pragma once
#ifndef _METHODS_H_ 
#define _METHODS_H_ 
#include <opencv2/opencv.hpp>
#include <vector>

enum PicNoSet { pic1, pic2 };
enum StitchOrientation { horizontal, vertical };
class FeaturesDetectors {
private:
    std::string img1Path, img2Path;
    cv::Mat I1, I2, I1gray, I2gray;
    std::vector<cv::KeyPoint> I1fp, I2fp;
    cv::Mat I1des, I2des;
    cv::Ptr<cv::SIFT> SIFTDescriptor;
    cv::Ptr<cv::ORB> ORBDescriptor;
    cv::Ptr <cv::DescriptorMatcher> matcher;
    cv::Mat TransMatrix;
    std::vector<char> MatchMask;
    std::vector <cv::DMatch> matches;

public:
    FeaturesDetectors(const std::string& Path1, const std::string& Path2)
        : img1Path(Path1), img2Path(Path2) {}
    void LoadingImages();
    void CreatingSIFTDescriptors();
    void CreatingORBDescriptors();
    void showKeypointsDefault(enum PicNoSet picNo);
    void showKeypointsRICH(enum PicNoSet picNo);
    void CreatingBruteForceDescriptorMatcher();
    void CreatingBruteForceDescriptorMatcherHamming();
    void Creating5KDtreesDescriptorMatcher();
    void CreatingLSHDescriptorMatcher();
    void SingleBestMatches();
    void kNearestBestMatches(int k);
    void DisplayTop10Matches();
    void RANSAC();
    void ShowObjectPosition();
    void StitchImages(enum StitchOrientation orientation);

};


void FeaturesDetectors::LoadingImages()
{
    I1 = cv::imread(img1Path, cv::IMREAD_COLOR);
    I2 = cv::imread(img2Path, cv::IMREAD_COLOR);
    cv::cvtColor(I1, I1gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(I2, I2gray, cv::COLOR_BGR2GRAY);
    SIFTDescriptor = cv::SIFT::create(100);
    ORBDescriptor = cv::ORB::create(100);
}

void FeaturesDetectors::CreatingSIFTDescriptors()
{
    //SIFT
    SIFTDescriptor->detectAndCompute(I1gray, cv::noArray(), I1fp, I1des);
    SIFTDescriptor->detectAndCompute(I2gray, cv::noArray(), I2fp, I2des);

}

void FeaturesDetectors::CreatingORBDescriptors()
{
    //ORB
    ORBDescriptor->detectAndCompute(I1gray, cv::noArray(), I1fp, I1des);
    ORBDescriptor->detectAndCompute(I2gray, cv::noArray(), I2fp, I2des);
}

void FeaturesDetectors::showKeypointsDefault(enum PicNoSet picNo)
{
    cv::Mat Iout;
    if (picNo == pic1)
    {
        cv::drawKeypoints(I1, I1fp, Iout);
        cv::imshow("KeypointsDefault of Pic1", Iout);

    }
    else if (picNo == pic2)
    {
        cv::drawKeypoints(I2, I2fp, Iout);
        cv::imshow("KeypointsDefault of Pic2", Iout);
    }
    else {
        throw std::invalid_argument("Use picNo 1 - 2 !");
    }
}

void FeaturesDetectors::showKeypointsRICH(enum PicNoSet picNo)
{
    cv::Mat Iout;
    if (picNo == pic1)
    {
        cv::drawKeypoints(I1, I1fp, Iout, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(" ORB detector of Pic1 ", Iout);

    }
    else if (picNo == pic2)
    {
        cv::drawKeypoints(I2, I2fp, Iout, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(" ORB detector of Pic2", Iout);
    }
    else {
        throw std::invalid_argument("Use picNo 1 - 2 !");
    }
}


void FeaturesDetectors::CreatingBruteForceDescriptorMatcher()
{
    //Creating brute force descriptor matcher
    bool crossCheck = false;
    matcher = cv::BFMatcher::create(cv::NormTypes::NORM_L2, crossCheck);
}

void FeaturesDetectors::CreatingBruteForceDescriptorMatcherHamming()
{
    //Creating brute force descriptor matcher with Hamming distance for ORB
    bool crossCheck = false;
    matcher = cv::BFMatcher::create(cv::NormTypes::NORM_HAMMING, crossCheck);
}

void FeaturesDetectors::Creating5KDtreesDescriptorMatcher()
{
    //Creating FLANN 5 KD-trees descriptor matcher for SIFT descriptors
    matcher = cv::makePtr <cv::FlannBasedMatcher>(cv::makePtr<cv::flann::KDTreeIndexParams>(5));
}

void FeaturesDetectors::CreatingLSHDescriptorMatcher()
{
    //In case of ORB descriptors it is recommended to use LSH algorithm
    //Creating FLANN LSH descriptor matcher
    matcher = cv::makePtr <cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1));
}

void FeaturesDetectors::SingleBestMatches()
{
    matcher->match(I1des, I2des, matches);
}

void FeaturesDetectors::kNearestBestMatches(int k)
{
    std::vector <std::vector <cv::DMatch>> knn_matches;
    // Find KNN matches with k = 5
    matcher->knnMatch(I1des, I2des, knn_matches, k);
    // Select good matches
    double knn_ratio = 0.75;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < knn_ratio * knn_matches[i][1].distance)
        {
            matches.push_back(knn_matches[i][0]);
        }
    }
}

void FeaturesDetectors::DisplayTop10Matches()
{
    cv::Mat Matches;
    sort(matches.begin(), matches.end(),[](const cv::DMatch& a,const cv::DMatch & b)
        {
        return a.distance < b.distance;
        });
    int num_matches = std::min(10, (int)matches.size());
    cv::drawMatches(I1, I1fp, I2, I2fp, std::vector <cv::DMatch>(matches.begin(), matches.begin() + num_matches), Matches, cv::Scalar(0, 255, 0), cv::Scalar(-1), std::vector < char >(0), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("Good Matches", Matches);
}

void FeaturesDetectors::RANSAC()
{
    // Create arrays of point coordinates
    std::vector<cv::Point2f> I1pts, I2pts;
    for (int m = 0; m < matches.size(); m++)
    {
        I1pts.push_back(I1fp[matches[m].queryIdx].pt);
        I2pts.push_back(I2fp[matches[m].trainIdx].pt);
    }
    // Run RANSAC method
    TransMatrix = cv::findHomography(I1pts, I2pts, cv::RANSAC, 5, MatchMask);
}
void FeaturesDetectors::ShowObjectPosition()
{
    std::vector<cv::Point2f> I1box, I1to2box;
    // Image corners
    I1box.push_back(cv::Point2f(0, 0));
    I1box.push_back(cv::Point2f(0, static_cast<float>(I1.rows) - 1));
    I1box.push_back(cv::Point2f(static_cast<float>(I1.cols) - 1, static_cast<float>(I1.rows) - 1));
    I1box.push_back(cv::Point2f(static_cast<float>(I1.cols) - 1, 0));

    cv::perspectiveTransform(I1box, I1to2box, TransMatrix);

    // Convert to integers
    std::vector<cv::Point2i> I1to2box_i;
    for (int i = 0; i < I1to2box.size(); i++)
    {
        I1to2box_i.push_back(cv::Point2i(I1to2box[i]));
    }

    // Draw a red box on the second image
    cv::Mat I2res = I2.clone();
    cv::polylines(I2res, I1to2box_i, true, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    cv::imshow("Search result", I2res);

    cv::Mat Itrans;
    cv::drawMatches(I1, I1fp, I2res, I2fp, matches, Itrans, cv::Scalar(0, 255, 0), cv::Scalar(-1), MatchMask, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow(" Transformation ", Itrans);
}

void FeaturesDetectors::StitchImages(enum StitchOrientation orientation) {
    int w1 = I1.cols, h1 = I1.rows;
    int w2 = I2.cols, h2 = I2.rows;

    //calc the size of  the canva
    int canvas_width, canvas_height;
    if (orientation == horizontal) {
        canvas_width = w1 + w2;
        canvas_height = std::max(h1, h2);
    }
    else if (orientation == vertical) {
        canvas_width = std::max(w1, w2);
        canvas_height = h1 + h2;
    }
    else {
        throw std::invalid_argument("Invalid orientation");
    }

    //put image1 on the canva
    cv::Mat canvas = cv::Mat::zeros(canvas_height, canvas_width, CV_8UC3);
    I1.copyTo(canvas(cv::Rect(0, 0, w1, h1)));

    //make transformation to image2 and put it on the canva
    cv::Mat transformed_image2;
    cv::warpPerspective(I2, transformed_image2, TransMatrix, cv::Size(canvas_width, canvas_height), cv::WARP_INVERSE_MAP);
    transformed_image2(cv::Rect(0, 0, w1, h1)) = cv::Mat::zeros(h1, w1, transformed_image2.type());
    cv::addWeighted(canvas, 1.0, transformed_image2, 1.0, 0, canvas);

    //Crop black borders
    cv::Mat gray;
    cv::cvtColor(canvas, gray, cv::COLOR_BGR2GRAY);
    cv::Mat thresh;
    cv::threshold(gray, thresh, 1, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Rect boundingRect = cv::boundingRect(contours[0]);
    cv::Mat croppedImage = canvas(boundingRect);
    cv::imshow("Stitched Image", croppedImage);

    cv::waitKey(0);
}
#endif