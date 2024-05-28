#include "methods.h"


int main()
{
    FeaturesDetectors detectors("pic4FD.jpg", "pic4FD1.jpg");
    detectors.LoadingImages();


    //Descriptors
    detectors.CreatingSIFTDescriptors();
    //detectors.CreatingORBDescriptors();

    //Show key points
    //detectors.showKeypointsDefault(pic1);
    //detectors.showKeypointsRICH(pic1);

    //Matchers
    //detectors.CreatingBruteForceDescriptorMatcher();
    //detectors.CreatingBruteForceDescriptorMatcherHamming();
    detectors.Creating5KDtreesDescriptorMatcher();
    //detectors.CreatingLSHDescriptorMatcher();

    //Matches
    //detectors.SingleBestMatches();
    detectors.kNearestBestMatches(5);

    //Display matches
    detectors.DisplayTop10Matches();

    //RANSAC
    detectors.RANSAC();
    detectors.ShowObjectPosition();


    //FeaturesDetectors StitchTester("pic4FD2.jpg", "pic4FD3.jpg");
    //StitchTester.LoadingImages();
    //StitchTester.CreatingSIFTDescriptors();
    //StitchTester.Creating5KDtreesDescriptorMatcher();
    //StitchTester.kNearestBestMatches(5);
    //StitchTester.RANSAC();
    //StitchTester.StitchImages(horizontal);
    //detectors.StitchImages(horizontal);


    cv::waitKey(0);
    return 0;
}
