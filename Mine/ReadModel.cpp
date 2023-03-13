
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaarithm.hpp>


using namespace std;
using namespace cv;
using namespace dnn;
using namespace cuda;



void main() 
{
	String modelWeights = "ocr_prediction1.pb"; //unet_membrane.pb "frozen_model.pb"
	Mat blob, Im;


	Im=imread("g.png");


	Net net = readNetFromTensorflow(modelWeights);


	blobFromImage(Im, blob, .00392156, Size(150, 150), Scalar(0, 0, 0), true, false);

	/*Sets the input to the network*/
	
	net.setInput(blob);

	//int score = net.forward();

	Mat score = net.forward();


}


