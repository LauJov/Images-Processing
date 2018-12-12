//g++ tracking.cpp -o tracking `pkg-config --cflags --libs opencv`&& ./tracking

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
 
using namespace cv;
using namespace std;
 
// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()
 

int cant = 0;
double px1=187, px2=23, py1=86, py2=320;
static void mouseEventHandler(int event, int x, int y, int, void* )
{
    if  ( event == EVENT_LBUTTONDOWN )
    {
        cout << "Left button of the mouse is clicked DOWN - position (" << x << ", " << y << ")" << endl;
        
        if (cant==0)
        {
            px1 = x;
            py1 = y;
        }
        if (cant==1)
        {
            px2 = x;
            py2 = y;
        }
        cant++;
    }
}

int main(int argc, char **argv)
{
    // List of tracker types in OpenCV 3.4.1
    string trackerTypes[8] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};
    // vector <string> trackerTypes(types, std::end(types));
 
    // Create a tracker
    string trackerType = trackerTypes[0];//0,1,6
 
    Ptr<Tracker> tracker;
 
    #if (CV_MINOR_VERSION < 3)
    {
        tracker = Tracker::create(trackerType);
    }
    #else
    {
        if (trackerType == "BOOSTING")
            tracker = TrackerBoosting::create();
        if (trackerType == "MIL")
            tracker = TrackerMIL::create();
        if (trackerType == "KCF")
            tracker = TrackerKCF::create();
        if (trackerType == "TLD")
            tracker = TrackerTLD::create();
        if (trackerType == "MEDIANFLOW")
            tracker = TrackerMedianFlow::create();
        if (trackerType == "GOTURN")
            tracker = TrackerGOTURN::create();
        if (trackerType == "MOSSE")
            tracker = TrackerMOSSE::create();
        if (trackerType == "CSRT")
            tracker = TrackerCSRT::create();
    }
    #endif
    // Read video
    VideoCapture video("video/chaplin.mp4");
     
    // Exit if video is not opened
    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl; 
        return 1; 
    } 
 
    // Read first frame 
    Mat frame; 
    bool ok = video.read(frame); 
 
    
 
    
 
    namedWindow("Tracking", 1);
    setMouseCallback("Tracking", mouseEventHandler, NULL);
    

    
    while(cant<2)
    {
        imshow("Tracking", frame); 
        waitKey(30);
    }

    // Define initial bounding box 
    if(px1>px2)
    {
        float aux = px1;
        px1 = px2;
        px2 = aux;
    }

    if(py1>py2)
    {
        float aux = py1;
        py1 = py2;
        py2 = aux;
    }

    Rect2d bbox(px1, py1, (px2-px1), (py2-py1)); 

    // Uncomment the line below to select a different bounding box 
    // bbox = selectROI(frame, false); 
    // Display bounding box. 
    rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 ); 

    imshow("Tracking", frame); 
    tracker->init(frame, bbox);

    while(video.read(frame))
    {     
        // Start timer
        double timer = (double)getTickCount();
         
        // Update the tracking result
        bool ok = tracker->update(frame, bbox);
         
        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);
         
        if (ok)
        {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        }
        else
        {
            // Tracking failure detected.
            putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        }
         
        // Display tracker type on frame
        putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
         
        // Display FPS on frame
        putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
 
        // Display frame.
        imshow("Tracking", frame);
         
        // Exit if ESC pressed.
        int k = waitKey(30);
        if(k == 27)
        {
            break;
        }
 
    }
}