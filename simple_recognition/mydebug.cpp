#include "mydebug.h"
#include <highgui.h>
#include <cv.h>
#include "mydefine.h"

void imshow(const char * windowName, const int width, const int height, const int channels, const uchar * data)
{
	IplImage *temp = cvCreateImage(cvSize(width, height), 8, channels);
	uchar *p = (uchar *)temp->imageData;
	int widthStep = width*channels;
	for (size_t y = 0; y < height; y++)
	{
		for (size_t x = 0; x < widthStep; x++)
		{
			for (size_t k = 0; k < channels; k++)
			{
				p[y*widthStep + x + k] = data[y*widthStep + x + k];
			}
		}
	}
	cvNamedWindow(windowName);
	cvShowImage(windowName, temp);
	cvReleaseImage(&temp);
}

void imshow(const char *windowName, const IplImage *image)
{
	cvNamedWindow(windowName);
	cvShowImage(windowName, image);
}

void drawPoint(IplImage * image, const Point & point, CvScalar & color, int size)
{
	cvCircle(image, cvPoint(point.x, point.y), 0, color, size);
}

void drawPoints(IplImage * image, const std::vector<Point>& points, CvScalar & color, int size)
{
	for (int i = 0; i < points.size(); i++)
	{
		cvCircle(image, cvPoint(points[i].x, points[i].y), 0, color, size);
	}
}

void drawBoxex(IplImage * image, const std::vector<Box>& boxes, CvScalar & color, int size)
{
	for (int i = 0; i < boxes.size(); i++)
	{
		const Box & box = boxes[i];
		for (int j = 0; j < 4; j++)
		{
			cvLine(image, cvPoint(box[j].x, box[j].y), cvPoint(box[(j + 1) % 4].x, box[(j + 1) % 4].y), color, size);
		}
	}
}

void drawBox(IplImage * image, const Box & box, CvScalar & color, int size)
{
	for (int j = 0; j < 4; j++)
	{
		cvLine(image, cvPoint(box[j].x, box[j].y), cvPoint(box[(j + 1) % 4].x, box[(j + 1) % 4].y), color, size);
	}
}

void rotate(const int width, const int height, const int channels, uchar * data, const int angle)
{
	;
}
