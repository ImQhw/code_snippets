#ifndef _MY_DEBUG_H_
#define _MY_DEBUG_H_

#include <highgui.h>

#include <vector>
#include "mydefine.h"

void imshow(const char *windowName, const int width, const int height, const int channels, const uchar *data);

void imshow(const char *windowName, const IplImage *image);

void drawPoint(IplImage *image, const Point & point, CvScalar & color, int size = 1);

void drawPoints(IplImage *image, const std::vector<Point> & points, CvScalar & color, int size = 1);

void drawBoxex(IplImage *image, const std::vector < Box > & boxes, CvScalar & color, int size = 1);

void drawBox(IplImage *image, const Box & box, CvScalar & color, int size = 1);

void rotate(const int width, const int height, const int channels, uchar * data, const int angle = 0);

#endif