#include "cimageproc.h"
#include "mydefine.h"
#include "mydebug.h"

#include <iostream>
#include <algorithm>
#include <stack>
#include <math.h>
#include <time.h>
#include <float.h>
#include <vector>

#include <highgui.h>

using namespace std;

#define DEBUG false

////******类静态成员初始化********////
int CImageProc::dx[9] = { 1,  1,  0, -1, -1, -1, 0, 1, 0 };
int CImageProc::dy[9] = { 0, -1, -1, -1,  0,  1, 1, 1, 0 };

Vector3 CImageProc::COLOR_TABLE[MAX_CLASS_COUNT] =
{
	Vector3(0, 0, 255),	//硬币 红色
	Vector3(0, 255, 0),	//螺母 绿色
	Vector3(255, 0, 0),	//L型扳手 蓝色
	Vector3(128, 0, 128),	//螺钉 紫色
	Vector3(255, 255, 255)	//其他 白色
};

////************************************************构造、析构函数定义*************************************************////
CImageProc::CImageProc(const int width, const int height, const int channels, const uchar * imageData)
	:width(width), height(height)
{
	m_Bin = new uchar[width*height];
	if (channels == 1)
	{
		otsuCut(imageData);
	}
	else if (channels == 3)
	{
		uchar *_gray = new uchar[width*height];
		cvtRGB2Gray(imageData, _gray);
#if DEBUG
		imshow("gray", width, height, 1, _gray);
#endif
		otsuCut(_gray);
		delete[] _gray;
	}

	erode();
	dilate();
#if DEBUG
	imshow("bin", width, height, 1, m_Bin);
#endif
	initMembers();
}
CImageProc::~CImageProc()
{
	delete[] m_Bin;
}


////********************************************初始化函数声明*************************************************////
void CImageProc::cvtRGB2Gray(const uchar * color, uchar * gray) const
{
	size_t i, j;
	const uchar *pColor = NULL;
	uchar * pGray = NULL;
	int widthStep = width * 3;
	for (i = 0; i < height; i++)
	{
		pColor = color + i*widthStep;
		pGray = gray + i*width;
		for (j = 0; j < width; j++)
		{
			pGray[j] = (pColor[3 * j] * 19595 + pColor[3 * j + 1] * 38469 + pColor[3 * j + 2] * 7472) >> 16;
		}
	}
}

void CImageProc::otsuCut(const uchar * data)
{
	int t = getOtshThrshold(data);
	size_t i, j;
	const uchar *pSrc = NULL;
	uchar *pBin = NULL;
	for (i = 0; i < height; i++)
	{
		pSrc = data + i*width;
		pBin = m_Bin + i*width;
		for (j = 0; j < width; j++)
		{
			pBin[j] = pSrc[j]>t ? 255 : 0;
		}
	}
}

int CImageProc::getOtshThrshold(const uchar * data) const
{
	double hist[256] = { 0.0 };
	normalizedHist(data, hist);

	double omega[256];
	double mu[256];
	omega[0] = hist[0];
	mu[0] = 0;
	int i;
	for (i = 1; i < 256; i++)
	{
		omega[i] = omega[i - 1] + hist[i]; //累积分布函数
		mu[i] = mu[i - 1] + i * hist[i];
	}
	double mean = mu[255];// 灰度平均值
	double max = 0;
	double PA, PB, MA, MB, value;
	int k_max = 0;
	for (i = 1; i < 255; i++)
	{
		PA = omega[i]; // A类所占的比例
		PB = 1 - omega[i]; //B类所占的比例
		if (fabs(PA) > 0.001 && fabs(PB) > 0.001)
		{
			MA = mu[i] / PA; //A 类的灰度均值
			MB = (mean - mu[i]) / PB;//B类灰度均值
			value = PA * (MA - mean) * (MA - mean) + PB * (MB - mean) * (MB - mean);//类间方差
			if (value > max)
			{
				max = value;
				k_max = i;
			}
		}
	}

	return k_max;
}

bool CImageProc::normalizedHist(const uchar * data, double hist[256]) const
{
	size_t i, j;
	const uchar *p = NULL;
	for (i = 0; i < height; i++)
	{
		p = data + i*width;
		for (j = 0; j < width; j++)
		{
			hist[p[j]]++;
		}
	}
	int N = height * width;
	for (i = 0; i < 256; i++)
	{
		hist[i] = hist[i] / N;
	}
	return true;
}

void CImageProc::erode()
{
	uchar *_temp = new uchar[width*height];
	::memcpy(_temp, m_Bin, width*height*sizeof(uchar));
	int x, y, k;
	uchar *p = NULL;
	uchar *pCur = NULL;
	for (y = 1; y < height - 1; y++)
	{
		pCur = m_Bin + y*width;
		p = _temp + y*width;
		for (x = 1; x < width - 1; x++)
		{
			//跳过背景点的处理
			if (pCur[x])
			{
				continue;
			}
			//若前景点8-邻域内存在背景点，则将该前景点置为背景点
			for (k = 0; k < 8; k++)
			{
				if (m_Bin[x + dx[k] + (y + dy[k])*width])
				{
					p[x] = 255;
				}
			}
		}
	}
	::memcpy(m_Bin, _temp, width*height*sizeof(uchar));

	delete[] _temp;
}

void CImageProc::dilate(const int times)
{
	uchar *_temp = new uchar[width*height];
	int x, y, i, k, accum;
	::memcpy(_temp, m_Bin, width*height*sizeof(uchar));
	uchar *pCur = NULL;
	uchar *p = NULL;
	for (i = 0; i < times; i++)
	{
		for (y = 1; y < height - 1; y++)
		{
			pCur = m_Bin + y*width;
			p = _temp + y*width;
			for (x = 1; x < width - 1; x++)
			{
				//跳过前景点
				if (pCur[x] == 0)
				{
					continue;
				}
				accum = 0;
				for (k = 0; k < 8; k++)
				{
					if (m_Bin[x + dx[k] + (y + dy[k])*width] == 0)
					{
						if (++accum >= 4)
						{
							p[x] = 0;
							break;
						}
					}
				}
			}
		}
		::memcpy(m_Bin, _temp, width*height*sizeof(uchar));
	} // for (i = 0; i < times; i++)

	delete[] _temp;
}

void CImageProc::initMembers()
{
#if DEBUG
	IplImage *rawContours = cvCreateImage(cvSize(width, height), 8, 3);
#endif
	findContours();
	m_ContourCount = m_Contours.size();
#if DEBUG
	imshow("rawContouors", rawContours);
#endif
}

bool CImageProc::findContours(int threshold)
{
	bool *_grad = new bool[width*height];
	bool *p = NULL;
	uchar *pCur = NULL;
	uchar *pLow = NULL;
	uchar * pUp = NULL;
	size_t x, y;
	//扫描图像，求图像的梯度幅值的二值图
	for (y = 1; y < height - 1; y++)
	{
		pUp = m_Bin + width*(y - 1);
		pCur = m_Bin + width*y;
		pLow = m_Bin + width*(y + 1);
		p = _grad + y*width;
		for (x = 1; x < width - 1; x++)
		{
			//跳过前景点
			if (pCur[x] > 0)
			{
				p[x] = false;
				continue;
			}
			//若背像素点的4-邻域内存在像素值不同的两点，则该像素点一定为边缘点
			p[x] = ((pCur[x - 1] != pCur[x + 1]) || (pUp[x] != pLow[x]));
		}
	}

	stack<Point> seeds;
	Point seed;
	CContour contour;
	size_t k, curx, cury;
	//遍历梯度图幅值图，通过区域生长，获取闭合边缘
	for (y = 1; y < height - 1; y++)
	{
		p = _grad + y*width;
		for (x = 1; x < width - 1; x++)
		{
			//找到一个边缘种子点，入栈保存
			if (p[x])
			{
				contour.clear();
				seeds.push(Point(x, y));
				_grad[x] = false; //从梯度图中删除边缘点
				//以当前点为种子点，利用区域生长，获取整个闭合边缘
				while (!seeds.empty())
				{
					seed = seeds.top();
					seeds.pop();
					//保存边缘点
					contour.push_back(seed);
					//遍历种子点8-邻域
					for (k = 0; k < 8; k++)
					{
						curx = seed.x + dx[k];
						cury = seed.y + dy[k];
						if (curx >= width || curx < 0 || cury >= height || cury < 0)
						{
							continue;
						}
						if (_grad[cury*width + curx])
						{
							seeds.push(Point(curx, cury));	//新种子点入栈保存
							_grad[cury*width + curx] = false;	    //从梯度图中删去种子点
						}
					}
				} // while (!seeds.empty())
				//只保留较长的边缘
				if (contour.size() > threshold)
				{
					contour.calculate_members();
					m_Contours.push_back(contour);
				}
			} //if (p[x] > 0)
		} // for (x = 1; x < width - 1; x++)
	} // for (y = 1; y < height - 1; y++)

	delete[] _grad;
	return true;
}

void CImageProc::divide(double distanceThreshold)
{
#if DEBUG
	IplImage *test = cvCreateImage(cvSize(width, height), 8, 3);
		vector<Point> corners;
#endif
	int i;
	int indexThreshold = INT_MAX;
	//找到所有轮廓中最短的轮廓
	for (i = 1; i < (int)m_ContourCount; i++)
	{
		if (m_Contours[i].type() != CONTOUR)
		{
			continue;
		}

		if (indexThreshold > m_Contours[i].size())
		{
			indexThreshold = m_Contours[i].size();
		}
	}
	//以最短轮廓作为下标阈值条件
	indexThreshold *= 0.8;

	vector<int> index;
	CContour newContour;
	double minDistance, dist;
	int gap, startIndex, endIndex, j, k;
	//遍历所有轮廓，对每条轮廓求角点，并通过角点数以及角点之间的空间距离和下标距离关系判断是否存在粘连
	//存在粘连则进行裁剪，否则不做任何处理
	for (i = 1; i < (int)m_ContourCount; i++)
	{
		if (m_Contours[i].type() != CONTOUR)
		{
			continue;
		}
		CContour & contour = m_Contours[i];
		if (contour.size() <= 0) continue;
		//寻找轮廓中所有符合阈值条件的角点
		index.clear();
		binSUSAN(contour.points(), index, 15, 1, false, 4);
		//若角点数大于2，则对角点进行两两匹配，寻找满足阈值条件的最优角点对
		//(最优角点对：角点间下标距离 > indexThreshold && 空间距离 < distanceThreshold 情况下空间距离最近的角点对)
		//若存在最优角点对，则尽心边缘裁剪，否则不做处理
		if (index.size() <= 1)
		{
			continue;
		}
		minDistance = DBL_MAX;
		startIndex = endIndex = -1;
		for (j = 0; j < index.size() - 1; j++)
		{
			for (k = j + 1; k < index.size(); k++)
			{
				dist = distance(contour[index[j]], contour[index[k]]);
				gap = abs(index[j] - index[k]);
				//找到距离和下标满足阈值条件且空间距离最近的角点对
				if ((dist < distanceThreshold) && (gap > indexThreshold) && ((m_Contours[i].size() - gap) > indexThreshold))
				{
					if (dist < minDistance)
					{
						minDistance = dist;
						startIndex = index[j];
						endIndex = index[k];
					}
					break;
				}
			}
		} // 对角点进行过滤
		//若startIndex下标为初始值，则表示不存在粘连，结束本次循环，开始处理下一条边缘
		if ((startIndex <= -1) || (endIndex >= contour.size()))
			continue;
		//处理粘连情况
		//保证startIndex < endIndex
		k = startIndex;
		startIndex = k < endIndex ? k : endIndex;
		endIndex = k > endIndex ? k : endIndex;
		newContour.clear();
		//从当前边缘点集中，删除角点对下标之间的点，并将之作为新轮廓添加到m_Contours集合中
		for (k = endIndex; k >= startIndex; k--)
		{
			newContour.push_back(contour[k]);
		}
		contour.erase_multiple(startIndex, endIndex);
		//更新类成员数据
		contour.calculate_members();
		newContour.calculate_members();
		m_Contours.push_back(newContour);
	}

	m_ContourCount = m_Contours.size();

#if DEBUG
	for (i = 1; i < m_Contours.size(); i++)
	{
		drawPoints(test, m_Contours[i].points(), cvScalar(255, 255, 255));
		drawPoint(test, m_Contours[i].center() , cvScalar(255, 0, 0), 8);
		drawBox(test, m_Contours[i].box(), cvScalar(0, 0, 255), 2);
		drawPoints(test, corners, cvScalar(0, 255, 0), 4);
	}
	imshow("divide", test);
	cvReleaseImage(&test);
#endif
}

////************************************************公有成员函数声明*************************************************////
void CImageProc::run()
{
	if (m_Contours.size() <= 1) return;
	
	//处理粘连情况
	divide();

	//找螺母
	findNut();

	//找硬币和螺母
	findCoin();
	
	//找L型扳手
	findHexKey();

	//找螺钉
	findScrew();
}

void CImageProc::writeResult(uchar * result, int channels, int _count[5])
{
	int widthStep = width*channels;
	size_t i, j, k, c, count;
	int x, y, index;
	Vector3 color;
	::memset(_count, 0, 5 * sizeof(int));
	for (i = 0; i < m_ContourCount; i++)
	{
		_count[(int)m_Contours[i].type()]++;
		color = COLOR_TABLE[(int)m_Contours[i].type()];
		const vector<Point> & contour = m_Contours[i].points();
		count = contour.size();
		for (j = 0; j < count; j++)
		{
			for (k = 0; k < 8; k += 2)
			{
				x = contour[j].x + dx[k];
				y = contour[j].y + dy[k];
				if (y < 0 || y >= height || x < 0 || x >= width)
				{
					continue;
				}
				index = y*widthStep + x*channels;
				for (c = 0; c < channels; c++)
				{
					result[index + c] = color[c];
				}
			}
		}
	}
}

void CImageProc::writeContour()
{
	::memset(m_Bin, 0, width*height*sizeof(uchar));
	size_t i, j, count;
	int x, y;
	uchar value = 0;
	for (i = 0; i < m_ContourCount; i++)
	{
		value = m_Contours[i].type();
		const vector<Point> & contour = m_Contours[i].points();
		count = contour.size();
		for (j = 0; j < count; j++)
		{
			x = contour[j].x;
			y = contour[j].y;
			m_Bin[y*width + x] = value;
		}
	}
}

vector<EContourType> CImageProc::getContourType() const
{
	vector<EContourType> contourType;
	for (int i = 0; i < (int)m_ContourCount; i++)
	{
		contourType.push_back(m_Contours[i].type());
	}
	return contourType;
}

vector<vector<Point>> CImageProc::getContour() const
{
	vector< vector<Point> > contours;
	for (int i = 0; i < (int)m_ContourCount; i++)
	{
		contours.push_back(m_Contours[i].points());
	}
	return contours;
}

vector<Box> CImageProc::getBox() const
{
	vector<Box> boxes;
	for (int i = 0; i < (int)m_ContourCount; i++)
	{
		boxes.push_back(m_Contours[i].box());
	}
	return boxes;
}

void CImageProc::findNut()
{
	//根据螺母内外边缘中心点重合的特征作为判据
	int i, j, k, x, y, accum;
	Point center;
	Point t(5, 5);
	bool flag = false;
	double temp;
	double lower = 0.7;
	double upper = 1.3;
	for (i = 1; i < m_ContourCount; i++)
	{
		if (m_Contours[i].type() != CONTOUR)
		{
			continue;	//只处理未分类的边缘
		}
		center = m_Contours[i].center();
		flag = false;
		for (j = i + 1; j < m_ContourCount; j++)
		{
			if (center.absDiff(m_Contours[j].center()) < t)
			{
				flag = true;
				//删去较短的内边缘
				if (m_Contours[i].size() > m_Contours[j].size())
				{
					m_Contours[i].setType(NUT);
					m_Contours.erase(m_Contours.begin() + j);
					m_ContourCount--;
				}
				else
				{
					m_Contours[j].setType(NUT);
					m_Contours.erase(m_Contours.begin() + i--);
					m_ContourCount--;
				}
				break;
			}
		}
		if (!flag)
		{
			//以外接矩形长宽比和中心像素点值为判据，再次进行判断
			temp = (double)m_Contours[i].box().width / m_Contours[i].box().height;
			if (lower < temp && temp < upper)
			{
				//计算边缘中心点及其邻域内像素值
				accum = 0;
				for (k = 0; k < 9; k++)
				{
					x = center.x + dx[k];
					y = center.y + dy[k];
					if (m_Bin[y*width + x])
					{
						accum++;
					}
				}
				if (accum >= 4)
				{
					m_Contours[i].setType(NUT);
				}
			}
		} // 	if(!flag)
	}
}

bool CImageProc::findCoin()
{
	size_t i, k;
	double lower = 0.7;
	double upper = 1.3;
	double temp = 0.0;
	//首先根据包围边缘的最大外接矩形的长宽比筛选出候选边缘
	for (i = 1; i < m_ContourCount; i++)
	{
		if (m_Contours[i].type() != CONTOUR)
		{
			continue;	//只处理未分类的边缘
		}
		temp = (double)m_Contours[i].box().width / m_Contours[i].box().height;
		if (lower < temp && temp < upper)
		{
			m_Contours[i].setType(COIN);
		}
	}
	//计算轮廓中心点到轮廓点的平均距离和最大、最小距离，比较最大-最小距离之差与平均距离的差值，差值过大，则剔除该边缘
	//剔除所有不符合条件的轮廓之后，剩余的轮廓即为属于圆的轮廓
	size_t count;
	Point center;
	double average, segma;
	vector<double> dist;
	for (i = 1; i < m_ContourCount; i++)
	{
		if (COIN == m_Contours[i].type())
		{
			const vector<Point> & contour = m_Contours[i].points();
			count = contour.size();
			if (count <= 0)
			{
				m_Contours[i].setType(CONTOUR);
				continue;
			}
			center = m_Contours[i].center();
			dist.clear();
			//计算轮廓点到中心点的方差
			average = 0.0;
			for (k = 0; k < count; k++)
			{
				temp = distance(center, contour[k]);
				dist.push_back(temp);
				average += temp;
			}
			average /= count;
			segma = 0.0;
			for (k = 0; k < dist.size(); k++)
			{
				segma += ((dist[k] - average)*(dist[k] - average));
			}
			segma /= (count - 1.0);
			if (segma > 10.0)
			{
				m_Contours[i].setType(CONTOUR);
			}
		} // if (COIN == m_ContourType[i])
	} // for (i = 1; i < m_ContourCount; i++)
	
	return true;
}

bool CImageProc::findHexKey()
{
	size_t i;
	for (i = 1; i < m_ContourCount; i++)
	{
		if (m_Contours[i].type() != CONTOUR)
		{
			continue;
		}
		if (isOutContour(i))
		{
			m_Contours[i].setType(HEX_KEY);
		}
	}
	return true;
}

bool CImageProc::findScrew()
{
	size_t i, t = 3;
	vector<int> index;
	for (i = 1; i < m_ContourCount; i++)
	{
		if (m_Contours[i].type() != CONTOUR)
		{
			continue;
		}
		//通过检测角点数量判断是否属于螺钉
		index.clear();
		binSUSAN(m_Contours[i].points(), index, 20);
		if (index.size() >= t)
		{
			m_Contours[i].setType(SCREW);
		}
	}

	return true;
}

//////************************************************私有成员函数定义**************************************************////

/////**************findHexKey函数及相关辅助函数******************//////
bool CImageProc::isOutContour(const int index)
{
	const vector<Point> & contour = m_Contours[index].points();
	int x = m_Contours[index].center().x;
	int y = m_Contours[index].center().y;

	//通过判断该点同行、同列是否被边缘点包围来确定该点是否是轮廓外的点
	//若不被包围，则一定是轮廓外的点，否则可能是轮廓内的点
	vector<int> coly;	//记录同一列的y坐标
	vector<int> rowx;	//记录同一行的x坐标
	size_t i, count = contour.size();
	int temp1, temp2;
	//遍历轮廓点，找到与point同行同列的轮廓点，并保存其对应的x或y坐标
	for (i = 0; i < count; i++)
	{
		temp1 = contour[i].x;
		temp2 = contour[i].y;
		//找同一列的点的y坐标
		if (abs(temp1 - x) <= 1)
			coly.push_back(temp2);
		//找同一行的x坐标
		if (abs(temp2 - y) <= 1)
			rowx.push_back(temp1);
	}
	//判断point在水平方向上是否被包围
	temp1 = temp2 = 0;
	for (i = 0; i < rowx.size(); i++)
	{
		if (rowx[i] > x)
			temp1++;
		else
			temp2++;
	}
	if (!temp1 || !temp2)
	{
		return true;	//同一行上只有一边有轮廓点，则说明该点在水平方向上不被包围
	}
	//判断point在垂直方向上是否被包围
	temp1 = temp2 = 0;
	for (i = 0; i < coly.size(); i++)
	{
		if (coly[i] > y)
			temp1++;
		else
			temp2++;
	}
	if (!temp1 || !temp2)
	{
		return true; //同一列上只有一边有轮廓点，则说明该点在垂直方向上不被包围
	}
	return false;
}

/////***************findScrew函数及相关辅助函数******************//////
bool CImageProc::binSUSAN(const vector<Point> & contour, vector<int> & _index, int upper, int lower, bool method, int r) const
{
	if (lower <= 0) lower = 1;
	int x, y, same;
	size_t i, j, k, contourSize = contour.size();
	int *_susan = new int[contourSize];
	::memset(_susan, 0, contourSize*sizeof(int));
	uchar curGray;
	//计算每个轮廓点出的SUSAN算子值,确定初始角点值
	for (i = 0; i < contourSize; i++)
	{
		x = contour[i].x;
		y = contour[i].y;

		//忽略边界像素的处理
		if (x<r || x>(width - r) || y<r || y>(height - r))
		{
			continue;
		}

		//计算当前点SUSAN算子值
		curGray = method ? m_Bin[y*width + x] : 255 - m_Bin[y*width + x];
		same = 0;

		for (j = x - r; j < x + r; j++)
		{
			for (k = y - r; k < y + r; k++)
			{
				double dist = (j - x) * (j - x) + (k - y) * (k - y);
				if (dist > r * r)
					continue;
				if (m_Bin[k*width + j] == curGray)
					same++;
			}
		}
		if (lower <= same && same <= upper)
		{
			_susan[i] = same;
		}
	}

	//计算局部极小值，并将该点作为角点保存
	int index, min;
	for (i = 0; i < contourSize; i++)
	{
		if (_susan[i] == 0)
		{
			continue;
		}
		//找到局部极小值
		min = _susan[i];
		index = i;
		for (i++; i < contourSize; i++)
		{
			if (_susan[i] == 0)
			{
				break;
			}
			if (_susan[i] < min)
			{
				min = _susan[i];
				index = i;
			}
		}
		_index.push_back(i);
	}

	delete[] _susan;
	return true;
}
