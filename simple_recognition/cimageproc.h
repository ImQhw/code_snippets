//自定义图片处理函数，只能处理单通道灰度图像
//
//
#ifndef _CIMAGE_PROC_H_
#define _CIMAGE_PROC_H_

#define MAX_CLASS_COUNT 5

#include "mydefine.h"
#include "ccontour.h"
#include <vector>
#include <stack>

using namespace std;

class CImageProc
{
public:
	typedef unsigned char uchar;
private:
	uchar *m_Bin;
	const int width;
	const int height;
	vector<CContour> m_Contours;		//边缘点
	size_t           m_ContourCount;
	
	//静态成员变量
	static int dx[9];
	static int dy[9];
	static Vector3 COLOR_TABLE[MAX_CLASS_COUNT];	//颜色表

public:
	////************************************************构造、析构函数声明*************************************************////
	//@param imageData : 入口图片数据
	CImageProc(const int width, const int height, const int channels, const uchar * imageData);
	~CImageProc();

private:
	////********************************************初始化函数声明*************************************************////
	//RGB图转灰度图
	//@param color : 入口参数，传入彩色图片数据
	//@param gray  : 出口参数，返回转化后的灰度图数据
	void cvtRGB2Gray(const uchar *color, uchar * gray)const; //cbc

															 //采用OTSU方法，对原图数据二值化后保存到bin成员中
															 //@param data : 入口参数，灰度图数据
	void otsuCut(const uchar *data);

	//计算OTSU分割的阈值，通过返回值返回
	//@param data : 入口参数，灰度图数据
	//@return : 返回值，返回阈值
	int getOtshThrshold(const uchar * data) const;

	//计算灰度直方图
	//@param data : 入口参数，灰度图数据
	//@param hist[256] : 出口参数，返回灰度直方图
	bool normalizedHist(const uchar * data, double hist[256]) const;

	//二值图像的腐蚀，会改变m_Bin成员的值
	void erode();

	//二值图像膨胀
	//@param times: 入口参数，膨胀迭代次数
	void dilate(const int times = 1);

	//初始化成员变量，准备好识别需要的数据，会改变所有非常量类成员变量
	void initMembers();

	//在二值图中寻找闭合边缘，并保存
	//@param threshold: 入口参数，边缘点数小于该参数的边缘被舍弃
	bool findContours(int threshold = 30);

	//判断是否存在简单粘连情况，存在则进行处理，否则不做任何处理
	//@param distanceThreshold : 若轮廓上角点之间的空间距离阈值
	void divide(double distanceThreshold = 30.0);

	////************************************************公有成员函数声明*************************************************////
public:
	//调用子函数，对m_Contour中的边缘进行分类
	void run();

	//将分类后的结果，写入参数中result中
	//@param result : 出口参数，所有边缘点按类别不同，被赋值为不同的颜色
	//@param channels : 入口参数，result参数代表的图片的通道数
	//@param count[5] : 出口参数，返回当前图片中每一类的数目
	void writeResult(uchar * result, int channels, int count[5]);

	//将m_Contours中的边缘画在m_Bin上
	void writeContour();

	//返回所有轮廓的外接矩形
	vector<Box> getBox() const ;

	//返回所有轮廓的类型
	vector<EContourType> getContourType() const ;

	//返回所有轮廓
	vector< vector<Point> > getContour() const;

	//获取m_Bin成员
	const uchar * getBin() const
	{
		return m_Bin;
	}

	////************************************************私有成员函数声明**************************************************////
private:

	//寻找螺母
	//通过判断两条边缘中心点重合或外接矩形满足长宽比且边缘中心点及其8-邻域都为背景点
	void findNut();	//cbc

	//寻找硬币
	//通过外接矩形长宽比得到硬币的候选轮廓，再通过判断轮廓点到中心点最大-最小距离以及平均距离关系，找出真正的硬币
	bool findCoin();	//cbc

	//寻找L形六角扳手
	//原理：要识别的目标中，除L形六角扳手外，其余目标边缘都为凸多边形，故其边缘点中心都位于边缘内
	//     L形六角扳手边缘为凹多边形，故其边缘中心位于轮廓外
	bool findHexKey();

	//寻找螺钉
	//通过判断角点数是否大于阈值判定是够为螺钉
	bool findScrew();

	//判断点是否位于轮廓外，是返回true，否则返回false
	//@param index : 入口参数，要处理的边缘的下标
	bool isOutContour(const int index);

	//二值图像的SUSAN角点检测算法
	//@param contour : 入口参数，边缘点的集合
	//@parame index : 出口参数，返回每个角点在contour参数中的下标
	//@param upper : 入口参数，上界
	//@param lower : 入口参数，下界
	//@param method : 入口参数，true: 寻找核同值区域; false: 寻找核不同值区域
	//@param radius : 入口参数，模板半径
	bool binSUSAN(const vector<Point> & contour, vector<int> & index, int upper, int lower = 1, bool method = true, int radius = 4) const; //cbc

	//函数功能：
	//		计算a、b两点的距离
	inline double distance(const Point & a, const Point & b)
	{
		return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
	}
	//计算点到直线的距离，直线方程为y=k*x+b
	inline double distance(const Point & a, const double & k, const double & b) const
	{
		return abs(k*a.x - a.y + b) / sqrt(k*k + 1);
	}
};
#endif
