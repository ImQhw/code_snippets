#include "ccontour.h"

CContour::CContour(const vector<Point>& points, const EContourType type)
{
	m_Size = points.size();
	m_Type = type;
	for (size_t i = 0; i < m_Size; i++)
	{
		m_Points.push_back(points[i]);
	}
	calculate_members();
}

bool CContour::erase_safe(const int index)
{
	if (index < 0 || index >= m_Size)
	{
		fprintf(stderr, "Contour::erase_safe 操作失败！下标越界！");
		return false;
	}
	m_Points.erase(m_Points.begin() + index);
	m_Size--;

	return true;
}

bool CContour::erase_multiple(const int startIndex, const int endIndex)
{
	if (startIndex < 0 || endIndex >= m_Size)
	{
		fprintf(stderr, "Contour::erase_multiple 操作失败！下标越界！");
		return false;
	}
	vector<Point>::iterator start = m_Points.begin() + startIndex;
	vector<Point>::iterator end = m_Points.begin() + endIndex + 1;
	m_Points.erase(start, end);
	m_Size -= (abs(endIndex - startIndex) + 1);
	vector<Point>(m_Points).swap(m_Points);	//调整容量
	return true;
}

bool CContour::calculate_members()
{
	if (m_Size <= 0) 
		return false;
	int minx = INT_MAX, miny = INT_MAX, maxx = INT_MIN, maxy = INT_MIN, curx, cury, x = 0, y = 0;
	for (size_t i = 0; i < m_Size; i++)
	{
		curx = m_Points[i].x;
		cury = m_Points[i].y;
		if (minx > curx)	minx = curx;
		if (maxx < curx)	maxx = curx;
		if (miny > cury)	miny = cury;
		if (maxy < cury)	maxy = cury;
		x += curx;
		y += cury;
	}

	m_Center = Point((double)x / m_Size + 0.5, (double)y / m_Size + 0.5);
	m_Box = Box(minx, miny, maxx, maxy);
}
