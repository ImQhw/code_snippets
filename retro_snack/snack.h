//snack类和map类的定义//

#ifndef _SNACK_H_
#define _SNACK_H_

#include <Windows.h>
#include <iostream>
#include <ctime>
#include <conio.h>
#include <cstdlib>
#include <list>

namespace SNACK
{
	using std::cout;
	using std::endl;
	using std::cin;
#define MAP_SIZE  22
	//节点定义//
	struct Node
	{
		int x;
		int y;
		char type;

		//构造函数//
		Node(int x = 0, int y = 0, char t = '*')
		{
			this->x = x; this->y = y; type = t;
		}

		//重载赋值运算符//
		Node & operator=(const Node &n)
		{
			this->x = n.x;
			this->y = n.y;
			this->type = n.type;
			return *this;
		}

		//重载<<运算符//
		friend std::ostream & operator<<(std::ostream & os, const Node &n)
		{
			os << "(" << n.x << "," << n.y << "," << n.type << ")" << "\t";
			return os;
		}
	};

	//地图类//
	class Map
	{
	private:
		//存放地图的数组//
		char map[MAP_SIZE][MAP_SIZE];

	public:
		Map();
		~Map() {}
		void MapPrint();
		//更新地图指定点的值//
		void MapUpdata(const Node &n)
		{
			map[n.x][n.y] = n.type;
		}
		//清除地图//
		void MapClean();
	};

	//snack类//
	class Snack
	{
	public:
		enum Direction { UP, DOWN, LEFT, RIGHT };
		enum MoveSpeed { LOW = 3, MIDDLE, HIGHT };
	private:
		//实例化一个Node list 容器类，用于存放snack身体的坐标//
		std::list<Node> snack;

		//存放地图的成员对象//
		Map snackMap;

		//存放蛇头的信息//
		Node snackHead;
		Direction dirction;
		MoveSpeed moveSpeed;
		
		//游戏是否结束标志//
		bool  gameOver = false;

		//食物的坐标//
		Node food;

	public:
		Snack()
		{
			Node temp(1, 4, '#');
			snackHead = temp;
			snack.push_back(snackHead);
			snackMap.MapUpdata(snackHead);
			for (int i = 1; i <= 3; i++)
			{
				temp.y = snackHead.y - i;
				temp.type = '*';
				snack.push_back(temp);
				snackMap.MapUpdata(temp);
			}
			temp.x = MAP_SIZE/2;
			temp.y = MAP_SIZE/2;
			temp.type = '*';
			food = temp;
			dirction = RIGHT;
			moveSpeed = LOW;
		}
		~Snack() {}
		void readKey();
		void snackMove(void);
		void snackMapUpdata();
		void snackShow(void);
		bool snackGameOver() { return gameOver; }
		void snackGenerateFood();
		bool snackEatFood();

		//测试成员函数//
		void TestsnackShow();
		void TestsnackDirection() { cout << dirction << endl; }
	};
}
#endif
