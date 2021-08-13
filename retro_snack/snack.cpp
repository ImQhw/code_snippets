#include "snack.h"
namespace SNACK
{
	//map类//
	//构造函数//
	Map::Map()
	{
		for (int i = 0; i < MAP_SIZE; i++)
		{
			map[0][i] = '-';
			map[MAP_SIZE - 1][i] = '-';
		}
		for (int i = 0; i < MAP_SIZE; i++)
		{
			map[i][0] = '|';
			map[i][MAP_SIZE - 1] = '|';
		}
		//初始化地图//
		for (int i = 1; i < MAP_SIZE - 1; i++)
			for (int j = 1; j < MAP_SIZE - 1; j++)
				map[i][j] = ' ';
	}
	//打印地图//
	void Map::MapPrint()
	{
		cout << "\t\t\t\t控制台下的贪吃蛇游戏\n";
		for (int i = 0; i < MAP_SIZE; i++)
		{
			for (int j = 0; j < MAP_SIZE; j++)
				cout << map[i][j];
			cout << endl;
		}
	}
	//清除地图//
	void Map::MapClean()
	{
		for (int i = 0; i < MAP_SIZE; i++)
		{
			map[0][i] = '-';
			map[MAP_SIZE - 1][i] = '-';
		}
		for (int i = 0; i < MAP_SIZE; i++)
		{
			map[i][0] = '|';
			map[i][MAP_SIZE - 1] = '|';
		}
		//初始化地图//
		for (int i = 1; i < MAP_SIZE - 1; i++)
			for (int j = 1; j < MAP_SIZE - 1; j++)
				map[i][j] = ' ';
	}

	//snack类//
	//读取用户操作//
	void Snack::readKey()
	{
		//延迟一段时间，等待按键按下//
		int keyValue;
		std::clock_t start;
		std::clock_t finish = start = clock();
		while ((double)(finish - start) / CLOCKS_PER_SEC < 1.0 / moveSpeed)
		{
			if (_kbhit())
			{
				keyValue = _getch();
				if (keyValue == 224)
				{
					keyValue = _getch();
					switch (keyValue)
					{
					case(72) : dirction = UP;    return; //上
					case(80) : dirction = DOWN;  return; //下
					case(75) : dirction = LEFT;  return; //左
					case(77) : dirction = RIGHT; return; //右
					default  : break;	//键入的是其他字符
					}
				}
				//键入Esc键//
				if (keyValue == 27)
				{
					gameOver = true;
					return;
				}
			}
			finish = clock();
		}
	}
	//蛇身移动函数//
	void Snack::snackMove()
	{
		switch (dirction)
		{
		case(UP)    : snackHead.x -= 1; break;
		case(DOWN)  : snackHead.x += 1; break;
		case(LEFT)  : snackHead.y -= 1; break;
		case(RIGHT) : snackHead.y += 1; break;
		}

		//判断游戏是否结束//
		//撞到了边界//
		if (snackHead.y == MAP_SIZE - 1 || snackHead.y == 0 || 
			snackHead.x == MAP_SIZE - 1 || snackHead.x == 0)
		{
			gameOver = true;
			return;
		}
		//撞在自己的身体//
		for (auto p = ++snack.begin(); p != snack.end(); p++)
			if (p->x == snackHead.x && p->y == snackHead.y)
			{
				gameOver = true;
				return;
			}

		//将蛇头蛇身全部标志都重置为'*'//
		snack.begin()->type = '*';
		//翻转snack，使得后面的遍历snack操作更方便//
		snack.reverse();
		std::list<Node>::iterator p = snack.begin();
		std::list<Node>::iterator pnext = ++snack.begin();

		//遍历snack，蛇身的每一个节点依次赋值为后一个节点的值//
		while (pnext != snack.end())
			*p++ = *pnext++;
		*p = snackHead;
		snack.reverse();
	}
	//地图更新函数//
	void Snack::snackMapUpdata()
	{
		//清空地图//
		snackMap.MapClean();

		//想地图内蛇身坐标处写入蛇身标志//
		for (auto p = snack.begin(); p != snack.end(); p++)
		{
			snackMap.MapUpdata(*p);
		}
		//将食物写入坐标//
		snackMap.MapUpdata(food);
	}
	//生成食物//
	void Snack::snackGenerateFood()
	{
		//初始化随机数种子//
		srand((unsigned int)time(NULL));
		//产生随机数//
		int x = 0;
		int y = 0;
		while (1)
		{
			x = rand() % (MAP_SIZE - 1);
			y = rand() % (MAP_SIZE - 1);
			//便利蛇身，查看食物坐标是否与蛇身重合//
			bool flag = false;
			for (auto p = snack.begin(); p != snack.end(); p++)
				//产生的食物与蛇身重合
				if (x == p->x && y == p->y)
				{
					flag = true;
					break;
				}
			//是否与边界重合//
			if (x == 0 || y == 0) flag = true;
			//都不重合//
			if (!flag)
			{
				food.x = x;
				food.y = y;
				return;
			}
		}
	}
	//吃食物//
	bool Snack::snackEatFood()
	{
		Node newNode;
		//蛇吃掉了食物//
		if (snackHead.x == food.x && snackHead.y == food.y)
		{
			//根据蛇身后两个节点，在蛇尾添加新节点//
			snack.reverse();
			auto p1 = snack.begin();
			auto p2 = ++snack.begin();

			//后两个节点位于同一行//
			if(p1->x == p2->x)
				//判断最后一个节点相对于倒数第二个节点的位置关系//
				//左边//
				if (p1->y < p2->y)
					//新节点位于最后一个节点左边//
				{
					Node temp(p1->x, p1->y - 1);
					newNode = temp;
				}
				//右边//
				else
					//新节点位于最后一个节点右边//
				{
					Node temp(p1->x, p1->y + 1);
					newNode = temp;
				}
			//后两个节点位于同一列//
			else if(p1->y == p2->y)
			{
				//判断最后一个节点相对于倒数第二个节点的位置关系//
				//上面//
				if (p1->x < p2->x)
					//新节点位于最后一个节点的上面//
				{
					Node temp(p1->x - 1, p1->y);
					newNode = temp;
				}
				//下面//
				else
					//新节点位于最后一个节点的下面//
				{
					Node temp(p1->x + 1, p1->y);
					newNode = temp;
				}
			}
			snack.push_front(newNode);
			snack.reverse();
			return true;
		}
		return false;
	}
	//游戏进程控制函数//
	void Snack::snackShow()
	{
		while (1)
		{
			//清屏//
			std::system("cls");
			//游戏结束//
			if (gameOver)
			{
				cout << "游戏结束" << endl;
				return;
			}
			//蛇吃掉了食物//
			if (snackEatFood())
				//产生食物//
				snackGenerateFood();
			//更新地图//
			snackMapUpdata();
			//打印地图//
			snackMap.MapPrint();
			//读取用户的方向//
			readKey();
			//移动//
			snackMove();
		}
	}

	//测试函数//
	void Snack::TestsnackShow()
	{
		for (auto p = snack.begin(); p != snack.end(); p++)
			std::cout << *p << endl;
	}
}