// Least-Recent-Used cachea algorithm

#include <iostream>
#include <unordered_map>

using std::cout;
using std::endl;

class LRUCache
{
private:
    class DLinkedListNode
    {

    public:
        int key;
        int value;
        DLinkedListNode *pre;
        DLinkedListNode *next;

        DLinkedListNode(int key, int value,
                        DLinkedListNode *pre = nullptr,
                        DLinkedListNode *next = nullptr)
            : key(key), value(value), pre(pre), next(next)
        {
            ;
        }
    };

public:
    LRUCache(int capacity) : capacity(capacity)
    {
        cached.clear();

        head = new DLinkedListNode(0, 0);
        tail = new DLinkedListNode(0, 0);

        head->next = tail;
        tail->pre = head;
    }

    virtual ~LRUCache()
    {
        while (head)
        {
            auto temp = head;
            head = head->next;

            delete temp;
        }

        head = nullptr;
        tail = nullptr;
        cached.clear();
    }

    // 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1
    int get(int key)
    {
        auto p = cached.find(key);

        // cache miss
        if (p == cached.end())
        {
            cout << "cache miss: " << key << endl;

            return -1;
        }

        // cache hit
        auto node = p->second;
        moveNodeToFirst(node);

        cout << "cache hit: " << key << ": " << node->value << endl;

        return node->value;
    }

    /* 如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组「关键字-值」。
       当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。
    **/
    void put(int key, int value)
    {
        auto p = cached.find(key);

        // cache hit
        if (p != cached.end())
        {
            auto node = p->second;
            node->value = value;
            moveNodeToFirst(node);
        }

        // cache miss
        else
        {
            auto node = new DLinkedListNode(key, value);
            insertNodeToFirst(node);

            // add to cache
            cached[key] = node;

            // remove last of list
            if (cached.size() > capacity)
                removeLastNode();
        }
    }

    size_t size() const
    {
        return cached.size();
    }

private:
    void removeNodeFromList(DLinkedListNode *node)
    {
        // ignore head and tail node
        if (node == head || node == tail)
            return;

        node->pre->next = node->next;
        node->next->pre = node->pre;
    }

    void insertNodeToFirst(DLinkedListNode *node)
    {
        if (node == head || node == tail)
            return;

        node->next = head->next;
        node->pre = head;

        head->next->pre = node;
        head->next = node;
    }

    void moveNodeToFirst(DLinkedListNode *node)
    {
        removeNodeFromList(node);
        insertNodeToFirst(node);
    }

    void removeLastNode()
    {
        auto node = tail->pre;

        // empty list
        if (node == head)
            return;

        removeNodeFromList(node);

        // remove from cache
        cached.erase(node->key);

        delete node;
    }

private:
    int capacity;
    DLinkedListNode *head;
    DLinkedListNode *tail;

    std::unordered_map<int, DLinkedListNode *> cached;
};

int main()
{
    LRUCache lru(10);

    cout << lru.get(10) << endl;
    lru.put(10, 12);
    cout << lru.get(10) << ", " << lru.size() << endl;
    lru.put(10, 13);
    cout << lru.get(10) << ", " << lru.size() << endl;

    return 0;
}
