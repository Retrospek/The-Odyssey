#include <utility>
#include <algorithm>

template <typename T, typename K>
class DoublyLinkedList
{
private:
    struct node
    {
        T _data; // surely follows some RAII guidelines
        node *_next = nullptr;
        node *_prev = nullptr;

        node(const T &data) : _data(data) {}

        ~node() {}
    };

    size_t _capacity;
    size_t _size{0};

    node *_head = nullptr;
    node *_tail = nullptr;

public:
    Doubly(const size_t &capacity) : _capacity(capacity) {}
    ~Double() {}

    void insert(const size_t &index, T data)
    {
    }

    void push_front(const T &data)
    {
    }

    void push_back(const T &data)
    {
    }

    T pop_front()
    {
    }

    T pop_back()
    {
    }

    void clear()
    {
    }

    void reverse()
    {
    }

    std::string &operator<<()
    {
    }
};