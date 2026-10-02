#include <utility>
#include <algorithm>
#include <exception>
template <typename T>
class DoublyLinkedList
{
private:
    struct node
    {
        T _data; // surely follows some RAII guidelines
        node *_next = nullptr;
        node *_prev = nullptr;

        node(const T &data) : _data(data) {}
    };

    size_t _size{0};

    node *_head = nullptr;
    node *_tail = nullptr;

public:
    DoublyLinkedList() = default;
    ~DoublyLinkedList()
    {
        clear();
    }

    void insert(size_t index, const T &data)
    {
        if (index > _size)
            throw std::out_of_range("Index out of bounds");

        if (index == 0)
        {
            push_back(data);
            return;
        }

        if (index == _size)
        {
            push_front(data);
            return;
        }

        node *curr = _tail;
        for (size_t i = 0; i < index; ++i)
        {
            curr = curr->_next;
        }

        node *NEW = new node(data);
        node *prev_node = curr->_prev;

        NEW->_next = curr;
        NEW->_prev = prev_node;

        prev_node->_next = NEW;
        curr->_prev = NEW;

        ++_size;
    }

    void push_front(const T &data)
    {
        node *NEW = new node(data);
        if (_size == 0)
        {
            _head = NEW;
            _tail = NEW;
        }
        else
        {
            NEW->_prev = _head;
            _head->_next = NEW;
            _head = NEW;
        }
        ++_size;
    }

    void push_back(const T &data)
    {
        node *NEW = new node(data);
        if (_size == 0)
        {
            _head = NEW;
            _tail = NEW;
        }
        else
        {
            NEW->_next = _tail;
            _tail->_prev = NEW;
            _tail = NEW;
        }
        ++_size;
    }

    T pop_front()
    {
        if (!_head)
        {
            throw std::out_of_range("List Empty => pop_front()");
        }

        node *temp = _head;
        T data = temp->_data;

        _head = _head->_prev;
        if (_head)
        {
            _head->_next = nullptr;
        }
        else
        {
            _tail = nullptr;
        }

        delete temp;
        --_size;
        return data;
    }

    T pop_back()
    {
        if (!_tail)
        {
            throw std::out_of_range("List Empty => pop_back()");
        }

        node *temp = _tail;
        T data = temp->_data;

        _tail = _tail->_next;
        if (_tail)
        {
            _tail->_prev = nullptr;
        }
        else
        {
            _head = nullptr;
        }

        delete temp;
        --_size;
        return data;
    }

    void clear()
    {
        node *curr = _tail;
        while (curr)
        {
            node *temp = curr->_next;
            delete curr;
            curr = temp;
        }

        _head = nullptr;
        _tail = nullptr;
        _size = 0;
    }

    void reverse()
    {
        node *curr = _tail;
        for (size_t i{0}; i < _size; ++i)
        {
            std::swap(curr->_next, curr->_prev);
            curr = curr->_prev;
        }

        std::swap(_head, _tail);
    }

    template <typename U>
    friend std::ostream &operator<<(std::ostream &os, const DoublyLinkedList<U> &list)
    {
        auto *curr = list._tail;
        os << "[";
        while (curr)
        {
            os << curr->_data << (curr->_next ? " -> " : "");
            curr = curr->_next;
        }
        os << "]";
        return os;
    }
};