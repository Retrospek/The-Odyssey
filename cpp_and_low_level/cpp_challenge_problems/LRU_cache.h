#include <set>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <stdexcept>

template <typename K, typename V>
class lru_cache
{
private:
    struct node
    {
        K _key;
        V _data;
        node *_next = nullptr;
        node *_prev = nullptr;

        node(K key, V data) : _key(std::move(key)), _data(std::move(data)) {}

        ~node() {}
    };

    std::unordered_map<K, node *> map;
    node *_head = nullptr;
    node *_tail = nullptr;
    size_t _capacity;
    size_t _size{0};

    void remove(K key)
    {
        node *NODE = map[key];
        node *prev = NODE->_prev;
        node *next = NODE->_next;

        if (prev)
            prev->_next = next;
        else
            _tail = next;
        if (next)
            next->_prev = prev;
        else
            _head = prev;

        map.erase(key);
        delete NODE;
    }

public:
    V get(K key)
    {

        if (!map.count(key))
        {
            throw std::out_of_range("invalid key => lru_cache::get()");
        }

        node *NODE = map[key];
        node *prev = NODE->_prev;
        node *next = NODE->_next;

        if (prev)
            prev->_next = next;
        else
            _tail = next;
        if (next)
            next->_prev = prev;
        else
            _head = prev;

        NODE->_next = nullptr;
        NODE->_prev = _head;

        if (_head)
            _head->_next = NODE;
        _head = NODE;
        if (_tail == nullptr)
            _tail = NODE;

        return NODE->_data;
    }
    void put(K key, V data)
    {
        node *NODE = new node(key, data);
        map[key] = NODE;

        if (_size == _capacity)
        {
            remove(_tail->_key);
        }

        if (_head == nullptr)
        {
            _head = NODE;
            _tail = NODE;
        }
        else
        {
            _head->_next = NODE;
            NODE->_prev = _head;
            _head = NODE;
        }
        ++_size;
    }

    lru_cache(size_t capacity) : _capacity(std::move(capacity)) {}

    ~lru_cache()
    {
    }
};