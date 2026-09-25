// 9/24/2026

#include <iostream>
#include <cmath>
#include <utility>
#include <algorithm>

template <typename T>
class DynArray
{
private:
    T *_data = nullptr;
    size_t _size = 0;
    size_t _capacity = 0;

    void resize(size_t offset = 0)
    /*
    When called by push_back or emplace_back it should resize the entire data buffer by a factor of 2
    Then it should copy all elements into that new buffer, and adjust the private members \
    (not the _size because that's handled by push_back)
    */
    {
        if (_capacity == 0)
        {
            _capacity = 1;
        }
        else
        {
            _capacity *= 2;
        }

        T *_new_data = static_cast<T *>(operator new(sizeof(T) * _capacity));
        for (size_t i{offset}; i < _size + offset; ++i)
        {
            new (&_new_data[i]) T(std::move(_data[i - offset]));
        }
        for (size_t i{0}; i < _size; ++i)
            _data[i].~T();
        _data = _new_data;
    }

    void shift_add(size_t shift = 1)
    {
        if (_size >= _capacity)
        {
            resize();
        }

        T *_new_data = static_cast<T *>(operator new(sizeof(T) * _capacity));
        for (size_t i{shift}; i < _size + shift; ++i)
        {
            new (&_new_data[i]) T(std::move(_data[i - shift]));
        }

        for (size_t i{0}; i < _size; ++i)
            _data[i].~T();
        operator delete(_data);
        _data = _new_data;
    }

    bool valid_capacity() { return _size < _capacity; }

public:
    // default constructor
    DynArray() = default;
    DynArray(size_t initial_capacity) : _capacity(initial_capacity)
    {
        if (_capacity > 0)
        {
            _data = static_cast<T *>(operator new(sizeof(T) * _capacity));
        }
    }
    // descructor
    ~DynArray()
    {
        for (size_t i{0}; i < _size; ++i)
            _data[i].~T();
        operator delete(_data);
    }

    // copy constructor
    DynArray(DynArray &other) : _size(other._size), _capacity(other._capacity)
    {
        _data = static_cast<T *>(operator new(sizeof(T) * _capacity));
        for (size_t i{0}; i < _size; ++i)
        {
            new (&_data[i]) T(other._data[i]);
        }
    }

    // copy assignment
    DynArray &operator=(DynArray &other)
    {
        if (this == &other)
            return *this;

        for (size_t i{0}; i < _size; ++i)
            _data[i].~T();
        operator delete(_data);

        _size = other._size;
        _capacity = other._capacity;

        _data = static_cast<T *>(operator new(sizeof(T) * _capacity));
        for (size_t i{0}; i < _size; ++i)
        {
            new (&_data[i]) T(other._data[i]);
        }

        return *this;
    }

    // move constructor
    DynArray(DynArray &&other) : _size(other._size), _capacity(other._capacity)
    {
        _data = other._data;

        other._data = nullptr;
        other._size = 0;
        other._capacity = 0;
    }

    // move assignment
    DynArray &operator=(DynArray &&other)
    {
        if (this == &other)
            return *this;

        for (size_t i{0}; i < _size; ++i)
            _data[i].~T();
        operator delete(_data);

        _data = other._data;
        _size = other._size;
        _capacity = other._capacity;

        other._data = nullptr;
        other._size = 0;
        other._capacity = 0;

        return *this;
    }

    size_t size() { return _size; }
    size_t capacity() { return _capacity; }

    T &operator[](const size_t &i)
    {
        return _data[i];
    }

    T &at(const size_t &i)
    {
        if (i < _size && i >= 0)
        {
            return _data[i];
        }
        else
        {
            throw std::runtime_error("Out of Bounds Access");
        }
    }

    void push_back(const T &value)
    {
        if (!valid_capacity())
        {
            resize();
        }
        new (&_data[_size]) T(value);
        ++_size;
    }

    void push_back(T &&value)
    {
        if (!valid_capacity())
        {
            resize();
        }
        new (&_data[_size]) T(std::move(value));
        ++_size;
    }

    void push_front(const T &value)
    {
        shift_add();
        new (&_data[0]) T(value);
        ++_size;
    }

    void push_front(T &&value)
    {
        shift_add();
        new (&_data[0]) T(std::move(value));
        ++_size;
    }

    T &pop_back()
    {
        T _end = std::move(_data[_size - 1]);
        --_size;

        return _end;
    }

    void emplace_back(T &value)
    {
    }

    void emplace_back(T &&value)
    {
    }

    void remove(size_t &i)
    {
    }

    void remove(size_t &&i)
    {
    }
};