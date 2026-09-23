// 9/23/2026

#include <iostream>
#include <utility>

template <typename T>
class unique_ptr
{
private:
    T *ptr = nullptr;

public:
    // default constructor
    unique_ptr(T *ptr = nullptr) : ptr(ptr) {};
    // copy constructor
    unique_ptr(unique_ptr &other) = delete;
    // copy assignment
    unique_ptr &operator=(unique_ptr &other) = delete;

    // move constructor
    unique_ptr(unique_ptr &&other) : ptr(other.ptr) { other.ptr = nullptr; }
    // move assignment
    unique_ptr &operator=(unique_ptr &&other)
    {
        if (this == &other)
        {
            return *this;
        }
        delete ptr;
        ptr = other.ptr;
        other.ptr = nullptr;
        return *this;
    }

    ~unique_ptr()
    {
        delete ptr;
        ptr = nullptr;
    }

    T &operator*() const { return *ptr; }
    T *operator->() const { return ptr; }

    T *get() const { return ptr; }
    T *release()
    {
        T *temp = ptr;
        ptr = nullptr;
        return temp;
    }
    void reset(T *ptr_new)
    {
        delete ptr;
        ptr = ptr_new;
    }

    explicit operator bool() const { return ptr != nullptr; }
};