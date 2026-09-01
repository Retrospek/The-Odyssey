#include <iostream>
using namespace std;

template <typename T>
class Auto_ptr
{
    T *m_ptr{};

public:
    Auto_ptr(T *ptr = nullptr) : m_ptr(ptr)
    {
    }

    ~Auto_ptr()
    {
        delete m_ptr;
    }
    Auto_ptr(Auto_ptr &a)
    {
        m_ptr = a.m_ptr;
        a.m_ptr = nullptr;
    }
    Auto_ptr &operator=(Auto_ptr &a)
    {
        if (&a == this)
        {
            return *this;
        }

        delete m_ptr;
        m_ptr = a.m_ptr;
        a.m_ptr = nullptr;
        return *this;
    }
};

class Resource
{
public:
    Resource()
    {
        cout << "Just created" << "\n";
    }

    ~Resource()
    {
        cout << "Just deleted" << "\n";
    }
};

int main()
{
    Auto_ptr ptr(new Resource());
    return 9;
};