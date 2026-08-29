#include "person.h"
#include "cricketer.h"

#include <string>
#include <string_view>

#include <iostream>

class Base
{
private:
    void print() const
    {
        std::cout << "Base";
    }
};

class Derived : public Base
{
public:
    void print() const
    {
        std::cout << "Derived ";
    }
};

int main()
{
    Person p1("Jonah", 5);
    Cricket_Player("Arjun", 20, "All-Rounder", 10);
}