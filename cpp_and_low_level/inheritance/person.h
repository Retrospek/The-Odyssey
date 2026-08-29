#pragma once

#include <iostream>
#include <string>
#include <string_view>

class Person
{
private:
    std::string name{};
    int age{};

public:
    Person(std::string name = "baby", int age = 0) : name{name}, age{age}
    {
        std::cout << name << " with an age of " << age << " was just created!" << "\n";
    }
    ~Person()
    {
        std::cout << name << " with and age of " << age << " just died :(" << "\n";
    }

    const std::string &getName() const { return name; }
    int getAge() const { return age; }
    void identify() const { std::cout << name << " with an age of " << age << " has been identified" << "\n"; }
};