#pragma once

#include "person.h"
#include <iostream>
#include <string>
#include <string_view>

class Cricket_Player : protected Person
{
private:
    std::string position{};
    int speed_kph{};

public:
    Cricket_Player(std::string name, int age, std::string position = "batsman", int speed_kph = 10)
        : Person(name, age), position(position), speed_kph(speed_kph)
    {
        std::cout << "Cricketer Created!" << "\n";
    }
    ~Cricket_Player()
    {
        std::cout << "Cricketer Dropped!" << "\n";
    }

    void identify() const { std::cout << "Cricket Player " << getName() << " with an age of " << getAge() << " has been identified" << "\n"; }
};