#include "LRU_cache.h"
#include <iostream>

int main()
{
    lru_cache<int, int> lru(2);
    lru.put(1, 100);
    lru.put(2, 200);
    std::cout << lru.get(1) << "\n";
    lru.put(3, 300);
    std::cout << lru.get(1) << "\n";
    std::cout << lru.get(3) << "\n";
    std::cout << lru.get(2) << "\n";
}