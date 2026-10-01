// 9/26/2026
#pragma once

#include <cstddef>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

template <typename T>
class BoundedQueue
{
private:
    std::queue<T> _queue;
    size_t _capacity;
    std::mutex _mutex;
    std::condition_variable _not_empty;
    std::condition_variable _not_full;

public:
    BoundedQueue(size_t _capacity) : _capacity(_capacity) {}
    T pop()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _not_empty.wait(lock, [this]
                        { return !empty(); });
        T front = std::move(_queue.front());
        _queue.pop();
        lock.unlock();
        _not_full.notify_one();
        return front;
    }

    void push(const T &value)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _not_full.wait(lock, [this]
                       { return size() < _capacity; });
        _queue.push(value);
        lock.unlock();
        _not_empty.notify_one();
    }

    bool empty()
    {
        return _queue.empty();
    }

    size_t size()
    {
        return _queue.size();
    }
};