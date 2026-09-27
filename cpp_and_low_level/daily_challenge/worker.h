// 9/27/2026

#ifndef WORKER_H
#define WORKER_H

#include <utility>
#include <queue>
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <functional>

class Worker
{
private:
    std::thread _thread;
    std::mutex _mutex;
    bool _stop = false;
    std::queue<std::function<void()>> _queue;
    std::condition_variable _no_work; // in the case where there's no functions to do work "with" place the thread in a sleep area

    void run_work()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _no_work.wait(lock, [this]
                      { return _stop || !_queue.empty(); });

        if (!_queue.empty())
        {
            std::function<void()> func = std::move(_queue.front());
            _queue.pop();
            lock.unlock();
            func();
        }
    }

public:
    void submit_work(std::function<void()> func) // analagous to push or enque
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _queue.push(func);
        lock.unlock();
        _no_work.notify_one();
    }

    void loop()
    {
        while (!(_stop && _queue.size() == 0))
        {
            run_work();
        }
    }

    Worker() : _thread(&Worker::loop, this) {}
    ~Worker()
    {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _stop = true;
        }
        _no_work.notify_one();
        _thread.join();
    }
};

#endif