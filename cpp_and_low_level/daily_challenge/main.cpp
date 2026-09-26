#include "thread_safe_bounded_queue.h"

int main()
{
    BoundedQueue<int> queue(2);
    queue.push(42);
    return queue.pop() == 42 ? 0 : 1;
}