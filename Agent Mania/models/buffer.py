import torch
import torch.nn as nn

import heapq


class BaseBuffer:
    def __init__(self):
        raise NotImplementedError("Not Completed Initialization of Buffer")

    def __len__(self):
        raise NotImplementedError("Not Completed Length method of Buffer")

    def push(self, item):
        raise NotImplementedError("Not Completed Push method of Buffer")

    def pop(self, batch_size):
        raise NotImplementedError("Not Completed Pop method of Buffer")

class ERB(BaseBuffer):
    def __init__(self, n):
        self.queue=[]
        self.curr_size=0
        self.max_len=n

    def __len__(self):
        return self.curr_size

    def push(self, item):
        self.queue.append(item)
        self.curr_size += 1
    
        if self.curr_size > self.max_len:
            self.queue = self.queue[self.curr_size - self.max_len: ]
            self.curr_size = self.max_len

    def pop(self, batch_size):
        if len(self.queue) < batch_size:
            raise ValueError("Cannot sample from an insufficient size of experiences!")
        
        random_sample = torch.randint(0, self.curr_size, (batch_size,))

        return [self.queue[i] for i in random_sample]

class PERB(BaseBuffer):
    def __init__(self, n, priority, feature):
        self.heap=[]
        self.experiences=[]

        self.curr_size=0
        self.max_len=n
        self.priority=priority
        self.feature=feature
        self._counter = 0
        
    def __len__(self):
        return self.curr_size

    def push(self, item):
        multiplier = 1 if self.priority == "max" else -1
        priority_value = getattr(item, self.feature) * multiplier
        
        heap_entry = (priority_value, self._counter, item)
        self._counter += 1
        
        if self.curr_size < self.max_len:
            heapq.heappush(self.heap, heap_entry)
            self.curr_size += 1
        else:
            heapq.heappushpop(self.heap, heap_entry)

    def pop(self, batch_size):
        if self.curr_size < batch_size:
            raise IndexError("Cannot sample from an insufficient size of experiences!")
        
        batch_items = []
        for _ in range(batch_size):
            priority, counter, item = heapq.heappop(self.heap)
            self.curr_size -= 1
            batch_items.append(item)
            
        return batch_items