#pragma once

#include "noncachedvector.hpp"
#include <mutex>
#include <queue>
#include <thread>
#include <atomic>
#include "common/queue.hpp"
template <typename T>
class ThreadSafeQueue {
    std::queue<T> queue;
    std::mutex mutex;

public:
    void Push(const T& t) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(t);
    }

    bool Pop(T& t) {
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.empty()) {
            return false;
        }
        t = queue.front();
        queue.pop();
        return true;
    }
};

template <typename T>
class ThreadSafeStack {
    std::vector<T> stack;
    std::mutex mutex;

public:
    void init(uint64_t size) {
        stack.reserve(size);
    }

    void Push(const T& t) {
        std::lock_guard<std::mutex> lock(mutex);
        stack.push_back(t);
    }

    bool Pop(T& t) {
        std::lock_guard<std::mutex> lock(mutex);
        if (stack.empty()) {
            return false;
        }
        t = stack.back();
        stack.pop_back();
        return true;
    }
};

template <typename BaseRW, class Iterator>
class RWManager {
    BaseRW* baseRWs = NULL;
    int parCount = 0;
    ThreadSafeQueue<BaseRW*> rwQueue;
    // lockfree::mpmc::Queue<BaseRW*, 64> rwQueue;
    std::atomic_int availableRWCount;
    using TRef = std::iterator_traits<Iterator>::reference;
public:
    RWManager() {}

    RWManager(const RWManager&) = delete;

    template<const bool align = true>
    void init(Iterator begin, Iterator end, uint32_t auth, int parCount) {
        size_t size_per_rw = divRoundUp(end - begin, parCount);
        if constexpr (align) {
            size_t size_per_page = begin.getVector().item_per_page;
            size_per_rw = divRoundUp(size_per_rw, size_per_page) * size_per_page;
        }
        
        assert(size_per_rw >= begin.getVector().item_per_page);
        parCount = divRoundUp(end - begin, size_per_rw); // adjust the number of reader / writer
        availableRWCount = parCount;
        this->parCount = parCount;
        baseRWs = new BaseRW[parCount];
        for (int i = 0; i < parCount; i++) {
            Iterator baseRWBegin = begin + size_per_rw * i;
            Iterator baseRWEnd = std::min(begin + size_per_rw * (i + 1), end);
            baseRWs[i].init(baseRWBegin, baseRWEnd, auth);
            rwQueue.Push(baseRWs + i);
        }
    }

    BaseRW* getRW() {
        BaseRW* ret;
        while (!rwQueue.Pop(ret)) {
            if (availableRWCount == 0) {
                return nullptr; // all rws have been consumed
            }
            // std::this_thread::yield();
        }
        return ret;
    }

    void returnRW(BaseRW* rw) { 
        if (rw->eof()) {
            // printf("availableRWCount reduced to %d\n", availableRWCount);
            availableRWCount--;
            return; // this rw has been consumed
        }
        rwQueue.Push(rw); }

    // flush all pages
    void flush() {
        for (int i = 0; i < parCount; i++) {
            baseRWs[i].flush();
        }
    }

    ~RWManager() {
        if (baseRWs) {
            delete[] baseRWs;
        }
    }
};