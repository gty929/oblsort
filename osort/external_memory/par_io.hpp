#pragma once

#include "noncachedvector.hpp"
#include <mutex>
#include <queue>
#include <thread>
#include <atomic>

template <typename T>
class ThreadSafeQueue {
    std::queue<T> queue;
    std::mutex mutex;

public:
    void push(T t) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(t);
    }

    bool pop(T& t) {
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.empty()) {
            return false;
        }
        t = queue.front();
        queue.pop();
        return true;
    }
};

template <typename BaseRW, class Iterator>
class RWManager {
    BaseRW* baseRWs;
    ThreadSafeQueue<BaseRW*> rwQueue;
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
        baseRWs = new BaseRW[parCount];
        for (int i = 0; i < parCount; i++) {
            Iterator baseRWBegin = begin + size_per_rw * i;
            Iterator baseRWEnd = std::min(begin + size_per_rw * (i + 1), end);
            baseRWs[i].init(baseRWBegin, baseRWEnd, auth);
            rwQueue.push(baseRWs + i);
        }
    }

    BaseRW* getRW() {
        BaseRW* ret;
        while (!rwQueue.pop(ret)) {
            if (availableRWCount == 0) {
                return nullptr; // all rws have been consumed
            }
            // std::this_thread::yield();
        }
        return ret;
    }

    void returnRW(BaseRW* rw) { 
        if (rw->eof()) {
            availableRWCount--;
            return; // this rw has been consumed
        }
        rwQueue.push(rw); }

    // flush all pages
    void flush() {
        for (int i = 0; i < availableRWCount; i++) {
            baseRWs[i].flush();
        }
    }

    ~RWManager() {
        delete[] baseRWs;
    }
};