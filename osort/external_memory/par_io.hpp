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

template <typename BaseReader, class Iterator>
class ReaderManager {
    BaseReader* baseReaders;
    ThreadSafeQueue<BaseReader*> readerQueue;
    std::atomic_int availableReaderCount;
    using TRef = std::iterator_traits<Iterator>::reference;
public:
    ReaderManager() {}

    ReaderManager(const ReaderManager&) = delete;

    void init(Iterator begin, Iterator end, uint32_t auth, int parCount) {
        availableReaderCount = parCount;
        baseReaders = new BaseReader[parCount];
        size_t size_per_reader = divRoundUp(end - begin, parCount);
        for (int i = 0; i < parCount; i++) {
            Iterator baseReaderBegin = begin + size_per_reader * i;
            Iterator baseReaderEnd = std::min(begin + size_per_reader * (i + 1), end);
            baseReaders[i].init(baseReaderBegin, baseReaderEnd, auth);
            readerQueue.push(baseReaders + i);
        }
    }

    BaseReader* getReader() {
        BaseReader* ret;
        while (!readerQueue.pop(ret)) {
            if (availableReaderCount == 0) {
                return nullptr; // all readers have been consumed
            }
            // std::this_thread::yield();
        }
        return ret;
    }

    void returnReader(BaseReader* reader) { 
        if (reader->eof()) {
            availableReaderCount--;
            return; // this reader has been consumed
        }
        readerQueue.push(reader); }

    ~ReaderManager() {
        delete[] baseReaders;
    }
};