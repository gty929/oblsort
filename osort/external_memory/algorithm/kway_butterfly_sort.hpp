#pragma once
#include "external_memory/dynamicvector.hpp"
#include "external_memory/noncachedvector.hpp"
#include "param_select.hpp"
#include "sort_building_blocks.hpp"
#include "external_memory/par_io.hpp"
// #include <atomic>
// #include <unordered_set>

/// This file implements the flex-way butterfly osort and oshuffle algorithms.

namespace EM::Algorithm {

/// @brief KWayButterflySort is an oblivious external memory sorting algorithm.
/// The implementation incurs approximately 2.23NlogN exchanges and
/// 2N/B log_{M/B} N/B page transfers, where N is the size of the input array, B
/// is the page size, and M is the size of the memory.
/// @tparam Iterator only supports NonCachedVector::Iterator
/// @param begin the begin iterator of the input array
/// @param end the end iterator of the input array
/// @param inAuth the authentication counter of the input array
/// @param heapSize the size of available memory in bytes
template <class Iterator>
void KWayButterflySort(Iterator begin, Iterator end, uint32_t inAuth = 0,
                       uint64_t heapSize = DEFAULT_HEAP_SIZE);
template <typename Vec>
void KWayButterflySort(Vec& vec, uint32_t inAuth = 0,
                       uint64_t heapSize = DEFAULT_HEAP_SIZE);

/// @brief KWayButterflyShuffling is an oblivious external memory shuffling
/// algorithm (i.e., random permutation).
/// The implementation incurs approximately 2NlogN exchanges and
/// N/B log_{M/B} N/B page transfers, where N is the size of the input array, B
/// is the page size, and M is the size of the memory.
/// @tparam Iterator only supports NonCachedVector::Iterator
/// @param begin the begin iterator of the input array
/// @param end the end iterator of the input array
/// @param inAuth the authentication counter of the input array
/// @param heapSize the size of available memory in bytes
template <class Iterator>
void KWayButterflyOShuffle(Iterator begin, Iterator end, uint32_t inAuth = 0,
                           uint64_t heapSize = DEFAULT_HEAP_SIZE);
template <typename Vec>
void KWayButterflyOShuffle(Vec& vec, uint32_t inAuth = 0,
                           uint64_t heapSize = DEFAULT_HEAP_SIZE);

/// @brief A manager class for flex-way butterfly o-sort.
/// @tparam T the type of elements to sort
/// @tparam WrappedT the wrapped type of elements to sort
template <typename IOIterator, SortMethod task>
class ButterflySorter {
 private:
  using T = typename std::iterator_traits<IOIterator>::value_type;
  using WrappedT = TaggedT<T>;
  using IOVector = typename
      std::remove_reference<decltype(*(IOIterator::getNullVector()))>::type;
  uint64_t Z;                 // bucket size
  uint64_t numTotalBucket;    // total number of buckets
  uint64_t numRealPerBucket;  // number of real elements per bucket

  uint64_t numBucketFit;   // number of buckets that can fit in the heap
  uint64_t numElementFit;  // number of elements that can fit in the heap

  KWayButterflyParams KWayParams =
      {};  // parameters for flex-way butterfly o-sort

  Vector<T>
      mergeSortFirstLayer;  // the first layer of external-memory merge sort
  std::vector<
      std::pair<typename Vector<T>::Iterator, typename Vector<T>::Iterator>>
      mergeSortRanges;  // pairs of iterators that specifies each sorted range
                        // in the first layer of external-memory merge sort

  // writer for the first layer of external-memory merge sort
  typename Vector<T>::Writer mergeSortFirstLayerWriter;

  RWManager<typename IOVector::PrefetchReader, typename IOVector::Iterator> inputReaderManager;  // input reader
  RWManager<typename IOVector::Writer, typename IOVector::Iterator> outputWriterManager;         // output writer
  WrappedT* batch;                                // batch for sorting
 public:
  /// @brief Construct a new Butterfly Sorter object
  /// @param inputBeginIt the begin iterator of the input array
  /// @param inputEndIt the end iterator of the input array
  /// @param inAuth the counter of the input array for authentication
  /// @param _heapSize the heap size in bytes
  ButterflySorter(IOIterator inputBeginIt, IOIterator inputEndIt,
                  uint32_t inAuth = 0, uint64_t _heapSize = DEFAULT_HEAP_SIZE)
      : mergeSortFirstLayer(
            task == KWAYBUTTERFLYOSORT ? inputEndIt - inputBeginIt : 0),
        numElementFit(_heapSize / sizeof(WrappedT)) {
    size_t size = inputEndIt - inputBeginIt;
    batch = new WrappedT[numElementFit]; // declare ahead to avoid fragmentation
    KWayParams = bestKWayButterflyParams(size, numElementFit, sizeof(T));
    inputReaderManager.init(inputBeginIt, inputEndIt, inAuth, thread_count);
    Z = KWayParams.Z;
    Assert(numElementFit > 8 * Z * thread_count);
    numElementFit -=
        divRoundUp(Z * 8 * (sizeof(WrappedT) + 2), sizeof(WrappedT)) * thread_count;
    // deduct the temp memory for merge split
    delete[] batch;
    batch = new WrappedT[numElementFit];
    // reserve space for 8 buckets of elements and marks
    numBucketFit = numElementFit / Z;
    numTotalBucket = KWayParams.totalBucket;
    numRealPerBucket = 1 + (size - 1) / numTotalBucket;

    if constexpr (task == KWAYBUTTERFLYOSORT) {
      mergeSortFirstLayerWriter.init(mergeSortFirstLayer.begin(),
                                     mergeSortFirstLayer.end());
    } else {
      static_assert(task == KWAYBUTTERFLYOSHUFFLE);
      outputWriterManager.init(inputBeginIt, inputEndIt, inAuth + 1, thread_count);
    }
  }

  ~ButterflySorter() {
    if (batch != NULL) {
      delete[] batch;
    }
  }

  const auto& getMergeSortBatchRanges() { return mergeSortRanges; }

  /// @brief Base case of flex-way butterfly o-sort when input fits in memory
  /// @tparam Iterator should support random access
  /// @param begin begin iterator of the input array
  /// @param end end iterator of the input array
  /// @param ioLayer current layer of butterfly network by page swap passes
  /// @param innerLayer current layer of butterfly network within the batch
  template <class Iterator>
  void KWayButterflySortBasic(Iterator begin, Iterator end, size_t ioLayer,
                              size_t innerLayer) {
    uint64_t numElement = end - begin;
    uint64_t numBucket = numElement / Z;
    uint64_t way = KWayParams.ways[ioLayer][innerLayer];
    Assert(numElement % Z == 0);
    Assert(numBucket % way == 0);
    uint64_t waySize = numElement / way;
    uint64_t wayBucket = numBucket / way;
    if (innerLayer > 0) {

      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < way; ++i) {
        KWayButterflySortBasic(begin + i * waySize, begin + (i + 1) * waySize,
                              ioLayer, innerLayer - 1);
      }
    
    } else {
      Assert(numBucket == way);
      if (ioLayer == 0) {
        Assert(waySize == Z);
        typename IOVector::PrefetchReader* inputReader = inputReaderManager.getRW();
        bool overFlag = !inputReader;
        // tag and pad input
        for (uint64_t i = 0; i < way; ++i) {
          if (overFlag) {
            // set rest of the buckets to dummy
            for (; i < way; ++i) {
              auto it = begin + i * waySize;
              for (uint64_t offset = 0; offset < Z; ++offset, ++it) {
                it->setDummy();
              }
            }
            break;
          }
          auto it = begin + i * waySize;
          for (uint64_t offset = 0; offset < Z; ++offset, ++it) {
            if (offset < numRealPerBucket) {
              if (inputReader->eof()) {
                inputReaderManager.returnRW(inputReader);
                inputReader = inputReaderManager.getRW();
                if (!inputReader) {
                  overFlag = true;
                  // all readers have been consumed, pad dummies to the end
                  for (; offset < Z; ++offset, ++it) {
                    it->setDummy();
                  }
                  break;
                }
              }
              it->setData(inputReader->read(), *inputReader->prng);
            } else {
              it->setDummy();
            }
          }
        }
        if (inputReader) {
          inputReaderManager.returnRW(inputReader);
        }
      }
    }
    int chunkSize = std::max(1, (int)wayBucket / thread_count);
      #pragma omp parallel for schedule(static, chunkSize)
      for (uint64_t j = 0; j < wayBucket; ++j) {
        Iterator KWayIts[8];
        for (uint64_t i = 0; i < way; ++i) {
          KWayIts[i] = begin + (i * wayBucket + j) * Z;
        }
        WrappedT* temp = new WrappedT[way * Z];
        uint8_t* marks = new uint8_t[8 * Z];
        MergeSplitKWay(KWayIts, way, Z, temp, marks);
        
        delete[] temp;
        delete[] marks;
      }
  }

  /// @brief Base case of flex-way butterfly o-sort when input fits in memory
  /// @tparam Iterator should support random access
  /// @param begin begin iterator of the input array
  /// @param end end iterator of the input array
  /// @param ioLayer current layer of butterfly network by page swap passes
  /// @param innerLayer current layer of butterfly network within the batch
  template <class Iterator>
  void KWayButterflySortBasicNonRecursive(Iterator begin, Iterator end, size_t ioLayer,
                              size_t innerLayer) {
    uint64_t numElement = end - begin;
    uint64_t numBucket = numElement / Z;
    
    for (uint64_t layer = 0, stride = 1; layer <= innerLayer; ++layer) {
      uint64_t way = KWayParams.ways[ioLayer][layer];
      uint64_t wayBucket = numBucket / way;
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < wayBucket; ++i) {
        // merge read input to obtain better locality
        if (layer == 0 && ioLayer == 0) {
          typename IOVector::PrefetchReader* inputReader = inputReaderManager.getRW();
          bool overFlag = !inputReader;
          // tag and pad input
          for (uint64_t j = 0; j < way; ++j) {
            if (overFlag) {
              // set rest of the buckets to dummy
              for (; j < way; ++j) {
                auto it = begin + (i * way + j) * Z;
                for (uint64_t offset = 0; offset < Z; ++offset, ++it) {
                  it->setDummy();
                }
              }
              break;
            }
            auto it = begin + (i * way + j) * Z;
            for (uint64_t offset = 0; offset < Z; ++offset, ++it) {
              if (offset < numRealPerBucket) {
                if (inputReader->eof()) {
                  inputReaderManager.returnRW(inputReader);
                  inputReader = inputReaderManager.getRW();
                  if (!inputReader) {
                    overFlag = true;
                    // all readers have been consumed, pad dummies to the end
                    for (; offset < Z; ++offset, ++it) {
                      it->setDummy();
                    }
                    break;
                  }
                }
                it->setData(inputReader->read(), *inputReader->prng);
              } else {
                it->setDummy();
              }
            }
          }
          if (inputReader) {
            inputReaderManager.returnRW(inputReader);
          }
        }
        uint64_t groupIdx = i / stride;
        uint64_t groupOffset = i % stride;
        Iterator KWayIts[8];
        for (uint64_t j = 0; j < way; ++j) {
          KWayIts[j] = begin + ((j + groupIdx * way) * stride + groupOffset) * Z;
        }
        WrappedT* temp = new WrappedT[way * Z];
        uint8_t* marks = new uint8_t[8 * Z];
        MergeSplitKWay(KWayIts, way, Z, temp, marks);
        
        delete[] temp;
        delete[] marks;
      }

      stride *= way;
    }
  }

  template <class Iterator>
  void KWayButterflySort(Iterator begin, Iterator end) {
    KWayButterflySort(begin, end, KWayParams.ways.size() - 1);
  }

  template <class Iterator>
  void KWayButterflySort(Iterator begin, Iterator end, size_t maxIoLayer) {
    for (size_t ioLayer = 0; ioLayer <= maxIoLayer; ++ioLayer) {
      #ifdef ENCLAVE_MODE
      uint64_t currTime;
      ocall_measure_time(&currTime);
      #endif
      bool isLastLayer = ioLayer == maxIoLayer;
      size_t numInternalWay = getVecProduct(KWayParams.ways[ioLayer]);
      size_t fetchInterval = 1;
      for (size_t layer = 0; layer < ioLayer; ++layer) {
        fetchInterval *= getVecProduct(KWayParams.ways[layer]);
      }
      size_t size = end - begin;
      size_t numInterval = size / Z / fetchInterval;
      Assert(size % numInternalWay == 0);
      // size_t subSize = size / numInternalWay;
      // if (ioLayer > 0) {
      //   for (size_t i = 0; i < numInternalWay; ++i) {
      //     KWayButterflySort(begin + i * subSize, begin + (i + 1) * subSize,
      //                       ioLayer - 1);
      //   }
      // }
      if (size / Z % numInternalWay != 0) {
        printf("size = %ld, Z = %ld, numInternalWay = %ld\n", size, Z, numInternalWay);
        abort();
      }
      size_t bucketPerBatch = std::min(size / Z, numBucketFit / numInternalWay * numInternalWay);
      Assert(bucketPerBatch > 0); 
      size_t batchSize = bucketPerBatch * Z;
      size_t batchCount = divRoundUp(size, batchSize);
      
      // std::unordered_set<size_t> extIts;
      std::mutex m;
      for (uint64_t batchIdx = 0; batchIdx < batchCount; ++batchIdx) {
        // printf("batchIdx = %d\n", batchIdx);
        size_t bucketThisBatch = bucketPerBatch;
        if (batchIdx == batchCount - 1) {
          bucketThisBatch = size / Z - bucketPerBatch * (batchCount - 1);
        }
        if (ioLayer) {  // fetch from intermediate ext vector
          int chunkSize = std::max(1, (int)bucketThisBatch / thread_count);
          // size_t batchWayRatio = bucketPerBatch / numInternalWay;
          
          #pragma omp parallel for schedule(static)
          for (uint64_t bucketIdx = 0; bucketIdx < bucketThisBatch; ++bucketIdx) {
            size_t bucketGlobalIdx = batchIdx * bucketPerBatch + bucketIdx;
            auto extBeginIt = begin + (bucketGlobalIdx / numInterval + (bucketGlobalIdx % numInterval) * fetchInterval) * Z;
            auto intBeginIt = batch + bucketIdx * Z;
            // {
            //   std::lock_guard<std::mutex> lock(m);
            //   if (extIts.find(extBeginIt.get_m_ptr()) != extIts.end()) {
            //     printf("extBeginIt = %p\n", extBeginIt.get_m_ptr());
            //     abort();
            //   }
            //   extIts.insert(extBeginIt.get_m_ptr());
            // }
            CopyIn(extBeginIt, extBeginIt + Z, intBeginIt, ioLayer - 1);
          }
        }
        KWayButterflySortBasicNonRecursive(batch, batch + bucketThisBatch * Z, ioLayer,
                              KWayParams.ways[ioLayer].size() - 1);
        if (isLastLayer) {
          // last layer, combine with bitonic sort and output
          const auto cmpTag = [](const auto& a, const auto& b) {
            return a.tag < b.tag;
          };
          // int chunkSize = std::max(1, (int)bucketThisBatch / thread_count);
          #pragma omp parallel for schedule(static)
          for (size_t i = 0; i < bucketThisBatch; ++i) {
            auto it = batch + i * Z;
            Assert(it + Z <= batch + numElementFit);
            BitonicSort(it, it + Z, cmpTag);
            // for shuffling, output directly
            
            if constexpr (task == KWAYBUTTERFLYOSHUFFLE) {
              auto* outputWriter = outputWriterManager.getRW();
              if (!outputWriter) {
                printf("outputWriter is null\n");
                abort();
              }
              for (auto fromIt = it; fromIt != it + Z; ++fromIt) {
                if (!fromIt->isDummy()) {
                  if (outputWriter->eof()) {
                    outputWriterManager.returnRW(outputWriter);
                    outputWriter = outputWriterManager.getRW();
                    if (!outputWriter) {
                      printf("outputWriter is null\n");
                      abort();
                    }
                  }
                  outputWriter->write(fromIt->getData());
                }
              }
              outputWriterManager.returnRW(outputWriter);
            }
            
          }
          if (task == KWAYBUTTERFLYOSORT) {
            // sort the batch and write to first layer of merge sort
            const auto cmpVal = [](const auto& a, const auto& b) {
              return a.v < b.v;
            };
            auto realEnd =
                partitionDummy(batch, batch + batchSize);
            // partition dummies to the end
            Assert(realEnd <= batch + numElementFit);
            std::sort(batch, realEnd, cmpVal);
            auto mergeSortReaderBeginIt = mergeSortFirstLayerWriter.it;
            for (auto it = batch; it != realEnd; ++it) {
              mergeSortFirstLayerWriter.write(it->getData());
            }

            mergeSortRanges.emplace_back(mergeSortReaderBeginIt,
                                        mergeSortFirstLayerWriter.it);
          }
        } else {  // not last layer, write to intermediate ext vector
          size_t batchWayRatio = bucketPerBatch / numInternalWay;
          // size_t chunkSize = std::max(1, (int)bucketThisBatch / thread_count);
          #pragma omp parallel for schedule(static)
          for (uint64_t bucketIdx = 0; bucketIdx < bucketThisBatch; ++bucketIdx) {
            size_t bucketGlobalIdx = batchIdx * bucketPerBatch + bucketIdx;
            auto extBeginIt = begin + (bucketGlobalIdx / numInterval + (bucketGlobalIdx % numInterval) * fetchInterval) * Z;
            auto intBeginIt = batch + bucketIdx * Z;
            CopyOut(intBeginIt, intBeginIt + Z, extBeginIt, ioLayer);
          }
        }
      }
      #ifdef ENCLAVE_MODE
      uint64_t currTime2;
      ocall_measure_time(&currTime2);
      uint64_t timediff = currTime2 - currTime;
      printf("Layer %ld: %d.%d\n", ioLayer, timediff / 1'000'000'000,
           timediff % 1'000'000'000);
      #endif
    }
    if constexpr (task == KWAYBUTTERFLYOSORT) {
      mergeSortFirstLayerWriter.flush();
    } else {
      outputWriterManager.flush();
    }
  }

  void sort() {
    if (batch == NULL) {
      printf("sorter called twice\n");
      abort();
    }
    EM::DynamicPageVector::Vector<TaggedT<T>> v(getOutputSize(),
                                                getBucketSize());
    KWayButterflySort(v.begin(), v.end());
    delete[] batch;
    batch = NULL;
  }

  size_t getOutputSize() { return numTotalBucket * Z; }

  size_t getBucketSize() { return Z; }
};

template <class Iterator>
void KWayButterflyOShuffle(Iterator begin, Iterator end, uint32_t inAuth,
                           uint64_t heapSize) {
  using T = typename std::iterator_traits<Iterator>::value_type;
  size_t N = end - begin;
  if (N <= 512) {
    std::vector<T> Mem(N);
    CopyIn(begin, end, Mem.begin(), inAuth);
    OrShuffle(Mem);
    CopyOut(Mem.begin(), Mem.end(), begin, inAuth + 1);
    return;
  }
  ButterflySorter<Iterator, KWAYBUTTERFLYOSHUFFLE> sorter(begin, end, inAuth,
                                                          heapSize);
  sorter.sort();
}

template <class Iterator>
void KWayButterflySort(Iterator begin, Iterator end, uint32_t inAuth,
                       uint64_t heapSize) {
  using T = typename std::iterator_traits<Iterator>::value_type;
  const uint64_t N = end - begin;
  if (N <= 512) {
    std::vector<T> Mem(N);
    CopyIn(begin, end, Mem.begin(), inAuth);
    BitonicSort(Mem);
    CopyOut(Mem.begin(), Mem.end(), begin, inAuth + 1);
    return;
  }
  // omp_set_num_threads(4);
  ButterflySorter<Iterator, KWAYBUTTERFLYOSORT> sorter(begin, end, inAuth,
                                                       heapSize);
  sorter.sort();
  const auto& mergeRanges = sorter.getMergeSortBatchRanges();
  ExtMergeSort(begin, end, mergeRanges, inAuth + 1,
               heapSize / (sizeof(T) * Vector<T>::item_per_page * 2));
}

template <typename Vec>
void KWayButterflySort(Vec& vec, uint32_t inAuth, uint64_t heapSize) {
  KWayButterflySort(vec.begin(), vec.end(), inAuth, heapSize);
}

template <typename Vec>
void KWayButterflyOShuffle(Vec& vec, uint32_t inAuth, uint64_t heapSize) {
  KWayButterflyOShuffle(vec.begin(), vec.end(), inAuth, heapSize);
}

}  // namespace EM::Algorithm