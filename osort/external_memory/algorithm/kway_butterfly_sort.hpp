#pragma once
#include "external_memory/dynamicvector.hpp"
#include "external_memory/noncachedvector.hpp"
#include "param_select.hpp"
#include "sort_building_blocks.hpp"
#include "external_memory/par_io.hpp"

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

  std::conditional_t<task == KWAYBUTTERFLYOSORT, Vector<T>, uint64_t>
      mergeSortFirstLayer;  // the first layer of external-memory merge sort
  std::vector<
      std::pair<typename Vector<T>::Iterator, typename Vector<T>::Iterator>>
      mergeSortRanges;  // pairs of iterators that specifies each sorted range
                        // in the first layer of external-memory merge sort

  // writer for the first layer of external-memory merge sort
  std::conditional_t<task == KWAYBUTTERFLYOSORT, typename Vector<T>::DeferedWriter, uint64_t> mergeSortFirstLayerWriter;

  RWManager<typename IOVector::PrefetchReader, typename IOVector::Iterator> inputReaderManager;  // input reader
  RWManager<typename IOVector::Writer, typename IOVector::Iterator> outputWriterManager;         // output writer
  ThreadSafeStack<uint64_t> freeTempIndices;
  WrappedT* batch;                                // batch for sorting
  uint8_t* tempBegin;
  uint64_t tempSize;
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
    batch = (WrappedT*)malloc(_heapSize); // declare ahead to avoid fragmentation
    if (!batch) {
      printf("batch allocation failed\n");
      abort();
    }
    KWayParams = bestKWayButterflyParams(size, numElementFit, sizeof(T));
    inputReaderManager.init(inputBeginIt, inputEndIt, inAuth, thread_count);
    Z = KWayParams.Z;
    Assert(numElementFit > 8 * Z * thread_count);
    uint64_t tempElementFit = divRoundUp(Z * 8 * (sizeof(WrappedT) + 2), sizeof(WrappedT));
    numElementFit -=
        tempElementFit * thread_count;

    freeTempIndices.init(thread_count);
    for (uint64_t i = 0; i < thread_count; ++i) {
      freeTempIndices.Push(i);
    }
    tempBegin = (uint8_t*)batch + numElementFit * sizeof(WrappedT);
    tempSize = tempElementFit * sizeof(WrappedT);
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
      free(batch);
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
  void KWayButterflySortBasicNonRecursive(Iterator begin, Iterator end, size_t ioLayer,
                              size_t innerLayer) {
    uint64_t numElement = end - begin;
    uint64_t numBucket = numElement / Z;
    
    for (uint64_t layer = 0, stride = 1; layer <= innerLayer; ++layer) {
      uint64_t way = KWayParams.ways[ioLayer][layer];
      uint64_t wayBucket = numBucket / way;
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < wayBucket; ++i) {
        uint64_t tempIdx;
        if (!freeTempIndices.Pop(tempIdx)) {
          printf("freeTempIndices is empty\n");
          abort();
        } // get a temp buffer
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
            uint64_t offset;
            for (offset = 0; offset < numRealPerBucket; ++offset, ++it) {
              if (inputReader->eof()) {
                inputReaderManager.returnRW(inputReader);
                inputReader = inputReaderManager.getRW();
                if (!inputReader) {
                  overFlag = true;
                  // all readers have been consumed, pad dummies to the end
                  break;
                }
              }
              it->setData(inputReader->read(), *inputReader->prng);
            }
            
            for (; offset < Z; ++offset, ++it) {
              it->setDummy();
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
        size_t tempBucketsSize = way * Z * sizeof(WrappedT);
        uint8_t* temp = tempBegin + tempIdx * tempSize; //new WrappedT[way * Z];
        uint8_t* marks = temp + tempBucketsSize;  //new uint8_t[8 * Z];
        MergeSplitKWay(KWayIts, way, Z, (WrappedT*)temp, marks);
        
        freeTempIndices.Push(tempIdx);

        // delete[] temp;
        // delete[] marks;
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
      // uint64_t currTime;
      // ocall_measure_time(&currTime);
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
          if constexpr (task == KWAYBUTTERFLYOSORT) {
            // sort the batch and write to first layer of merge sort
            // const auto cmpVal = [](const auto& a, const auto& b) {
            //   return a.v < b.v;
            // };
            const auto isNotDummy = [](const auto& element) {
                return !element.isDummy();
            };

            auto realEnd = std::partition(batch, batch + bucketThisBatch * Z, isNotDummy);
            // auto realEnd =
            //     partitionDummy(batch, batch + bucketThisBatch * Z);
            // partition dummies to the end
            Assert(realEnd <= batch + numElementFit);
            ParallelSort(batch, realEnd);
            // std::sort(batch, realEnd, cmpVal);
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
      // uint64_t currTime2;
      // ocall_measure_time(&currTime2);
      // uint64_t timediff = currTime2 - currTime;
      // printf("Layer %ld: %d.%d\n", ioLayer, timediff / 1'000'000'000,
      //      timediff % 1'000'000'000);
      #endif
    }
    if constexpr (task == KWAYBUTTERFLYOSORT) {
      mergeSortFirstLayerWriter.flush();
    } else {
      outputWriterManager.flush();
    }
  }

  template <typename Iterator>
  void quickSortParallelRecursive(Iterator begin, Iterator end) {
    auto cmpVal = [](const auto& a, const auto& b) {
      return a.v < b.v;
    };

    if (end - begin > 1) {
      if (end - begin >= 1024) {
        std::nth_element(begin, begin + 5, begin + 10,
          [](const auto& a, const auto& b) { return a.v < b.v; }
        );
        std::iter_swap(begin + 5, end - 1);
        auto pivot = (end - 1);
        auto partitionPoint = std::partition(begin, end - 1,
          [pivot](const auto& e) { return e.v < pivot->v; });
        std::iter_swap(partitionPoint, end - 1);
        #pragma omp taskgroup 
        {
          #pragma omp task shared(begin, partitionPoint, end) untied if (end - begin >= (1<<11))
          quickSortParallelRecursive(begin, partitionPoint);
          #pragma omp task shared(begin, partitionPoint, end) untied if (end - begin >= (1<<11))
          quickSortParallelRecursive(partitionPoint + 1, end); 
          #pragma omp taskyield
        }  
      } else {
        std::sort(begin, end, cmpVal);
      }
    }
  }

  template <typename Iterator>
  void mergeSortParallelRecursive(Iterator begin, Iterator end) {
    auto cmpVal = [](const auto& a, const auto& b) {
      return a.v < b.v;
    };

    if (end - begin > 1) {
      if (end - begin >= 1024) {
        auto mid = begin + (end - begin) / 2;
        #pragma omp taskgroup 
        {
          #pragma omp task shared(begin, mid, end) untied if (end - begin >= (1<<11))
          mergeSortParallelRecursive(begin, mid);
          #pragma omp task shared(begin, mid, end) untied if (end - begin >= (1<<11))
          mergeSortParallelRecursive(mid, end); 
          #pragma omp taskyield
        }  
        std::inplace_merge(begin, mid, end, cmpVal); 
      } else {
        std::sort(begin, end, cmpVal);
      }
    }
  }

  template <typename Iterator>
  void ParallelSort(Iterator begin, Iterator end) {
    #pragma omp parallel
    #pragma omp single
    quickSortParallelRecursive(begin, end);
  }

  void sort() {
    if (batch == NULL) {
      printf("sorter called twice\n");
      abort();
    }
    EM::DynamicPageVector::Vector<TaggedT<T>> v(getOutputSize(),
                                                getBucketSize());
    KWayButterflySort(v.begin(), v.end());
    free(batch);
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

