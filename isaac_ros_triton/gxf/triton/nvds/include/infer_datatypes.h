// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef __NVDSINFERSERVER_DATA_TYPES_H__
#define __NVDSINFERSERVER_DATA_TYPES_H__

#include <infer_defines.h>
#include <infer_ioptions.h>
#include <nvdsinfer.h>
#include <stdarg.h>

#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

namespace nvdsinferserver {

enum class InferTensorOrder : int {
    kNone = 0,
    kLinear = 1,
    kNHWC = 2,
};

enum class InferMemType : int {
    kNone = 0,
    kGpuCuda = 1,
    kCpu = 2,
    kCpuCuda = 3,
    kNvSurface = 5,
    kNvSurfaceArray = 6,
};

enum class InferDataType : int {
    kFp32 = FLOAT,  // 0
    kFp16 = HALF,  // 1
    kInt8 = INT8,  // 2
    kInt32 = INT32,  // 3
    kInt16 = 7,
    kUint8,
    kUint16,
    kUint32,
    // New
    kFp64,
    kInt64,
    kUint64,
    kString,  // for text/bytes => str_len(4byte) + str('a\0')
    kBool,
    kNone = -1,
};

enum class InferPostprocessType : int {
    kDetector = 0,
    kClassifier = 1,
    kSegmentation = 2,
    kTrtIsClassifier = 3,
    kOther = 100,
};

enum class InferMediaFormat : int {
    /** 24-bit interleaved R-G-B */
    kRGB = 0,
    /** 24-bit interleaved B-G-R */
    kBGR,
    /** 8-bit Luma */
    kGRAY,
    /** 32-bit interleaved R-G-B-A */
    kRGBA,
    /** 32-bit interleaved B-G-R-x */
    kBGRx,
    kUnknown = -1,
};

// typedef NvDsInferDims InferDims;

struct InferDims
{
  /** Number of dimesions of the layer.*/
  unsigned int numDims = 0;
  /** Size of the layer in each dimension. */
  int d[NVDSINFER_MAX_DIMS] = {0};
  /** Number of elements in the layer including all dimensions.*/
  unsigned int numElements = 0;
};

/**
 * Holds full dimensions (including batch size) for a layer.
 */
struct InferBatchDims
{
    int batchSize = 0;
    InferDims dims;
};

struct InferBufferDescription {
    InferMemType memType;
    long int devId;
    InferDataType dataType;
    InferDims dims;
    uint32_t elementSize;  // per element bytes, except kString(with elementSize
                           // is 0)
    std::string name;
    bool isInput;
};

// Common buffer interface [external]
class IBatchBuffer;
class IBatchArray;
class IOptions;

using SharedIBatchBuffer = std::shared_ptr<IBatchBuffer>;
using SharedIBatchArray = std::shared_ptr<IBatchArray>;
using SharedIOptions = std::shared_ptr<IOptions>;

class IBatchBuffer {
public:
    IBatchBuffer() = default;
    virtual ~IBatchBuffer() = default;
    virtual const InferBufferDescription& getBufDesc() const = 0;
    virtual void* getBufPtr(uint32_t batchIdx) const = 0;
    virtual uint32_t getBatchSize() const = 0;
    virtual uint64_t getTotalBytes() const = 0;

private:
    DISABLE_CLASS_COPY(IBatchBuffer);
};

class IBatchArray {
public:
    IBatchArray() = default;
    virtual ~IBatchArray() = default;
    virtual uint32_t getSize() const = 0;
    virtual const IBatchBuffer* getBuffer(uint32_t arrayIdx) const = 0;
    virtual const IOptions* getOptions() const = 0;

    virtual SharedIBatchBuffer getSafeBuf(uint32_t arrayIdx) const = 0;

    // add values
    virtual void appendIBatchBuf(SharedIBatchBuffer buf) = 0;
    virtual void setIOptions(SharedIOptions o) = 0;

private:
    DISABLE_CLASS_COPY(IBatchArray);
};

struct LayerInfo {
    InferDataType dataType = InferDataType::kFp32;
    InferDims inferDims;
    int bindingIndex = 0;
    bool isInput = 0;
    std::string name;
    // New
    int maxBatchSize;  // 0=> nonBatching
};

} // namespace nvdsinferserver

#endif
