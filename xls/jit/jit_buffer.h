// Copyright 2023 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLS_JIT_JIT_BUFFER_H_
#define XLS_JIT_JIT_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/jit/type_buffer_metadata.h"

namespace xls {

// Helpers to do the actual allocation, std::aligned_alloc has some odd
// requirements.
void* AllocateAligned(int64_t alignment, int64_t size);
// Helpers to do the actual allocation, std::aligned_alloc has some odd
// requirements.
void DeallocateAligned(void* ptr);
// Helper to call std::free as required for memory allocated with
// std::aligned_alloc
class DeleteAligned {
 public:
  void operator()(void* data) {
    if (data != nullptr) {
      DeallocateAligned(data);
    }
  }
};

// Data structure containing a block of allocated memory and a set of pointers
// which point into the block.
struct JitBuffer {
  using AlignedPtr = std::unique_ptr<uint8_t[], DeleteAligned>;

  AlignedPtr buffer;
  std::vector<uint8_t*> pointers;
};

// Allocate a buffer which can hold the types described by the given
// metadata. The `pointers` in the returned JitBuffer correspond to the elements
// in `metadata`. If zero is true, the buffer is zero-ed out.
JitBuffer AllocateAlignedBuffer(absl::Span<const TypeBufferMetadata> metadata,
                                bool zero = false);

class JittedFunctionBase;

// A buffer & pointers capable of being used as the inputs and/or outputs for a
// jitted function. The data structure includes tags (source, is_input,
// is_output) to check it is being used correctly. The base JitArgumentSet
// type does not own the underlying allocated memory.
class JitArgumentSet {
 public:
  explicit JitArgumentSet(const JittedFunctionBase* source,
                          std::vector<uint8_t*> pointers, bool is_inputs,
                          bool is_outputs)
      : source_(source),
        pointers_(std::move(pointers)),
        is_inputs_(is_inputs),
        is_outputs_(is_outputs) {}
  virtual ~JitArgumentSet() = default;

  // Return the pointers to the elements within the buffer. Each corresponds to
  // a single XLS value.
  absl::Span<uint8_t* const> get_element_pointers() const { return pointers_; }
  absl::Span<uint8_t*> get_element_pointers() {
    return absl::MakeSpan(pointers_);
  }

  // Return a pointer to the array of elements pointers. This is what the jitted
  // code receives.
  const uint8_t* const* get_base_pointer() const { return pointers_.data(); }
  uint8_t* const* get_base_pointer() { return pointers_.data(); }

  // What function this was created for. May only be used on this function.
  const JittedFunctionBase* source() const { return source_; }
  // Is this buffer acceptable as the inputs set for the function.
  //
  // NB a single buffer might be acceptable as both inputs and outputs. It
  // should only ever be passed in one of these slots however.
  bool is_inputs() const { return is_inputs_; }
  // Is this buffer acceptable as the outputs set for the function.
  //
  // NB a single buffer might be acceptable as both inputs and outputs. It
  // should only ever be passed in one of these slots however.
  bool is_outputs() const { return is_outputs_; }

 protected:
  const JittedFunctionBase* source_;

  // Pointers to buffers. Not necessarily owned by this object.
  std::vector<uint8_t*> pointers_;
  bool is_inputs_;
  bool is_outputs_;
};

// A JitArgumentSet derived class which owns the underlying buffer.
class JitArgumentSetOwnedBuffer : public JitArgumentSet {
 public:
  explicit JitArgumentSetOwnedBuffer(const JittedFunctionBase* source,
                                     JitBuffer&& buffer, bool is_inputs,
                                     bool is_outputs)
      : JitArgumentSet(source, buffer.pointers, is_inputs, is_outputs),
        buffer_(std::move(buffer.buffer)) {}

  ~JitArgumentSetOwnedBuffer() override = default;

  // Create an argument set with the given alignments and sizes. NB Should
  // only be called by jit-code. The passed in sizes and alignments are not
  // checked against the source.
  //
  // If zero then zero-initialize the buffers.
  static std::unique_ptr<JitArgumentSetOwnedBuffer> CreateInput(
      const JittedFunctionBase* source,
      absl::Span<const TypeBufferMetadata> metadata, bool zero = false);
  // Create an argument set with the given alignments and sizes. NB Should only
  // be called by jit-code. The passed in sizes and alignments are not checked
  // against the source.
  static std::unique_ptr<JitArgumentSetOwnedBuffer> CreateOutput(
      const JittedFunctionBase* source,
      absl::Span<const TypeBufferMetadata> metadata);
  // Create an argument set with the given alignments and sizes. NB Should only
  // be called by jit-code. The passed in sizes and alignments are not checked
  // against the source.
  static absl::StatusOr<std::unique_ptr<JitArgumentSetOwnedBuffer>>
  CreateInputOutput(const JittedFunctionBase* source,
                    absl::Span<const TypeBufferMetadata> input_metadata,
                    absl::Span<const TypeBufferMetadata> output_metadata);

 private:
  JitBuffer::AlignedPtr buffer_;
};

// A wrapper for a temporary buffer which is aligned as required for the jitted
// function.
class JitTempBuffer {
 public:
  explicit JitTempBuffer(const JittedFunctionBase* source, size_t align,
                         size_t size)
      : source_(source), data_(MakeBuffer(align, size)) {}

  const JittedFunctionBase* source() const { return source_; }
  void* get_base_pointer() const { return data_.get(); }

 private:
  JitBuffer::AlignedPtr MakeBuffer(size_t align, size_t size) {
    JitBuffer::AlignedPtr result(
        absl::bit_cast<uint8_t*>(AllocateAligned(align, size)));
    CHECK(result != nullptr) << "size: " << size << " align: " << align;
    return result;
  }

  const JittedFunctionBase* source_;
  JitBuffer::AlignedPtr data_;
};

}  // namespace xls

#endif  // XLS_JIT_JIT_BUFFER_H_
