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

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

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

class JittedFunctionBase;
// class BlockJitContinuation;
// A buffer & pointers capable of being used as the inputs and/or outputs for
// a jitted function.
class JitArgumentSet {
 public:
  JitArgumentSet(JitArgumentSet&&) = default;
  JitArgumentSet& operator=(JitArgumentSet&&) = default;
  JitArgumentSet(const JitArgumentSet&) = delete;
  JitArgumentSet& operator=(const JitArgumentSet&) = delete;

  // The pointers to the values.
  absl::Span<uint8_t* const> pointers() const { return pointers_; }

  // The raw pointer the jitted code receives.
  const uint8_t* const* get() const { return pointers_.data(); }
  uint8_t* const* get() { return pointers_.data(); }

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

  // Create an argument set with the given alignments and sizes. NB Should only
  // be called by jit-code. The passed in sizes and alignments are not checked
  // against the source.
  static JitArgumentSet CreateInput(const JittedFunctionBase* source,
                                    absl::Span<int64_t const> aligns,
                                    absl::Span<int64_t const> sizes);
  // Create an argument set with the given alignments and sizes. NB Should only
  // be called by jit-code. The passed in sizes and alignments are not checked
  // against the source.
  static JitArgumentSet CreateOutput(const JittedFunctionBase* source,
                                     absl::Span<int64_t const> aligns,
                                     absl::Span<int64_t const> sizes);
  // Create an argument set with the given alignments and sizes. NB Should only
  // be called by jit-code. The passed in sizes and alignments are not checked
  // against the source.
  static absl::StatusOr<JitArgumentSet> CreateInputOutput(
      const JittedFunctionBase* source,
      std::array<absl::Span<int64_t const>, 2> aligns,
      std::array<absl::Span<int64_t const>, 2> sizes);

 private:
  JitArgumentSet(const JittedFunctionBase* source,
                 std::unique_ptr<uint8_t[], DeleteAligned>&& data,
                 std::vector<uint8_t*>&& pointers, bool is_inputs,
                 std::optional<bool> is_outputs = std::nullopt)
      : source_(source),
        data_(std::move(data)),
        pointers_(std::move(pointers)),
        is_inputs_(is_inputs),
        is_outputs_(is_outputs.value_or(!is_inputs_)) {}

  const JittedFunctionBase* source_;
  // Data backing some or all of the pointers.
  //
  // NB For buffers created by combining other buffers this will be null. The
  // actual value here should never be used.
  std::unique_ptr<uint8_t[], DeleteAligned> data_;
  // The pointers to each element.
  std::vector<uint8_t*> pointers_;
  bool is_inputs_;
  bool is_outputs_;

  // To allow the continuation to make combined argument sets.
  friend class BlockJitContinuation;
};

// A wrapper for a temporary buffer which is aligned as required for the jitted
// function.
class JitTempBuffer {
 public:
  explicit JitTempBuffer(const JittedFunctionBase* source, size_t align,
                         size_t size)
      : source_(source), data_(MakeBuffer(align, size)) {}

  const JittedFunctionBase* source() const { return source_; }
  void* get() const { return data_.get(); }

 private:
  std::unique_ptr<uint8_t[], DeleteAligned> MakeBuffer(
      size_t align, size_t size) {
    std::unique_ptr<uint8_t[], DeleteAligned> result(
        absl::bit_cast<uint8_t*>(AllocateAligned(align, size)));
    CHECK(result != nullptr) << "size: " << size << " align: " << align;
    return result;
  }

  const JittedFunctionBase* source_;
  std::unique_ptr<uint8_t[], DeleteAligned> data_;
};


}  // namespace xls

#endif  // XLS_JIT_JIT_BUFFER_H_
