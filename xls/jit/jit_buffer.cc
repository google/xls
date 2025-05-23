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

#include "xls/jit/jit_buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/jit/type_buffer_metadata.h"

namespace xls {

JitBuffer AllocateAlignedBuffer(absl::Span<const TypeBufferMetadata> metadata,
                                bool zero) {
  static_assert(sizeof(int64_t) >= sizeof(intptr_t),
                "More than 64 bit pointers");
  if (metadata.empty()) {
    return JitBuffer{.buffer = JitBuffer::AlignedPtr(),
                     .pointers = std::vector<uint8_t*>{}};
  }
  int64_t max_align =
      absl::c_max_element(metadata, [](const TypeBufferMetadata& a,
                                       const TypeBufferMetadata& b) {
        return a.preferred_alignment < b.preferred_alignment;
      })->preferred_alignment;
  std::vector<int64_t> offsets;
  offsets.reserve(metadata.size());
  offsets.push_back(0);
  for (int64_t i = 1; i < metadata.size(); ++i) {
    int64_t cur_idx = offsets.back() + metadata[i - 1].size;
    offsets.push_back(
        RoundUpToNearest(cur_idx, metadata[i].preferred_alignment));
  }
  int64_t total_size = offsets.back() + metadata.back().size;
  if (total_size == 0) {
    // Leave with nullptr to catch illegal accesses if the size is 0.
    return JitBuffer{
        .buffer = JitBuffer::AlignedPtr(),
        .pointers = std::vector<uint8_t*>(metadata.size(), nullptr)};
  }
  std::vector<uint8_t*> ptrs;
  uint8_t* buffer =
      absl::bit_cast<uint8_t*>(AllocateAligned(max_align, total_size));
  CHECK(buffer != nullptr) << "Unable to allocate. align:" << max_align
                           << " size: " << total_size;
  if (zero) {
    memset(buffer, 0, total_size);
  }
  ptrs.reserve(offsets.size());
  absl::c_transform(offsets, std::back_inserter(ptrs),
                    [&](int64_t p) { return buffer + p; });
  return JitBuffer{.buffer = JitBuffer::AlignedPtr(buffer),
                   .pointers = std::move(ptrs)};
}

std::unique_ptr<JitArgumentSetOwnedBuffer>
JitArgumentSetOwnedBuffer::CreateInput(
    const JittedFunctionBase* source,
    absl::Span<const TypeBufferMetadata> metadata, bool zero) {
  return std::make_unique<JitArgumentSetOwnedBuffer>(
      source, AllocateAlignedBuffer(metadata, zero),
      /*is_inputs=*/true,
      /*is_outputs=*/false);
}

std::unique_ptr<JitArgumentSetOwnedBuffer>
JitArgumentSetOwnedBuffer::CreateOutput(
    const JittedFunctionBase* source,
    absl::Span<const TypeBufferMetadata> metadata) {
  return std::make_unique<JitArgumentSetOwnedBuffer>(
      source, AllocateAlignedBuffer(metadata),
      /*is_inputs=*/false,
      /*is_outputs=*/true);
}

absl::StatusOr<std::unique_ptr<JitArgumentSetOwnedBuffer>>
JitArgumentSetOwnedBuffer::CreateInputOutput(
    const JittedFunctionBase* source,
    absl::Span<const TypeBufferMetadata> input_metadata,
    absl::Span<const TypeBufferMetadata> output_metadata) {
  XLS_RET_CHECK(absl::c_equal(input_metadata, output_metadata));
  JitBuffer buf = AllocateAlignedBuffer(input_metadata);
  return std::make_unique<JitArgumentSetOwnedBuffer>(source, std::move(buf),
                                                     /*is_inputs=*/true,
                                                     /*is_outputs=*/true);
}

void* AllocateAligned(int64_t alignment, int64_t size) {
  // https://en.cppreference.com/w/c/memory/aligned_alloc
  // https://en.cppreference.com/w/c/memory/malloc
  //
  // Aligned_alloc is annoying since C14 had some issues with the specification
  // and ASAN (and possibly some other things) still use that as the source of
  // the implementation.
  //
  // malloc returns fundamental alignment. Just use it if we can. This avoids
  // issue where some aligned_allocs return nullptr if given alignments smaller
  // than the fundamental alignment.
  CHECK_GE(alignment, 0);
  CHECK(IsPowerOfTwo(static_cast<uint64_t>(alignment)))
      << "Alignment must be power of 2";
  if (alignment <= std::alignment_of_v<std::max_align_t>) {
    return std::malloc(size);
  }

  // Aligned_alloc (prior to 17) requires that size be a multiple of alignment.
  // This requirement is in place for ASAN and some other allocators so honor
  // it.
  return std::aligned_alloc(alignment, RoundUpToNearest(size, alignment));
}

void DeallocateAligned(void* ptr) { std::free(ptr); }
}  // namespace xls
