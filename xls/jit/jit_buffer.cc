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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

namespace xls {

namespace {
using AlignedPtr = std::unique_ptr<uint8_t[], DeleteAligned>;
// This allocates the actual memory that's used in the jit-argument-set and
// returns it and the pointers into it that make up each argument.
std::pair<AlignedPtr, std::vector<uint8_t*>> AllocateAlignedBuffer(
    absl::Span<int64_t const> sizes, absl::Span<int64_t const> alignments) {
  static_assert(sizeof(int64_t) >= sizeof(intptr_t),
                "More than 64 bit pointers");
  CHECK_EQ(sizes.size(), alignments.size());
  if (alignments.empty()) {
    return {AlignedPtr(), std::vector<uint8_t*>{}};
  }
  int64_t max_align = *absl::c_max_element(alignments);
  std::vector<int64_t> offsets;
  offsets.reserve(sizes.size());
  offsets.push_back(0);
  for (int64_t i = 1; i < sizes.size(); ++i) {
    int64_t cur_idx = offsets.back() + sizes[i - 1];
    offsets.push_back(RoundUpToNearest(cur_idx, alignments[i]));
  }
  int64_t total_size = offsets.back() + sizes.back();
  if (total_size == 0) {
    // Leave with nullptr to catch illegal accesses if the size is 0.
    return {AlignedPtr(), std::vector<uint8_t*>(sizes.size(), nullptr)};
  }
  std::vector<uint8_t*> ptrs;
  uint8_t* buffer =
      absl::bit_cast<uint8_t*>(AllocateAligned(max_align, total_size));
  CHECK(buffer != nullptr) << "Unable to allocate. align:" << max_align
                           << " size: " << total_size;
  ptrs.reserve(offsets.size());
  absl::c_transform(offsets, std::back_inserter(ptrs),
                    [&](int64_t p) { return buffer + p; });
  return {AlignedPtr(buffer), std::move(ptrs)};
}
}  // namespace

JitArgumentSet JitArgumentSet::CreateInput(const JittedFunctionBase* source,
                                           absl::Span<int64_t const> aligns,
                                           absl::Span<int64_t const> sizes) {
  auto [buf, ptr] = AllocateAlignedBuffer(sizes, aligns);
  return JitArgumentSet(source, std::move(buf), std::move(ptr),
                        /*is_input=*/true, /*is_output=*/false);
}
JitArgumentSet JitArgumentSet::CreateOutput(const JittedFunctionBase* source,
                                            absl::Span<int64_t const> aligns,
                                            absl::Span<int64_t const> sizes) {
  auto [buf, ptr] = AllocateAlignedBuffer(sizes, aligns);
  return JitArgumentSet(source, std::move(buf), std::move(ptr),
                        /*is_input=*/false, /*is_output=*/true);
}
absl::StatusOr<JitArgumentSet> JitArgumentSet::CreateInputOutput(
    const JittedFunctionBase* source,
    std::array<absl::Span<int64_t const>, 2> aligns,
    std::array<absl::Span<int64_t const>, 2> sizes) {
  XLS_RET_CHECK(absl::c_equal(sizes[0], sizes[1]));
  XLS_RET_CHECK(absl::c_equal(aligns[0], aligns[1]));
  auto [buf, ptr] = AllocateAlignedBuffer(sizes[0], aligns[0]);
  return JitArgumentSet(source, std::move(buf), std::move(ptr),
                        /*is_input=*/true, /*is_output=*/true);
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
