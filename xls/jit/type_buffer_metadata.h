
// Copyright 2025 The XLS Authors
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

#ifndef XLS_JIT_TYPE_BUFFER_METADATA_H_
#define XLS_JIT_TYPE_BUFFER_METADATA_H_

#include <cstdint>
#include <string>

namespace xls {

// Data structure describing the characteristics of a buffer holding a single
// value of a particular XLS type.
struct TypeBufferMetadata {
  int64_t size;
  int64_t preferred_alignment;
  int64_t abi_alignment;
  int64_t packed_size;

  bool operator==(const TypeBufferMetadata& other) const {
    return size == other.size &&
           preferred_alignment == other.preferred_alignment &&
           abi_alignment == other.abi_alignment;
  }
  std::string ToString() const;
};

}  // namespace xls

#endif  // XLS_JIT_TYPE_BUFFER_METADATA_H_
