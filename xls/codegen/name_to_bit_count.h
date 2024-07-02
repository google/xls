// Copyright 2020 The XLS Authors
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

// Helpful typedefs for common mappings that resolve names into bit-values or
// bit-counts.
//
// Note that these types are unordered, so stabilizing sorts must be performed
// on their keys if reproducible traversals are required.

#ifndef XLS_CODEGEN_NAME_TO_BIT_COUNT_H_
#define XLS_CODEGEN_NAME_TO_BIT_COUNT_H_

#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "xls/ir/bits.h"

namespace xls {

using NameToBitCount = absl::flat_hash_map<std::string, int64_t>;
using NameToBits = absl::flat_hash_map<std::string, Bits>;

}  // namespace xls

#endif  // XLS_CODEGEN_NAME_TO_BIT_COUNT_H_
