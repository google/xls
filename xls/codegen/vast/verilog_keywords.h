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

#ifndef XLS_CODEGEN_VAST_VERILOG_KEYWORDS_H_
#define XLS_CODEGEN_VAST_VERILOG_KEYWORDS_H_

#include <string>

#include "absl/container/flat_hash_set.h"

namespace xls {

const absl::flat_hash_set<std::string>& VerilogKeywords();
const absl::flat_hash_set<std::string>& SystemVerilogKeywords();

}  // namespace xls

#endif  // XLS_CODEGEN_VAST_VERILOG_KEYWORDS_H_
