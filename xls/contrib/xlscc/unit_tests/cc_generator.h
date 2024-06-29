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

#ifndef XLS_CONTRIB_XLSCC_CC_GENERATOR_H_
#define XLS_CONTRIB_XLSCC_CC_GENERATOR_H_

#include <cstdint>
#include <string>

namespace xlscc {
enum class VariableType : std::uint8_t { kAcInt, kAcFixed, kInt };

std::string GenerateTest(uint32_t seed, VariableType type);

}  // namespace xlscc

#endif  // XLS_CONTRIB_XLSCC_CC_GENERATOR_H_
