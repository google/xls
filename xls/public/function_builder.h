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

// Public API header that exposes the XLS IR FunctionBuilder APIs with external
// visibility.

#ifndef XLS_PUBLIC_FUNCTION_BUILDER_H_
#define XLS_PUBLIC_FUNCTION_BUILDER_H_

// IWYU pragma: begin_exports
#include "xls/ir/fileno.h"  // IWYU pragma: keep
#include "xls/ir/function_builder.h"
namespace xls {
class Bits;           // IWYU pragma: keep
enum class LsbOrMsb;  // IWYU pragma: keep
struct SourceInfo;    // IWYU pragma: keep
class Type;           // IWYU pragma: keep
}  // namespace xls
// IWYU pragma: end_exports

#endif  // XLS_PUBLIC_FUNCTION_BUILDER_H_
