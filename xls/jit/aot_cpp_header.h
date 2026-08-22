// Copyright 2026 The XLS Authors
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

#ifndef XLS_JIT_AOT_CPP_HEADER_H_
#define XLS_JIT_AOT_CPP_HEADER_H_

#include <string>

#include "absl/status/statusor.h"
#include "xls/jit/aot_entrypoint.pb.h"

namespace xls {

// Options controlling generation of the self-contained C++ header describing
// the packed ABI of an AOT compiled entrypoint.
struct AotCppHeaderOptions {
  // C++ namespace to place the generated code in. May be a (possibly empty)
  // sequence of C++ identifiers separated by "::", e.g.
  // "aot_example::generated".
  std::string cpp_namespace;
  // Stable C++ identifier used as the (nested) namespace dedicated to a single
  // function entrypoint. It does not depend on the DSLX/IR mangling or the
  // symbol salt.
  std::string entrypoint_name;
};

// Generates a self-contained C++20 header textually describing the packed
// (bit-level) ABI of the AOT entrypoints described by `package`.
//
// The generated header only depends on the C++ standard library, does not
// reference XLS/protobuf/Abseil headers or symbols. The packed symbol binding
// uses the GCC/Clang asm-label extension; generated headers require a compiler
// which supports that extension (MSVC is not supported).
//
// Exactly one FUNCTION entrypoint with complete packed ABI metadata must be
// present in `package`.
absl::StatusOr<std::string> GenerateAotCppHeader(
    const AotPackageEntrypointsProto& package,
    const AotCppHeaderOptions& options);

}  // namespace xls

#endif  // XLS_JIT_AOT_CPP_HEADER_H_
