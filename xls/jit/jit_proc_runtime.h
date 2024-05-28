// Copyright 2022 The XLS Authors
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

#ifndef XLS_JIT_JIT_PROC_RUNTIME_H_
#define XLS_JIT_JIT_PROC_RUNTIME_H_

#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/package.h"
#include "xls/jit/aot_entrypoint.pb.h"
#include "xls/jit/function_base_jit.h"

namespace xls {

// Create a SerialProcRuntime composed of ProcJits. Supports old-style
// procs.
absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateJitSerialProcRuntime(
    Package* package);

// Create a SerialProcRuntime composed of ProcJits. Constructed from the
// elaboration of the given proc. Supports new-style procs.
absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateJitSerialProcRuntime(
    Proc* top);

struct ProcAotEntrypoints {
  // What proc these entrypoints are associated with.
  Proc* proc;
  // unpacked entrypoint
  JitFunctionType unpacked;
  // packed entrypoint
  std::optional<JitFunctionType> packed = std::nullopt;
};

// Create a SerialProcRuntime composed of ProcJits. Constructed from the
// elaboration of the given proc using the given impls. All procs in the
// elaboration must have an associated entry in the entrypoints and impls lists.
// TODO(allight): Requiring the whole package here makes a lot of things simpler
// but it would be nice to not need to parse the package in the aot case.
absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateAotSerialProcRuntime(
    Proc* top, const AotPackageEntrypointsProto& entrypoints,
    absl::Span<ProcAotEntrypoints const> impls);

// Create a SerialProcRuntime composed of ProcJits. Constructed from the
// elaboration of the given package using the given impls. All procs in the
// elaboration must have an associated entry in the entrypoints and impls lists.
// TODO(allight): Requiring the whole package here makes a lot of things simpler
// but it would be nice to not need to parse the package in the aot case.
absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateAotSerialProcRuntime(
    Package* package, const AotPackageEntrypointsProto& entrypoints,
    absl::Span<ProcAotEntrypoints const> impls);

// Generate AOT code for the given proc elaboration.
absl::StatusOr<JitObjectCode> CreateProcAotObjectCode(Package* package,
                                                      bool with_msan);
// Generate AOT code for the given proc elaboration.
absl::StatusOr<JitObjectCode> CreateProcAotObjectCode(Proc* top,
                                                      bool with_msan);

}  // namespace xls

#endif  // XLS_JIT_JIT_PROC_RUNTIME_H_
