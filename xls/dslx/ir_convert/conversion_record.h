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

#ifndef XLS_DSLX_IR_CONVERSION_RECORD_H_
#define XLS_DSLX_IR_CONVERSION_RECORD_H_

#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/proc_id.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

// Record used in sequence, noting order functions should be converted in.
//
// Describes a function instance that should be emitted (in an order determined
// by an encapsulating sequence). Annotated with metadata that describes the
// call graph instance.
//
// Attributes:
//   f: Function AST node to convert.
//   module: Module that f resides in.
//   type_info: Node to type mapping for use in converting this
//     function instance.
//   callees: Function names that 'f' calls.
//   parametric_env: Parametric bindings for this function instance.
//   callees: Functions that this instance calls.
class ConversionRecord {
 public:
  // Note: performs ValidateParametrics() to potentially return an error status.
  static absl::StatusOr<ConversionRecord> Make(
      Function* f, const Invocation* invocation, Module* module,
      TypeInfo* type_info, ParametricEnv parametric_env,
      std::optional<ProcId> proc_id, bool is_top);

  // Integrity-checks that the parametric_env provided are sufficient to
  // instantiate f (i.e. if it is parametric). Returns an internal error status
  // if they are not sufficient.
  static absl::Status ValidateParametrics(Function* f,
                                          const ParametricEnv& parametric_env);

  Function* f() const { return f_; }
  const Invocation* invocation() const { return invocation_; }
  Module* module() const { return module_; }
  TypeInfo* type_info() const { return type_info_; }
  const ParametricEnv& parametric_env() const { return parametric_env_; }
  std::optional<ProcId> proc_id() const { return proc_id_; }
  bool IsTop() const { return is_top_; }

  std::string ToString() const;

 private:
  ConversionRecord(Function* f, const Invocation* invocation, Module* module,
                   TypeInfo* type_info, ParametricEnv parametric_env,
                   std::optional<ProcId> proc_id, bool is_top)
      : f_(f),
        invocation_(invocation),
        module_(module),
        type_info_(type_info),
        parametric_env_(std::move(parametric_env)),
        proc_id_(std::move(proc_id)),
        is_top_(is_top) {}

  Function* f_;
  const Invocation* invocation_;
  Module* module_;
  TypeInfo* type_info_;
  ParametricEnv parametric_env_;
  std::optional<ProcId> proc_id_;
  bool is_top_;
};

std::string ConversionRecordsToString(
    absl::Span<const ConversionRecord> records);

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERSION_RECORD_H_
