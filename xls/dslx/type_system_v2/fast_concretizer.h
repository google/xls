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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_FAST_CONCRETIZER_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_FAST_CONCRETIZER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {

// An object that can concretize certain simple type annotations with no
// resolution or evaluation.
class FastConcretizer {
 public:
  static std::unique_ptr<FastConcretizer> Create(const FileTable& file_table);

  virtual ~FastConcretizer() = default;

  // Concretizes the given type annotation if it requires no resolution or
  // evaluation. If more sophisticated logic is required to concretize it, this
  // function returns an error. Therefore, the caller should generally check for
  // the error and invoke other logic, rather than using assign-or-return
  // semantics.
  virtual absl::StatusOr<std::unique_ptr<Type>> Concretize(
      const TypeAnnotation* annotation) = 0;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_FAST_CONCRETIZER_H_
