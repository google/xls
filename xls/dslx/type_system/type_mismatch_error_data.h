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

#ifndef XLS_DSLX_TYPE_SYSTEM_TYPE_MISMATCH_ERROR_DATA_H_
#define XLS_DSLX_TYPE_SYSTEM_TYPE_MISMATCH_ERROR_DATA_H_

#include <memory>
#include <string>

#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/concrete_type.h"

namespace xls::dslx {

// Holds structured data on a type mismatch error that occurred during the type
// checking process.
struct TypeMismatchErrorData {
  Span error_span;
  const AstNode* lhs_node;
  std::unique_ptr<Type> lhs;
  const AstNode* rhs_node;
  std::unique_ptr<Type> rhs;
  std::string message;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_TYPE_MISMATCH_ERROR_DATA_H_
