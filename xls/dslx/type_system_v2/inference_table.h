// Copyright 2024 The XLS Authors
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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_INFERENCE_TABLE_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_INFERENCE_TABLE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

// The kinds of variables that can be defined in an `InferenceTable`.
enum class InferenceVariableKind : uint8_t { kInteger, kBool, kType };

// A table that facilitates a type inference algorithm where unknowns during the
// course of inference are represented using variables (which we call "inference
// variables"). An inference variable may be internally fabricated by the
// inference system, or it may be a parametric named in the DSLX source code.
//
// The type inference system that uses the table defines variables via
// `DefineXXXVariable` functions, getting back `NameRef` objects for them, which
// it can then use in expressions if desired. The inference system also stores
// the inferred type of each processed AST node in the table, in the form of
// either a `TypeAnnotation` (when specified explicitly in the source code) or
// a defined inference variable.
//
// Once the inference system is finished populating the table, there should be
// enough information in it to concretize the stored type of every node, i.e.
// turn it into a `Type` object with concrete information. This can be done via
// the `InferenceTableToTypeInfo` utility.
class InferenceTable {
 public:
  virtual ~InferenceTable();

  // Creates an empty inference table for the given module.
  static std::unique_ptr<InferenceTable> Create(Module& module,
                                                const FileTable& file_table);

  // Defines an inference variable fabricated by the type inference system,
  // which has no direct representation in the DSLX source code that is being
  // analyzed. It is up to the inference system using the table to decide a
  // naming scheme for such variables.
  virtual absl::StatusOr<NameRef*> DefineInternalVariable(
      InferenceVariableKind kind, AstNode* definer, std::string_view name) = 0;

  // Sets the type variable associated with `node`. The `type` must refer to a
  // type variable previously defined in this table. This can serve as a way to
  // constrain one or more nodes to match some unknown type, such as the left
  // and right sides of a `+` operation.
  virtual absl::Status SetTypeVariable(const AstNode* node,
                                       const NameRef* type) = 0;

  // Sets the explicit type annotation associated with `node`. Not all nodes
  // have one. For example, a `Let` node like `let x:u32 = something;` has a
  // type annotation, but `let x = something;` does not.
  virtual absl::Status SetTypeAnnotation(const AstNode* node,
                                         const TypeAnnotation* type) = 0;

  // Returns all the nodes that have information stored in the table, in the
  // order the table first became aware of the nodes.
  virtual const std::vector<const AstNode*>& GetNodes() const = 0;

  // Returns the type annotation for `node` in the table, if any.
  virtual std::optional<const TypeAnnotation*> GetTypeAnnotation(
      const AstNode* node) const = 0;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_INFERENCE_TABLE_H_
