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

#ifndef XLS_DSLX_CPP_TRANSPILER_CPP_TYPE_GENERATOR_H_
#define XLS_DSLX_CPP_TRANSPILER_CPP_TYPE_GENERATOR_H_

#include <memory>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

struct CppSource {
  std::string header;
  std::string source;
};

// Class for generating a C++ type and associated support code which mirrors a
// particular DSLX type. The C++ types may be enums, structs, or type
// aliases. There is a CppTypeGenerator defined for the following
// dslx::TypeDefinition kinds: type aliases, structs, enums.
class CppTypeGenerator {
 public:
  explicit CppTypeGenerator(std::string_view cpp_type,
                            std::string_view dslx_type)
      : cpp_type_(cpp_type), dslx_type_(dslx_type) {}
  virtual ~CppTypeGenerator() = default;

  // Generates the C++ source code (header and source) defining the type.
  virtual absl::StatusOr<CppSource> GetCppSource() const = 0;

  // Returns the C++ type used to represent the underlying DSLX type.
  std::string_view cpp_type() const { return cpp_type_; }

  // Returns the name of the underlying DSLX type if it is a named type (e.g.,
  // not a tuple or array).
  std::string dslx_type() const { return dslx_type_; }

  // Returns a type generator for the given TypeDefinition.
  static absl::StatusOr<std::unique_ptr<CppTypeGenerator>> Create(
      const TypeDefinition& type_definition, TypeInfo* type_info,
      ImportData* import_data);

 protected:
  std::string cpp_type_;
  std::string dslx_type_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_CPP_TRANSPILER_CPP_TYPE_GENERATOR_H_
