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

#ifndef XLS_DSLX_CPP_TRANSPILER_CPP_EMITTER_H_
#define XLS_DSLX_CPP_TRANSPILER_CPP_EMITTER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

// Sanitizes the given name for C++. C++ keywords are prefixed with "_".
std::string SanitizeCppName(std::string_view name);

// Returns the C++ type name used to represent the given DSLX type name.
std::string DslxTypeNameToCpp(std::string_view dslx_type);

// A class which handles generation of snippets of C++ code for a particular
// type which may be represented with a TypeAnnotation (e.g., array, tuple,
// bit-vector).
//
// Code-emitting methods take a `nesting` argument which is the nesting depth of
// the emitter calls (emitters can call other emitters) and is used to generate
// non-conflicting variable names.
class CppEmitter {
 public:
  explicit CppEmitter(std::string_view cpp_type, std::string_view dslx_type)
      : cpp_type_(cpp_type), dslx_type_(dslx_type) {}
  virtual ~CppEmitter() = default;

  // Emits c++ code which assigns `rhs` of xls::Value type to `lhs` of
  // `cpp_type()`.
  virtual std::string AssignToValue(std::string_view lhs, std::string_view rhs,
                                    int64_t nesting) const = 0;

  // Emits and returns c++ code which assigns `rhs` of `cpp_type()` to `lhs` of
  // xls::Value type.
  virtual std::string AssignFromValue(std::string_view lhs,
                                      std::string_view rhs,
                                      int64_t nesting) const = 0;

  // Emits and returns c++ code which verifies that `identifier` of type
  // `cpp_type()` is properly formed. `name` is a descriptive string (e.g., type
  // or field name) which can be used in an error message. The code will raise
  // an error, for example, if the values in the c++ type (e.g., uint8_t) does
  // not fit in the underlying DSLX type (e.g., u3).
  virtual std::string Verify(std::string_view identifier, std::string_view name,
                             int64_t nesting) const = 0;

  // Emits and returns c++ code which appends a string representation of
  // `identifier` of type `cpp_type()` to the std::string with the name
  // `str_to_append`. `indent_amount` is the name of a int variable holding the
  // indentation level.  Bit-vector values (leaves) are emitted as unsigned hex
  // numbers. For example, an emitter for DSLX type `u42` will emit c++ code
  // which generates strings like:
  //
  //   bits[42]:0x1234
  //
  // An emitter for a tuple type might emit C++ which generate strings like:
  //
  //   (bits[1]:, bits[32]:0x12, MyEnum::kSomething)
  //
  // TODO(meheff): Make this emit a form which is parsable as a C++ designated
  // initializer. For example:
  //
  //   Foo {
  //     .Bar = 0x42,
  //     .Baz = 0x123,
  //   }
  virtual std::string ToString(std::string_view str_to_append,
                               std::string_view indent_amount,
                               std::string_view identifier,
                               int64_t nesting) const = 0;

  // Emits and returns c++ code which appends a valid DSLX representation of
  // `identifier` of type `cpp_type()` to the std::string with the name
  // `str_to_append`. `indent_amount` is the name of a int variable holding the
  // indentation level. For example, an emitter for DSLX type `u42` will emit
  // c++ code which generates strings like:
  //
  //   u42:0x1234
  //
  // An emitter for a tuple type might emit C++ which generate strings like:
  //
  //   (bool: false, u32:0x12, MyEnum::kSomething)
  virtual std::string ToDslxString(std::string_view str_to_append,
                                   std::string_view indent_amount,
                                   std::string_view identifier,
                                   int64_t nesting) const = 0;

  // If the underlying DSLX type is a bit vector then return its bit
  // count. Otherwise return std::nullopt.
  virtual std::optional<int64_t> GetBitCountIfBitVector() const {
    return std::nullopt;
  }

  // If the underlying DSLX type is a bit vector then return its signedness.
  // Otherwise return std::nullopt.
  virtual std::optional<bool> GetSignednessIfBitVector() const {
    return std::nullopt;
  }

  // Returns the C++ type handled by this emitter.
  std::string_view cpp_type() const { return cpp_type_; }

  // Returns true if the c++ type is primitive (e.g., uint16_t).
  bool IsCppPrimitiveType() const {
    return GetBitCountIfBitVector().has_value();
  }

  // Returns the name of the underlying DSLX type.
  std::string dslx_type() const { return dslx_type_; }

  // Factory which creates and returns a CppEmitter for the DSLX type given by
  // `type_annotation`.
  static absl::StatusOr<std::unique_ptr<CppEmitter>> Create(
      const TypeAnnotation* type_annotation, std::string_view dslx_type,
      TypeInfo* type_info, ImportData* import_data);

 private:
  std::string cpp_type_;
  std::string dslx_type_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_CPP_TRANSPILER_CPP_EMITTER_H_
