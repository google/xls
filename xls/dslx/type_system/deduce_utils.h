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

#ifndef XLS_DSLX_TYPE_SYSTEM_DEDUCE_UTILS_H_
#define XLS_DSLX_TYPE_SYSTEM_DEDUCE_UTILS_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

// Validates that "number" fits in `bits_like` if the size is known. `bits_like`
// should have been derived from `type` -- this routine uses `type` for error
// reporting.
absl::Status TryEnsureFitsInType(const Number& number,
                                 const BitsLikeProperties& bits_like,
                                 const Type& type);

// Validates whether the purported array being indexed by an `Index` operation
// is really a container that's allowed to be indexed like an array. Note that
// the hard assumption that it must be an actual `ArrayType` is only in the v1
// consumer of this function, which uses the `ArrayType` to produce the element
// type.
absl::Status ValidateArrayTypeForIndex(const Index& node, const Type& type,
                                       const FileTable& file_table);

// Validates whether the purported tuple being indexed by a `TupleIndex`
// operation is really a tuple.
absl::Status ValidateTupleTypeForIndex(const TupleIndex& node, const Type& type,
                                       const FileTable& file_table);

// Validates the index expression for an array index operation that is actually
// an index and not a slice.
absl::Status ValidateArrayIndex(const Index& node, const Type& array_type,
                                const Type& index_type, const TypeInfo& ti,
                                const FileTable& file_table);

// Validates the index expression for a tuple index operation.
absl::Status ValidateTupleIndex(const TupleIndex& node, const Type& tuple_type,
                                const Type& index_type, const TypeInfo& ti,
                                const FileTable& file_table);

// Returns whether "e" is a NameRef referring to the given "name_def".
bool IsNameRefTo(const Expr* e, const NameDef* name_def);

// Checks that "number" can legitmately conform to type "type".
absl::Status ValidateNumber(const Number& number, const Type& type);

// Checks that the given argument being formatted by a macro like `trace_fmt!`
// or `vtrace_fmt!` is actually possible to format as a string.
absl::Status ValidateFormatMacroArgument(const Type& type, const Span& span,
                                         const FileTable& file_table);

// Finds the Proc identified by the given node (either NameRef or ColonRef),
// using the associated ImportData for import Module lookup.
// The target proc must have been typechecked prior to this call.
absl::StatusOr<Proc*> ResolveProc(Expr* callee, const TypeInfo* type_info);

// Normalizes the given optional start and width values for a slice of a bit
// vector of size `bit_count`. One or both values may be omitted (to indicate
// the absolute start or end), negative (i.e. an end-based index), or out of
// range. This function will produce positive, in-range values.
absl::StatusOr<StartAndWidth> ResolveBitSliceIndices(
    int64_t bit_count, std::optional<int64_t> start_opt,
    std::optional<int64_t> limit_opt);

// Checks that the number of tuple elements in the name def tree matches the
// number of tuple elements in the type; if a "rest of tuple" leaf is
// present, only one is allowed, and it is not counted in the number of names.
//
// Returns the number of tuple elements (first) and the number of names that
// will be bound in the given NameDefTree (second).
//
// The latter may be less than the former if there is a "rest of tuple" leaf.
using TupleTypeOrAnnotation =
    std::variant<const TupleType*, const TupleTypeAnnotation*>;
absl::StatusOr<std::pair<int64_t, int64_t>> GetTupleSizes(
    const NameDefTree* name_def_tree, TupleTypeOrAnnotation tuple_type);

// Typechecks the name def tree items against type, and recursively processes
// the node/type pairs according to the `process_tuple_member` function.
// If `constexpr_value` is provided for the tuple, the appropriate
// subvalue will also be passed into `process_tuple_member`.
//
// For example:
//
//    (a, (b, c))  vs (u8, (u4, u2))
//
// Will call `process_tuple_member` with the following arguments:
//
//    (a, u8, ...)
//    (b, u4, ...)
//    (c, u2, ...)
//
using TypeOrAnnotation = std::variant<const Type*, const TypeAnnotation*>;
absl::Status MatchTupleNodeToType(
    std::function<absl::Status(AstNode*, TypeOrAnnotation,
                               std::optional<InterpValue>)>
        process_tuple_member,
    const NameDefTree* name_def_tree, TypeOrAnnotation type,
    const FileTable& file_table, std::optional<InterpValue> constexpr_value);

// Returns true if the cast-conversion from "from" to "to" is acceptable (i.e.
// should not cause a type error to occur).
bool IsAcceptableCast(const Type& from, const Type& to);

// Notes the constexpr value for a builtin function invocation in `ti` if
// applicable.
absl::Status NoteBuiltinInvocationConstExpr(std::string_view fn_name,
                                            const Invocation* invocation,
                                            const FunctionType& fn_type,
                                            TypeInfo* ti,
                                            ImportData* import_data);

// Returns the TypeInfo for the given node, preferring the current TypeInfo if
// the node is in the same module, otherwise giving the root TypeInfo for
// the node's module.
const TypeInfo& GetTypeInfoForNodeIfDifferentModule(
    AstNode* node, const TypeInfo& current_type_info,
    const ImportData& import_data);

// It's common to accidentally use different constant naming conventions
// coming from other environments -- warn folks if it's not following
// https://doc.rust-lang.org/1.0.0/style/style/naming/README.html
void WarnOnInappropriateConstantName(std::string_view identifier,
                                     const Span& span, const Module& module,
                                     WarningCollector* warning_collector);

// Gets the total bit count of the given `type` as a u32 `InterpValue`.
absl::StatusOr<InterpValue> GetBitCountAsInterpValue(const Type* type);

// Gets the element count of the given `type` as a u32 `InterpValue`. The
// element count depends on the kind of type:
//
//  kind          element count
//  -----------------------------
//  bits-like     total bit count
//  array         number of elements indexable by first index
//  tuple         number of top-level members
//  struct        number of top-level members
absl::StatusOr<InterpValue> GetElementCountAsInterpValue(const Type* type);

// Gets the override value of the given `type` as a `InterpValue`. The override
// value is either a string representation of a boolean, integer, or float.
absl::StatusOr<InterpValue> GetConfiguredValueAsInterpValue(
    std::string override_value, const Type* type, const TypeInfo* type_info,
    ImportData* import_data, const Span& span);

// Returns the patterns of the arm as a string.
std::string PatternsToString(const MatchArm* arm);

// Returns an error if the array dimension is greater than 2^31-1.
absl::Status CheckArrayDimTooLarge(Span span, uint64_t dim,
                                   const FileTable& file_table);
}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_DEDUCE_UTILS_H_
