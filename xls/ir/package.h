// Copyright 2020 Google LLC
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

#ifndef THIRD_PARTY_XLS_IR_PACKAGE_H_
#define THIRD_PARTY_XLS_IR_PACKAGE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/container/node_hash_map.h"
#include "absl/strings/string_view.h"
#include "xls/common/integral_types.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/fileno.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

class Function;

class Package {
 public:
  explicit Package(absl::string_view name,
                   absl::optional<absl::string_view> entry = absl::nullopt);

  // Note: functions have parent pointers to their packages, so we don't want
  // them to be moved or copied; this makes Package non-moveable non-copyable.
  Package(const Package& other) = delete;
  Package& operator=(const Package& other) = delete;

  virtual ~Package();

  // Returns whether the given type is one of the types owned by this package.
  bool IsOwnedType(const Type* type) {
    return owned_types_.find(type) != owned_types_.end();
  }
  bool IsOwnedFunctionType(const FunctionType* function_type) {
    return owned_function_types_.find(function_type) !=
           owned_function_types_.end();
  }

  BitsType* GetBitsType(int64 bit_count);
  ArrayType* GetArrayType(int64 size, Type* element_type);
  TupleType* GetTupleType(absl::Span<Type* const> element_types);
  FunctionType* GetFunctionType(absl::Span<Type* const> args_types,
                                Type* return_type);

  // Creates and returned an owned type constructed from the given proto.
  xabsl::StatusOr<Type*> GetTypeFromProto(const TypeProto& proto);
  xabsl::StatusOr<FunctionType*> GetFunctionTypeFromProto(
      const FunctionTypeProto& proto);

  Type* GetTypeForValue(const Value& value);

  Function* AddFunction(std::unique_ptr<Function> f);

  // Get a function by name.
  xabsl::StatusOr<Function*> GetFunction(absl::string_view func_name) const;

  // Remove (dead) functions.
  void DeleteDeadFunctions(absl::Span<Function* const> dead_funcs);

  // Returns the entry function of the package.
  xabsl::StatusOr<Function*> EntryFunction();
  xabsl::StatusOr<const Function*> EntryFunction() const;

  // Returns a new SourceLocation object containing a Fileno and Lineno pair.
  // SourceLocation objects are added to XLS IR nodes and used for debug
  // tracing.
  //
  // An example of how this function  might be used from C++ is shown below:
  // __FILE__ and __LINE__ macros are passed as arguments to a SourceLocation
  // builder and the result is passed to a Node builder method:
  //
  //   SourceLocation loc = package.AddSourceLocation(__FILE__, __LINE__)
  //   function_builder.SomeNodeType(node_args, loc);
  //
  // An alternative front-end could instead call AddSourceLocation with the
  // appropriate metada annotated on the front-end AST.
  //
  // If the file "filename" has been seen before, the Fileno is retrieved from
  // an internal lookup table, otherwise a new Fileno id is generated and added
  // to the table.
  // TODO(dmlockhart): update to use ABSL_LOC and xabsl::SourceLocation.
  SourceLocation AddSourceLocation(absl::string_view filename, Lineno lineno,
                                   Colno colno);

  // Translates a SourceLocation object into a human readable debug identifier
  // of the form: "<source_file_path>:<line_number>".
  std::string SourceLocationToString(const SourceLocation loc);

  // Retrieves the next node ID to assign to a node in the package and
  // increments the next node counter. For use in node construction.
  int64 GetNextNodeId() { return next_node_id_++; }

  // Adds a file to the file-number table and returns its corresponding number.
  // If it already exists, returns the existing file-number entry.
  Fileno GetOrCreateFileno(absl::string_view filename);

  // Returns the total number of nodes in the graph. Traverses the functions and
  // sums the node counts.
  int64 GetNodeCount() const;

  // Returns the functions in this package.
  absl::Span<std::unique_ptr<Function>> functions() {
    return absl::MakeSpan(functions_);
  }
  absl::Span<const std::unique_ptr<Function>> functions() const {
    return functions_;
  }

  const std::string& name() const { return name_; }

  // Returns true if analysis indicates that this package always produces the
  // same value as 'other' when run with the same arguments. The analysis is
  // conservative and false may be returned for some "equivalent" packages.
  bool IsDefinitelyEqualTo(const Package* other) const;

  // Dumps the IR in a parsable text format.
  std::string DumpIr() const;

  std::vector<std::string> GetFunctionNames() const;

  int64 next_node_id() const { return next_node_id_; }

  // Intended for use by the parser when node ids are suggested by the IR text.
  void set_next_node_id(int64 value) { next_node_id_ = value; }

 private:
  friend class FunctionBuilder;

  absl::optional<std::string> entry_;

  #define UnorderedSet std::unordered_set
  #define UnorderedMap std::unordered_map
  #define StableMap std::map

  // Helper that returns a map from the names of functions inside this package
  // to the functions themselves.
  UnorderedMap<std::string, Function*> GetFunctionByName();

  // Name of this package.
  std::string name_;

  // Ordinal to assign to the next node created in this package.
  int64 next_node_id_ = 1;

  std::vector<std::unique_ptr<Function>> functions_;

  // Set of owned types in this package.
  UnorderedSet<const Type*> owned_types_;

  // Set of owned function types in this package.
  UnorderedSet<const FunctionType*> owned_function_types_;

  // Mapping from bit count to the owned "bits" type with that many bits. Use
  // node_hash_map for pointer stability.
  StableMap<int64, BitsType> bit_count_to_type_;

  // Mapping from the size and element type of an array type to the owned
  // ArrayType. Use node_hash_map for pointer stability.
  using ArrayKey = std::pair<int64, const Type*>;
  StableMap<ArrayKey, ArrayType> array_types_;

  // Mapping from elements to the owned tuple type.
  //
  // Uses node_hash_map for pointer stability.
  using TypeVec = absl::InlinedVector<const Type*, 4>;
  StableMap<TypeVec, TupleType> tuple_types_;

  // Mapping from Type:ToString to the owned function type. Use
  // node_hash_map for pointer stability.
  StableMap<std::string, FunctionType> function_types_;

  // Mapping of Fileno ids to string filenames, and vice-versa for reverse
  // lookups. These two data structures must be updated together for consistency
  // and should always contain the same number of entries.
  UnorderedMap<Fileno, std::string> fileno_to_filename_;
  UnorderedMap<std::string, Fileno> filename_to_fileno_;

#undef StableMap
#undef UnorderedMap
#undef UnorderedSet
};

std::ostream& operator<<(std::ostream& os, const Package& package);

}  // namespace xls

#endif  // THIRD_PARTY_XLS_IR_PACKAGE_H_
