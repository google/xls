// Copyright 2020 The XLS Authors
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

#ifndef XLS_IR_PACKAGE_H_
#define XLS_IR_PACKAGE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/fileno.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

class Block;
class Channel;
class Function;
class FunctionBase;
class Proc;
class SingleValueChannel;
class StreamingChannel;

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

  BitsType* GetBitsType(int64_t bit_count);
  ArrayType* GetArrayType(int64_t size, Type* element_type);
  TupleType* GetTupleType(absl::Span<Type* const> element_types);
  TokenType* GetTokenType();
  FunctionType* GetFunctionType(absl::Span<Type* const> args_types,
                                Type* return_type);

  // Returns a pointer to a type owned by this package that is of the same
  // type as 'other_package_type', which may be owned by another package.
  absl::StatusOr<Type*> MapTypeFromOtherPackage(Type* other_package_type);

  // Creates and returned an owned type constructed from the given proto.
  absl::StatusOr<Type*> GetTypeFromProto(const TypeProto& proto);
  absl::StatusOr<FunctionType*> GetFunctionTypeFromProto(
      const FunctionTypeProto& proto);

  Type* GetTypeForValue(const Value& value);

  // Add a function (proc/block) to the package. Ownership is tranferred to the
  // package.
  Function* AddFunction(std::unique_ptr<Function> f);
  Proc* AddProc(std::unique_ptr<Proc> proc);
  Block* AddBlock(std::unique_ptr<Block> block);

  // Get a function (or proc) by name. Returns an error if no such function/proc
  // of the indicated kind (Function or Proc) exists with that name.
  absl::StatusOr<Function*> GetFunction(absl::string_view func_name) const;
  absl::StatusOr<Proc*> GetProc(absl::string_view proc_name) const;
  absl::StatusOr<Block*> GetBlock(absl::string_view block_name) const;

  // Remove (dead) functions.
  void DeleteDeadFunctions(absl::Span<Function* const> dead_funcs);

  // Returns the entry function of the package.
  absl::StatusOr<Function*> EntryFunction();
  absl::StatusOr<const Function*> EntryFunction() const;

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
  // appropriate metadata annotated on the front-end AST.
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
  int64_t GetNextNodeId() { return next_node_id_++; }

  // Adds a file to the file-number table and returns its corresponding number.
  // If it already exists, returns the existing file-number entry.
  Fileno GetOrCreateFileno(absl::string_view filename);

  // Returns the total number of nodes in the graph. Traverses the functions and
  // sums the node counts.
  int64_t GetNodeCount() const;

  // Returns the functions in this package.
  absl::Span<std::unique_ptr<Function>> functions() {
    return absl::MakeSpan(functions_);
  }
  absl::Span<const std::unique_ptr<Function>> functions() const {
    return functions_;
  }

  // Returns the procs in this package.
  absl::Span<std::unique_ptr<Proc>> procs() { return absl::MakeSpan(procs_); }
  absl::Span<const std::unique_ptr<Proc>> procs() const { return procs_; }

  // Returns the blocks in this package.
  absl::Span<std::unique_ptr<Block>> blocks() {
    return absl::MakeSpan(blocks_);
  }
  absl::Span<const std::unique_ptr<Block>> blocks() const { return blocks_; }

  // Returns the procs, functions, and blocks in this package (all types derived
  // from FunctionBase).
  // TODO(meheff): Consider holding functions and procs in a common vector.
  std::vector<FunctionBase*> GetFunctionBases() const;

  const std::string& name() const { return name_; }

  // Returns true if analysis indicates that this package always produces the
  // same value as 'other' when run with the same arguments. The analysis is
  // conservative and false may be returned for some "equivalent" packages.
  bool IsDefinitelyEqualTo(const Package* other) const;

  // Dumps the IR in a parsable text format.
  std::string DumpIr() const;

  std::vector<std::string> GetFunctionNames() const;

  // Returns whether this package contains a function with the "target" name.
  bool HasFunctionWithName(absl::string_view target) const;

  int64_t next_node_id() const { return next_node_id_; }

  // Intended for use by the parser when node ids are suggested by the IR text.
  void set_next_node_id(int64_t value) { next_node_id_ = value; }

  // Create a channel. Channels are used with send/receive nodes in communicate
  // between procs or between procs and external (to XLS) components. If no
  // channel ID is specified, a unique channel ID will be automatically
  // allocated.
  // TODO(meheff): Consider using a builder for constructing a channel.
  absl::StatusOr<StreamingChannel*> CreateStreamingChannel(
      absl::string_view name, ChannelOps supported_ops, Type* type,
      absl::Span<const Value> initial_values = {},
      FlowControl flow_control = FlowControl::kReadyValid,
      const ChannelMetadataProto& metadata = ChannelMetadataProto(),
      absl::optional<int64_t> id = absl::nullopt);

  absl::StatusOr<SingleValueChannel*> CreateSingleValueChannel(
      absl::string_view name, ChannelOps supported_ops, Type* type,
      const ChannelMetadataProto& metadata = ChannelMetadataProto(),
      absl::optional<int64_t> id = absl::nullopt);

  // Returns a span of the channels owned by the package. Sorted by channel ID.
  absl::Span<Channel* const> channels() const { return channel_vec_; }

  // Returns the channel with the given ID or returns an error if no such
  // channel exists.
  absl::StatusOr<Channel*> GetChannel(int64_t id) const;
  absl::StatusOr<Channel*> GetChannel(absl::string_view name) const;

  // Returns whether there exists a channel with the given ID.
  bool HasChannelWithId(int64_t id) const {
    return channels_.find(id) != channels_.end();
  }

  // Removes the given channel. If the channel has any associated send/receive
  // nodes an error is returned.
  absl::Status RemoveChannel(Channel* channel);

 private:
  // Adds the given channel to the package.
  absl::Status AddChannel(std::unique_ptr<Channel> channel);

  friend class FunctionBuilder;

  absl::optional<std::string> entry_;

  // Helper that returns a map from the names of functions inside this package
  // to the functions themselves.
  absl::flat_hash_map<std::string, Function*> GetFunctionByName();

  // Name of this package.
  std::string name_;

  // Ordinal to assign to the next node created in this package.
  int64_t next_node_id_ = 1;

  std::vector<std::unique_ptr<Function>> functions_;
  std::vector<std::unique_ptr<Proc>> procs_;
  std::vector<std::unique_ptr<Block>> blocks_;

  // Set of owned types in this package.
  absl::flat_hash_set<const Type*> owned_types_;

  // Set of owned function types in this package.
  absl::flat_hash_set<const FunctionType*> owned_function_types_;

  // Mapping from bit count to the owned "bits" type with that many bits. Use
  // node_hash_map for pointer stability.
  absl::node_hash_map<int64_t, BitsType> bit_count_to_type_;

  // Mapping from the size and element type of an array type to the owned
  // ArrayType. Use node_hash_map for pointer stability.
  using ArrayKey = std::pair<int64_t, const Type*>;
  absl::node_hash_map<ArrayKey, ArrayType> array_types_;

  // Mapping from elements to the owned tuple type.
  //
  // Uses node_hash_map for pointer stability.
  using TypeVec = absl::InlinedVector<const Type*, 4>;
  absl::node_hash_map<TypeVec, TupleType> tuple_types_;

  // Owned token type.
  TokenType token_type_;

  // Mapping from Type:ToString to the owned function type. Use
  // node_hash_map for pointer stability.
  absl::node_hash_map<std::string, FunctionType> function_types_;

  // Mapping of Fileno ids to string filenames, and vice-versa for reverse
  // lookups. These two data structures must be updated together for consistency
  // and should always contain the same number of entries.
  absl::flat_hash_map<Fileno, std::string> fileno_to_filename_;
  absl::flat_hash_map<std::string, Fileno> filename_to_fileno_;

  // Channels owned by this package. Indexed by channel id. Stored as
  // unique_ptrs for pointer stability.
  absl::flat_hash_map<int64_t, std::unique_ptr<Channel>> channels_;

  // Vector of channel pointers. Kept in sync with the channels_ map. Enables
  // easy, stable iteration over channels.
  std::vector<Channel*> channel_vec_;

  // The next channel ID to assign.
  int64_t next_channel_id_ = 0;
};

std::ostream& operator<<(std::ostream& os, const Package& package);

}  // namespace xls

#endif  // XLS_IR_PACKAGE_H_
