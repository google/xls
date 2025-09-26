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
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/fileno.h"
#include "xls/ir/source_location.h"
#include "xls/ir/transform_metrics.pb.h"
#include "xls/ir/type.h"
#include "xls/ir/type_manager.h"
#include "xls/ir/value.h"
#include "xls/ir/xls_type.pb.h"

namespace xls {

class Block;
class Channel;
class Function;
class FunctionBase;
class Proc;
class SingleValueChannel;
class StreamingChannel;

// Data structure collecting aggregate transformation metrics for the IR.  These
// values are zero-ed at construction time and only increment.
struct TransformMetrics {
  // Number of nodes added (number of invocations of
  // FunctionBase::AddNodeInternal).
  int64_t nodes_added = 0;

  // Number of nodes removed (number of invocations of
  // FunctionBase::RemoveNode).
  int64_t nodes_removed = 0;

  // Number of nodes replaced (number of calls to Node::ReplaceUsesWith).
  int64_t nodes_replaced = 0;

  // Number of operands replaced (number of calls to
  // Node::ReplaceOperand[Number]).
  int64_t operands_replaced = 0;

  // Number of operands removed (number of calls to
  // Node::RemoveOptionalOperand).
  int64_t operands_removed = 0;

  static TransformMetrics FromProto(const TransformMetricsProto& proto);
  TransformMetrics operator+(const TransformMetrics& other) const;
  TransformMetrics operator-(const TransformMetrics& other) const;
  std::string ToString() const;
  TransformMetricsProto ToProto() const;
};

class Package {
 public:
  explicit Package(std::string_view name);

  // Note: functions have parent pointers to their packages, so we don't want
  // them to be moved or copied; this makes Package non-moveable non-copyable.
  // If a copy is required one can explicitly use ClonePackage in
  // clone_package.h.
  Package(const Package& other) = delete;
  Package& operator=(const Package& other) = delete;

  virtual ~Package();

  std::optional<FunctionBase*> GetTop() const;
  bool HasTop() const { return top_.has_value(); }
  // Sets the top entity of the package.
  absl::Status SetTop(std::optional<FunctionBase*> top);
  // Sets the top to a FunctionBase with its name equivalent to the 'top_name'
  // parameter. The function calls xls::Package::SetTop function. Prerequisite:
  // a single function base with the with its name equivalent to the 'top_name'
  // parameter must exist.
  absl::Status SetTopByName(std::string_view top_name);

  // Returns true if the package's top is set to `f`.
  bool IsTop(const FunctionBase* f) const {
    return top_.has_value() && f == *top_;
  }

  // Helper function to get the top as a function, proc or block.
  absl::StatusOr<Function*> GetTopAsFunction() const;
  absl::StatusOr<Proc*> GetTopAsProc() const;
  absl::StatusOr<Block*> GetTopAsBlock() const;

  // Returns a FunctionBase with the given name if a single instance exists.
  absl::StatusOr<FunctionBase*> GetFunctionBaseByName(
      std::string_view name) const;

  TypeManager& type_manager() { return type_manager_; }
  const TypeManager& type_manager() const { return type_manager_; }
  // Returns whether the given type is one of the types owned by this package.
  bool IsOwnedType(const Type* type) const {
    return type_manager_.IsOwnedType(type);
  }
  bool IsOwnedFunctionType(const FunctionType* function_type) const {
    return type_manager_.IsOwnedFunctionType(function_type);
  }

  BitsType* GetBitsType(int64_t bit_count) {
    return type_manager_.GetBitsType(bit_count);
  }
  ArrayType* GetArrayType(int64_t size, Type* element_type) {
    return type_manager_.GetArrayType(size, element_type);
  }
  TupleType* GetTupleType(absl::Span<Type* const> element_types) {
    return type_manager_.GetTupleType(element_types);
  }
  TokenType* GetTokenType() { return type_manager_.GetTokenType(); }
  FunctionType* GetFunctionType(absl::Span<Type* const> args_types,
                                Type* return_type) {
    return type_manager_.GetFunctionType(args_types, return_type);
  }

  // Returns a pointer to a type owned by this package that is of the same
  // type as 'other_package_type', which may be owned by another package.
  absl::StatusOr<Type*> MapTypeFromOtherPackage(Type* other_package_type) {
    return type_manager_.MapTypeFromOtherArena(other_package_type);
  }

  // Creates and returned an owned type constructed from the given proto.
  absl::StatusOr<Type*> GetTypeFromProto(const TypeProto& proto) {
    return type_manager_.GetTypeFromProto(proto);
  }
  absl::StatusOr<FunctionType*> GetFunctionTypeFromProto(
      const FunctionTypeProto& proto) {
    return type_manager_.GetFunctionTypeFromProto(proto);
  }

  Type* GetTypeForValue(const Value& value) {
    return type_manager_.GetTypeForValue(value);
  }

  // Add a function, proc, or block to the package. Ownership is transferred to
  // the package.
  Function* AddFunction(std::unique_ptr<Function> f);
  Proc* AddProc(std::unique_ptr<Proc> proc);
  Block* AddBlock(std::unique_ptr<Block> block);

  struct PackageMergeResult {
    // other package -> this package name mapping (channels, procs, functions,
    // and blocks)
    absl::flat_hash_map<std::string, std::string> name_updates;
    // other package -> this package channel id mapping
    absl::flat_hash_map<std::string, std::string> channel_updates;
  };
  // Import all IR constructs from another package to this package. Ownership of
  // all constructs is transferred to this package.
  absl::StatusOr<PackageMergeResult> ImportFromPackage(const Package* other);

  // Get a function, proc, or block by name. Returns an error if no such
  // construct of the indicated kind exists with that name.
  absl::StatusOr<Function*> GetFunction(std::string_view func_name) const;
  absl::StatusOr<Proc*> GetProc(std::string_view proc_name) const;
  absl::StatusOr<Block*> GetBlock(std::string_view block_name) const;

  // Gets a function, proc, or block by name. Returns nullopt if no such
  // construct of the indicated kind exists with that name.
  std::optional<Function*> TryGetFunction(std::string_view func_name) const;
  std::optional<Proc*> TryGetProc(std::string_view proc_name) const;
  std::optional<Block*> TryGetBlock(std::string_view block_name) const;

  // Remove a function, proc, or block. The caller is responsible for ensuring
  // no references to the construct remain (e.g., via invoke operations). The
  // function, proc, or block must not be the top entity of the package. Use the
  // xls::Package::UnsetTop function to unset the top. The function, proc, or
  // block cannot be nullptr.
  absl::Status RemoveFunctionBase(FunctionBase* function_base);
  absl::Status RemoveFunction(Function* function);
  absl::Status RemoveProc(Proc* proc);
  absl::Status RemoveBlock(Block* block);

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
  SourceLocation AddSourceLocation(std::string_view filename, Lineno lineno,
                                   Colno colno);

  // Translates a SourceLocation object into a human readable debug identifier
  // of the form: "<source_file_path>:<line_number>".
  std::string SourceLocationToString(const SourceLocation& loc);

  // Retrieves the next node ID to assign to a node in the package and
  // increments the next node counter. For use in node construction.
  int64_t GetNextNodeIdAndIncrement() { return next_node_id_++; }

  // Adds a file to the file-number table and returns its corresponding number.
  // If it already exists, returns the existing file-number entry.
  Fileno GetOrCreateFileno(std::string_view filename);

  // Forcibly sets a given file number to map to a given file.
  // Used when parsing a `Package`.
  void SetFileno(Fileno file_number, std::string_view filename);

  // Get the filename corresponding to the given `Fileno`.
  std::optional<std::string> GetFilename(Fileno file_number) const;

  const absl::flat_hash_map<Fileno, std::string>& fileno_to_name() const {
    return fileno_to_filename_;
  }

  // Returns the total number of nodes in the graph. Traverses the functions,
  // procs and blocks and sums the node counts.
  int64_t GetNodeCount() const;
  // Returns the total number of nodes in the blocks in the graph. Traverses the
  // blocks and sums the node counts.
  int64_t GetBlockNodeCount() const;
  // Returns the total number of nodes in the functions in the graph. Traverses
  // the functions and sums the node counts.
  int64_t GetFunctionNodeCount() const;
  // Returns the total number of nodes in the procs in the graph. Traverses the
  // procs and sums the node counts.
  int64_t GetProcNodeCount() const;

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

  int64_t next_node_id() const { return next_node_id_; }

  // Intended for use by the parser when node ids are suggested by the IR text.
  void set_next_node_id(int64_t value) { next_node_id_ = value; }

  // Create a channel. Channels are used with send/receive nodes in communicate
  // between procs or between procs and external (to XLS) components. If no
  // channel ID is specified, a unique channel ID will be automatically
  // allocated.
  // TODO(meheff): Consider using a builder for constructing a channel.
  absl::StatusOr<StreamingChannel*> CreateStreamingChannel(
      std::string_view name, ChannelOps supported_ops, Type* type,
      absl::Span<const Value> initial_values = {},
      ChannelConfig channel_config = ChannelConfig(),
      FlowControl flow_control = FlowControl::kReadyValid,
      ChannelStrictness strictness = kDefaultChannelStrictness,
      std::optional<int64_t> id = std::nullopt);

  // Create a channel without special flop control. Channels are used with
  // send/receive nodes in communicate between procs or between procs and
  // external (to XLS) components. If no channel ID is specified, a unique
  // channel ID will be automatically allocated.
  // TODO(meheff): Consider using a builder for constructing a channel.
  absl::StatusOr<StreamingChannel*> CreateStreamingChannel(
      std::string_view name, ChannelOps supported_ops, Type* type,
      absl::Span<const Value> initial_values,
      std::optional<FifoConfig> fifo_config,
      FlowControl flow_control = FlowControl::kReadyValid,
      ChannelStrictness strictness = kDefaultChannelStrictness,
      std::optional<int64_t> id = std::nullopt) {
    return CreateStreamingChannel(
        /*name=*/name, /*supported_ops=*/supported_ops, /*type=*/type,
        /*initial_values=*/initial_values,
        /*channel_config=*/
        ChannelConfig(fifo_config, std::nullopt, std::nullopt),
        /*flow_control=*/flow_control, /*strictness=*/strictness,
        /*id=*/id);
  }

  absl::StatusOr<SingleValueChannel*> CreateSingleValueChannel(
      std::string_view name, ChannelOps supported_ops, Type* type,
      std::optional<int64_t> id = std::nullopt);

  // Variants which add a channel on a proc for new style procs.
  // TODO(https://github.com/google/xls/issues/869): Move these methods to
  // xls::Proc.
  absl::StatusOr<StreamingChannel*> CreateStreamingChannelInProc(
      std::string_view name, ChannelOps supported_ops, Type* type, Proc* proc,
      absl::Span<const Value> initial_values = {},
      ChannelConfig channel_config = ChannelConfig(),
      FlowControl flow_control = FlowControl::kReadyValid,
      ChannelStrictness strictness = kDefaultChannelStrictness,
      std::optional<int64_t> id = std::nullopt);
  absl::StatusOr<SingleValueChannel*> CreateSingleValueChannelInProc(
      std::string_view name, ChannelOps supported_ops, Type* type, Proc* proc,
      std::optional<int64_t> id = std::nullopt);

  // Returns a span of the channels owned by the package. Sorted by channel ID.
  absl::Span<Channel* const> channels() const { return channel_vec_; }

  // Returns the channel with the given ID or returns an error if no such
  // channel exists.
  absl::StatusOr<Channel*> GetChannel(int64_t id) const;
  absl::StatusOr<Channel*> GetChannel(std::string_view name) const;

  // Change the name of channel 'original' to 'new_name'.
  //
  // Only valid for old-style procs (ChannelsAreProcScoped() == false).
  absl::Status RenameChannel(std::string_view original,
                             std::string_view new_name);

  // Returns true if channels are proc scoped in this package. Returns false if
  // there are no channels or procs.
  bool ChannelsAreProcScoped() const;

  // Returns whether there exists a channel with the given ID / name.
  bool HasChannelWithId(int64_t id) const {
    for (Channel* channel : channel_vec_) {
      if (channel->id() == id) {
        return true;
      }
    }
    return false;
  }
  bool HasChannelWithName(std::string_view name) const {
    return channels_.find(name) != channels_.end();
  }

  // Removes the given channel. If the channel has any associated send/receive
  // nodes an error is returned.
  absl::Status RemoveChannel(Channel* channel);

  // Builder to collect overrides when cloning channels.
  // Each field is optional where std::nullopt indicates that the cloned channel
  // should share the same value as the original. If a field contains a value,
  // the cloned channel should ignore the original's value and use the override.
  class CloneChannelOverrides {
   public:
    explicit CloneChannelOverrides() = default;

    CloneChannelOverrides& OverrideSupportedOps(ChannelOps supported_ops) {
      supported_ops_ = supported_ops;
      return *this;
    }

    CloneChannelOverrides& OverrideInitialValues(
        absl::Span<const Value> values) {
      initial_values_ = values;
      return *this;
    }

    CloneChannelOverrides& OverrideChannelConfig(ChannelConfig channel_config) {
      channel_config_ = channel_config;
      return *this;
    }

    CloneChannelOverrides& OverrideFlowControl(FlowControl flow_control) {
      flow_control_ = flow_control;
      return *this;
    }

    CloneChannelOverrides& OverrideStrictness(ChannelStrictness strictness) {
      strictness_ = strictness;
      return *this;
    }

    std::optional<ChannelOps> supported_ops() const { return supported_ops_; }
    std::optional<absl::Span<const Value>> initial_values() const {
      return initial_values_;
    }
    std::optional<ChannelConfig> channel_config() const {
      return channel_config_;
    }
    std::optional<FlowControl> flow_control() const { return flow_control_; }
    std::optional<ChannelStrictness> strictness() const { return strictness_; }

   private:
    std::optional<ChannelOps> supported_ops_;
    std::optional<absl::Span<const Value>> initial_values_;
    std::optional<ChannelConfig> channel_config_;
    std::optional<FlowControl> flow_control_;
    std::optional<ChannelStrictness> strictness_;
  };

  // Clone channel, potentially from another package, with new name. Channel id
  // may differ from the old channel (it definitely will if it's coming from
  // this package).
  // CloneChannelOverrides is a builder class that optionally overrides fields
  // from the original channel.
  absl::StatusOr<Channel*> CloneChannel(
      Channel* channel, std::string_view name,
      const CloneChannelOverrides& overrides = CloneChannelOverrides());

  // Returns the transform metrics aggregated across all FunctionBases.
  const TransformMetrics& transform_metrics() const {
    return transform_metrics_;
  }
  TransformMetrics& transform_metrics() { return transform_metrics_; }

  // Allocate a new user data id. This function will not reuse an id until
  // ReleaseNodeUserDataId is called on it.
  int64_t AllocateNodeUserDataId();

  // Releases the user data id and allows it to be reused.
  //
  // NB This must be called once for each value returned by
  // AllocateNodeUserDataId.
  //
  // When this is called all nodes with user data *MUST* have *already* had
  // TakeUserData called on them to delete the user data associated with them.
  // On DEBUG builds this is CHECKed.
  void ReleaseNodeUserDataId(int64_t id);
  bool IsLiveUserDataId(int64_t id) { return user_data_ids_.Get(id); }

 private:
  std::vector<std::string> GetChannelNames() const;

  // Adds the given channel to the package.
  absl::Status AddChannel(std::unique_ptr<Channel> channel, Proc* proc);

  friend class FunctionBuilder;

  std::optional<FunctionBase*> top_;

  // Name of this package.
  std::string name_;

  // Ordinal to assign to the next node created in this package.
  int64_t next_node_id_ = 1;

  std::vector<std::unique_ptr<Function>> functions_;
  std::vector<std::unique_ptr<Proc>> procs_;
  std::vector<std::unique_ptr<Block>> blocks_;

  // Underlying manager for types used in this package.
  TypeManager type_manager_;

  // The largest `Fileno` used in this `Package`.
  std::optional<Fileno> maximum_fileno_;

  // Mapping of Fileno ids to string filenames, and vice-versa for reverse
  // lookups. These two data structures must be updated together for consistency
  // and should always contain the same number of entries.
  absl::flat_hash_map<Fileno, std::string> fileno_to_filename_;
  absl::flat_hash_map<std::string, Fileno> filename_to_fileno_;

  // Channels owned by this package. Indexed by channel name. Stored as
  // unique_ptrs for pointer stability.
  absl::flat_hash_map<std::string, std::unique_ptr<Channel>> channels_;

  // Vector of channel pointers. Kept in sync with the channels_ map. Enables
  // easy, stable iteration over channels.
  std::vector<Channel*> channel_vec_;

  // The next channel ID to assign.
  int64_t next_channel_id_ = 0;

  // Metrics which record the total number of transformations to the package.
  TransformMetrics transform_metrics_ = {0};

  // Bitmap containing allocated user data ids.
  InlineBitmap user_data_ids_;
};

// Printers for fuzztest use.
void FuzzTestPrintSourceCode(const std::unique_ptr<Package>& p,
                             std::ostream* os);
void FuzzTestPrintSourceCode(const std::shared_ptr<Package>& p,
                             std::ostream* os);

std::ostream& operator<<(std::ostream& os, const Package& package);

// Implements the common idiom of:
// a) if top_str is given and non-empty, sets that entity as the top entity of
//    the package, then
// b) retrieve the top entity for the package
absl::StatusOr<FunctionBase*> FindTop(Package* p,
                                      std::optional<std::string_view> top_str);

}  // namespace xls

#endif  // XLS_IR_PACKAGE_H_
