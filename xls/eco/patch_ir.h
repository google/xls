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
#ifndef XLS_ECO_PATCH_IR_H_
#define XLS_ECO_PATCH_IR_H_

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/eco/ir_patch.pb.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/xls_type.pb.h"
#include "xls/ir/xls_value.pb.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {

class PatchIr {
 public:
  explicit PatchIr(FunctionBase* function_base, xls_eco::IrPatchProto& patch);
  absl::Status ApplyPatch();
  absl::Status PrintPatch();
  absl::Status ExportIr(const std::string& export_path) const;
  absl::Status ExportScheduleProto();
  absl::Status PatchSchedule(const PipelineSchedule& schedule);
  absl::StatusOr<PipelineSchedule> GetPatchedSchedule();
  enum class EditPathPriority : uint8_t {
    kEdgeDelete,
    kNodeDelete,
    kNodeUpdate,
    kNodeInsert,
    kEdgeUpdate,
    kEdgeInsert
  };

 private:
  absl::Status ApplyPath(const xls_eco::EditPathProto& edit_path);
  absl::Status ApplyDeletePath(const xls_eco::NodeEditPathProto& node_delete);
  absl::Status ApplyDeletePath(const xls_eco::EdgeEditPathProto& edge_delete);
  absl::Status ApplyInsertPath(const xls_eco::NodeEditPathProto& node_insert);
  absl::Status ApplyInsertPath(const xls_eco::EdgeEditPathProto& edge_insert);
  absl::Status ApplyUpdatePath(const xls_eco::NodeEditPathProto& node_update);
  absl::Status ApplyUpdatePath(const xls_eco::EdgeEditPathProto& edge_update);

  bool CompareEditPaths(const xls_eco::EditPathProto& lhs,
                        const xls_eco::EditPathProto& rhs);
  absl::StatusOr<std::vector<Node*>> MakeDummyNodes(absl::Span<Type*> types);
  absl::StatusOr<Node*> MakeDummyNode(Type* type);
  absl::StatusOr<int64_t> GetProtoBitCount(const TypeProto& type);
  absl::StatusOr<Node*> ResolveNodeByPatchName(std::string_view patch_name);
  absl::Status UpdateNodeMaps(Node* n, absl::Span<Node*> dummy_operands,
                              std::string_view node_name);
  absl::Status CleanupDummyNodes(Node* node);
  xls_eco::IrPatchProto patch_;
  std::vector<xls_eco::EditPathProto> sorted_edit_paths_;
  absl::flat_hash_set<std::string> inserted_node_names_;
  FunctionBase* function_base_;
  Package* package_;
  std::optional<PipelineSchedule> schedule_;
  absl::flat_hash_map<Node*, std::vector<Node*>> dummy_nodes_map_;
  absl::flat_hash_map<std::pair<Node*, uint>, uint>
      commutative_edge_index_map_;
  // Tracks available slots (dummy operands) on commutative nodes.
  absl::flat_hash_map<Node*, int64_t> commutative_free_slots_;
  Node* dummy_return_node_ = nullptr;
  absl::flat_hash_map<std::string, std::string> patch_to_ir_node_map_;
  absl::Status IsolateReturnNode();
  absl::Status RestoreReturnNode();
  absl::Status ValidatePatch();
  absl::Status PatchContainsNode(std::string_view node_name);
};

}  // namespace xls
#endif  // PLATFORMS_HLS_XLS_ECO_PATCH_IR_H_
