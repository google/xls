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

#include "xls/tools/node_coverage_utils.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/jit/jit_runtime.h"
#include "xls/tools/node_coverage_stats.pb.h"

namespace xls {

namespace {
absl::StatusOr<LeafTypeTree<InlineBitmap>> ToBitmapTree(const Value& v,
                                                        Type* ty) {
  XLS_ASSIGN_OR_RETURN(LeafTypeTree<Value> ltt, ValueToLeafTypeTree(v, ty));
  return leaf_type_tree::Map<InlineBitmap, Value>(
      ltt.AsView(), [](const Value& v) -> InlineBitmap {
        if (v.IsBits()) {
          return v.bits().bitmap();
        }
        return InlineBitmap(0);
      });
}

absl::StatusOr<Value> ToValue(LeafTypeTreeView<InlineBitmap> ltt) {
  XLS_ASSIGN_OR_RETURN(
      LeafTypeTree<Value> vltt,
      (leaf_type_tree::MapIndex<Value, InlineBitmap>(
          ltt,
          [](Type* ty, const InlineBitmap& bm,
             absl::Span<const int64_t>) -> absl::StatusOr<Value> {
            if (ty->IsBits()) {
              return Value(Bits::FromBitmap(bm));
            }
            if (ty->IsToken()) {
              return Value::Token();
            }
            if (ty->IsTuple()) {
              XLS_RET_CHECK_EQ(ty->AsTupleOrDie()->size(), 0);
              return Value::Tuple({});
            }
            XLS_RET_CHECK(ty->IsArray());
            XLS_RET_CHECK_EQ(ty->AsArrayOrDie()->size(), 0);
            return Value::Array({});
          })));
  return LeafTypeTreeToValue(vltt.AsView());
}
}  // namespace

void CoverageEvalObserver::NodeEvaluated(Node* n, const Value& v) {
  if (paused_) {
    VLOG(2) << "Ignoring " << n << "=" << v << " due to pause.";
    return;
  }
  VLOG(2) << "Saw value " << n << " is " << v;
  absl::StatusOr<LeafTypeTree<InlineBitmap>> bitmap =
      ToBitmapTree(v, n->GetType());
  if (!bitmap.ok()) {
    // Just ignore.
    LOG(ERROR) << "Unable to record " << n << " due to " << bitmap.status();
    return;
  }
  if (coverage_.contains(n)) {
    leaf_type_tree::SimpleUpdateFrom<InlineBitmap, InlineBitmap>(
        coverage_[n].AsMutableView(), bitmap->AsView(),
        [](InlineBitmap& l, const InlineBitmap& r) { l.Union(r); });
    return;
  }

  coverage_[n] = bitmap.value();
}

absl::Status CoverageEvalObserver::Finalize() {
  if (!jit_) {
    XLS_RET_CHECK(raw_coverage_.empty()) << "no jit but raw data was present.";
    return absl::OkStatus();
  }
  for (const auto& [node, data] : raw_coverage_) {
    NodeEvaluated(node,
                  jit_.value()->UnpackBuffer(data.data(), node->GetType()));
  }
  raw_coverage_.clear();
  return absl::OkStatus();
}

absl::StatusOr<NodeCoverageStatsProto> CoverageEvalObserver::proto() const {
  XLS_RET_CHECK(raw_coverage_.empty()) << "Need to call finalize first";
  NodeCoverageStatsProto res;
  if (coverage_.empty()) {
    LOG(WARNING) << "No coverage information collected.";
    return res;
  }
  if (!coverage_.begin()->first->package()->fileno_to_name().empty()) {
    std::vector<std::string> names;
    names.resize(absl::c_max_element(
                     coverage_.begin()->first->package()->fileno_to_name(),
                     [](const auto& l, const auto& r) {
                       return l.first.value() < r.first.value();
                     })
                     ->first.value() +
                 1);
    for (const auto& [fileno, name] :
         coverage_.begin()->first->package()->fileno_to_name()) {
      names[fileno.value()] = name;
    }
    res.mutable_files()->Assign(names.begin(), names.end());
  }

  for (const auto& [node, bitmaps] : coverage_) {
    NodeCoverageStatsProto::NodeStats* node_stats = res.add_nodes();
    node_stats->set_node_id(node->id());
    *node_stats->mutable_node_text() = node->ToString();
    for (const SourceLocation& sl : node->loc().locations) {
      NodeCoverageStatsProto::NodeStats::Loc* loc = node_stats->add_loc();
      loc->set_fileno(sl.fileno().value());
      loc->set_lineno(sl.lineno().value());
      loc->set_colno(sl.colno().value());
    }
    XLS_ASSIGN_OR_RETURN(Value v, ToValue(bitmaps.AsView()));
    XLS_ASSIGN_OR_RETURN(*node_stats->mutable_set_bits(), v.AsProto());
    node_stats->set_total_bit_count(v.GetFlatBitCount());
    node_stats->set_unset_bit_count(absl::c_accumulate(
        bitmaps.elements(), 0,
        [](int64_t v, const InlineBitmap& bm) -> int64_t {
          return v + (bm.bit_count() - Bits::FromBitmap(bm).PopCount());
        }));
  }

  return res;
}
void CoverageEvalObserver::RecordNodeValue(int64_t node_ptr,
                                           const uint8_t* data) {
  CHECK(jit_);
  if (paused_) {
    VLOG(2) << "Ignoring " << node_ptr << " due to pause.";
    return;
  }
  Node* node = reinterpret_cast<Node*>(static_cast<intptr_t>(node_ptr));
  if (!raw_coverage_.contains(node)) {
    raw_coverage_[node] =
        std::vector<uint8_t>(jit_.value()->GetTypeByteSize(node->GetType()), 0);
  }
  if (node->GetType()->GetFlatBitCount() == 0) {
    return;
  }
  std::vector<uint8_t>& bits = raw_coverage_[node];
  for (int64_t i = 0; i < bits.size(); ++i) {
    bits[i] = bits[i] | data[i];
  }
}

ScopedRecordNodeCoverage::~ScopedRecordNodeCoverage() {
  if (!txtproto_ && !binproto_) {
    return;
  }
  CHECK_OK(obs_.Finalize());
  absl::StatusOr<NodeCoverageStatsProto> proto = obs_.proto();
  if (!proto.ok()) {
    LOG(ERROR) << "Unable to turn coverage stats to proto: " << proto.status();
    return;
  }
  if (txtproto_) {
    std::string out;
    if (google::protobuf::TextFormat::PrintToString(*proto, &out)) {
      absl::Status write = SetFileContents(*txtproto_, out);
      if (!write.ok()) {
        LOG(ERROR) << "Unable to write textproto: " << write;
      }
    }
  }
  if (binproto_) {
    absl::Status write =
        SetFileContents(*binproto_, proto->SerializeAsString());
    if (!write.ok()) {
      LOG(ERROR) << "Unable to write textproto: " << write;
    }
  }
}
}  // namespace xls
