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

#ifndef XLS_VISUALIZATION_IR_VIZ_NODE_ATTRIBUTE_VISITOR_H_
#define XLS_VISUALIZATION_IR_VIZ_NODE_ATTRIBUTE_VISITOR_H_

#include "absl/status/status.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/visualization/ir_viz/visualization.pb.h"

namespace xls {

// Visitor which constructs the attributes (if any) of a node and returns them
// as a JSON object.
// Note that attributes that become operands (e.g. indices=[...] in an
// array_index) are not added to the proto as these are represented as operands.
// In some cases where the operand is ambiguous, a field in the attribute proto
// may disambiguate the operands (e.g. has_default for a select).
class AttributeVisitor : public DfsVisitorWithDefault {
 public:
  absl::Status DefaultHandler(Node* node) override;

  absl::Status HandleArraySlice(ArraySlice* array_slice) override;
  absl::Status HandleAssert(Assert* assert) override;
  absl::Status HandleBitSlice(BitSlice* bit_slice) override;
  absl::Status HandleCountedFor(CountedFor* counted_for) override;
  absl::Status HandleCover(Cover* cover) override;
  absl::Status HandleDecode(Decode* decode) override;
  absl::Status HandleDynamicBitSlice(DynamicBitSlice* bit_slice) override;
  absl::Status HandleDynamicCountedFor(DynamicCountedFor* counted_for) override;
  absl::Status HandleInputPort(InputPort* input_port) override;
  absl::Status HandleInstantiationInput(
      InstantiationInput* instantiation_input) override;
  absl::Status HandleInstantiationOutput(
      InstantiationOutput* instantiation_output) override;
  absl::Status HandleInvoke(Invoke* invoke) override;
  absl::Status HandleLiteral(Literal* literal) override;
  absl::Status HandleMap(Map* map) override;
  absl::Status HandleMinDelay(MinDelay* min_delay) override;
  absl::Status HandleOneHot(OneHot* one_hot) override;
  absl::Status HandleOutputPort(OutputPort* output_port) override;
  absl::Status HandleReceive(Receive* receive) override;
  absl::Status HandleRegisterRead(RegisterRead* register_read) override;
  absl::Status HandleRegisterWrite(RegisterWrite* register_write) override;
  absl::Status HandleSel(Select* sel) override;
  absl::Status HandleSend(Send* send) override;
  absl::Status HandleSignExtend(ExtendOp* sign_ext) override;
  absl::Status HandleTrace(Trace* trace) override;
  absl::Status HandleTupleIndex(TupleIndex* tuple_index) override;
  absl::Status HandleZeroExtend(ExtendOp* zero_ext) override;

  const viz::NodeAttributes& attributes() const { return attributes_; }

 private:
  viz::NodeAttributes attributes_;
};

}  // namespace xls

#endif  // XLS_VISUALIZATION_IR_VIZ_NODE_ATTRIBUTE_VISITOR_H_
