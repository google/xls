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

#include "xls/visualization/ir_viz/node_attribute_visitor.h"

#include "absl/status/status.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/format_strings.h"
#include "xls/ir/function.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"

namespace xls {

absl::Status AttributeVisitor::DefaultHandler(Node* node) {
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleArraySlice(ArraySlice* array_slice) {
  attributes_.set_width(array_slice->width());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleAssert(Assert* assert) {
  attributes_.set_message_(assert->message());
  if (assert->label().has_value()) {
    attributes_.set_label(*assert->label());
  }
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleBitSlice(BitSlice* bit_slice) {
  attributes_.set_start(bit_slice->start());
  attributes_.set_width(bit_slice->width());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleCountedFor(CountedFor* counted_for) {
  attributes_.set_trip_count(counted_for->trip_count());
  attributes_.set_stride(counted_for->stride());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleCover(Cover* cover) {
  attributes_.set_label(cover->label());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleDecode(Decode* decode) {
  attributes_.set_width(decode->width());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleDynamicBitSlice(
    DynamicBitSlice* bit_slice) {
  attributes_.set_width(bit_slice->width());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleDynamicCountedFor(
    DynamicCountedFor* counted_for) {
  attributes_.set_body(counted_for->body()->name());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleInputPort(InputPort* input_port) {
  attributes_.set_name(input_port->name());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleInstantiationInput(
    InstantiationInput* instantiation_input) {
  attributes_.set_instantiation(instantiation_input->instantiation()->name());
  attributes_.set_port_name(instantiation_input->port_name());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleInstantiationOutput(
    InstantiationOutput* instantiation_output) {
  attributes_.set_instantiation(instantiation_output->instantiation()->name());
  attributes_.set_port_name(instantiation_output->port_name());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleInvoke(Invoke* invoke) {
  attributes_.set_to_apply(invoke->to_apply()->name());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleLiteral(Literal* literal) {
  attributes_.set_value(literal->value().ToHumanString(FormatPreference::kHex));
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleMap(Map* map) {
  attributes_.set_to_apply(map->to_apply()->name());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleMinDelay(MinDelay* min_delay) {
  attributes_.set_delay(min_delay->delay());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleOneHot(OneHot* one_hot) {
  attributes_.set_lsb_prio(one_hot->priority() == LsbOrMsb::kLsb);
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleOutputPort(OutputPort* output_port) {
  attributes_.set_name(output_port->name());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleReceive(Receive* receive) {
  attributes_.set_channel(receive->channel_name());
  attributes_.set_blocking(receive->is_blocking());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleRegisterRead(RegisterRead* register_read) {
  attributes_.set_register_(register_read->GetRegister()->name());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleRegisterWrite(
    RegisterWrite* register_write) {
  attributes_.set_register_(register_write->GetRegister()->name());
  attributes_.set_has_load_enable(register_write->load_enable().has_value());
  attributes_.set_has_reset(register_write->reset().has_value());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleSel(Select* sel) {
  attributes_.set_has_default(sel->default_value().has_value());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleSend(Send* send) {
  attributes_.set_channel(send->channel_name());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleSignExtend(ExtendOp* sign_ext) {
  attributes_.set_new_bit_count(sign_ext->new_bit_count());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleTrace(Trace* trace) {
  attributes_.set_format(StepsToVerilogFormatString(trace->format()));
  attributes_.set_verbosity(trace->verbosity());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleTupleIndex(TupleIndex* tuple_index) {
  attributes_.set_index(tuple_index->index());
  return absl::OkStatus();
}

absl::Status AttributeVisitor::HandleZeroExtend(ExtendOp* zero_ext) {
  attributes_.set_new_bit_count(zero_ext->new_bit_count());
  return absl::OkStatus();
}

}  // namespace xls
