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

#include "xls/dev_tools/extract_interface.h"

#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/register.h"
#include "xls/ir/type.h"
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls {
namespace {

void AddTyped(PackageInterfaceProto::NamedValue* n, Node* node, Type* ty) {
  *n->mutable_type() = ty->ToProto();
  n->set_name(node->GetName());
}

void AddNamed(PackageInterfaceProto::NamedValue* n, Node* node) {
  AddTyped(n, node, node->GetType());
}

void AddFunctionBase(PackageInterfaceProto::FunctionBase* f, FunctionBase* ir,
                     bool top) {
  f->set_name(ir->name());
  f->set_top(top);
}

}  // namespace

PackageInterfaceProto::Function ExtractFunctionInterface(Function* func) {
  PackageInterfaceProto::Function proto;
  AddFunctionBase(proto.mutable_base(), func,
                  /*top=*/func->package()->GetTop() == func);
  for (auto* param : func->params()) {
    AddNamed(proto.add_parameters(), param);
  }
  *proto.mutable_result_type() = func->GetType()->return_type()->ToProto();
  return proto;
}

PackageInterfaceProto::Proc ExtractProcInterface(Proc* proc) {
  PackageInterfaceProto::Proc proto;
  AddFunctionBase(proto.mutable_base(), proc,
                  /*top=*/proc->package()->GetTop() == proc);
  for (auto* param : proc->params()) {
    AddNamed(proto.add_state(), param);
  }
  for (const auto& c : proc->channel_references()) {
    if (c->direction() == Direction::kSend) {
      *proto.add_send_channels() = c->name();
    } else {
      *proto.add_recv_channels() = c->name();
    }
  }
  return proto;
}

PackageInterfaceProto::Block ExtractBlockInterface(Block* block) {
  PackageInterfaceProto::Block proto;
  AddFunctionBase(proto.mutable_base(), block,
                  /*top=*/block->package()->GetTop() == block);
  for (const Register* reg : block->GetRegisters()) {
    auto* named = proto.add_registers();
    named->set_name(reg->name());
    *named->mutable_type() = reg->type()->ToProto();
  }
  for (Node* port : block->GetInputPorts()) {
    AddNamed(proto.add_input_ports(), port);
  }
  for (OutputPort* port : block->GetOutputPorts()) {
    AddTyped(proto.add_output_ports(), port, port->output_type());
  }
  return proto;
}

PackageInterfaceProto ExtractPackageInterface(Package* package) {
  PackageInterfaceProto proto;
  // Basic information
  proto.set_name(package->name());
  for (const auto& [_, f] : package->fileno_to_name()) {
    *proto.add_files() = f;
  }
  // Fill in channels
  for (const Channel* c : package->channels()) {
    auto* chan = proto.add_channels();
    chan->set_name(c->name());
    *chan->mutable_type() = c->type()->ToProto();
    PackageInterfaceProto::Channel::Direction dir;
    if (c->CanSend()) {
      if (c->CanReceive()) {
        dir = PackageInterfaceProto::Channel::INOUT;
      } else {
        dir = PackageInterfaceProto::Channel::OUT;
      }
    } else if (c->CanReceive()) {
      dir = PackageInterfaceProto::Channel::IN;
    } else {
      dir = PackageInterfaceProto::Channel::INVALID;
    }
    chan->set_direction(dir);
  }
  for (const auto& f : package->functions()) {
    *proto.add_functions() = ExtractFunctionInterface(f.get());
  }
  for (const auto& p : package->procs()) {
    *proto.add_procs() = ExtractProcInterface(p.get());
  }
  for (const auto& b : package->blocks()) {
    *proto.add_blocks() = ExtractBlockInterface(b.get());
  }
  return proto;
}

}  // namespace xls
