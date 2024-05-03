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

#include "xls/tools/extract_interface.h"

#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/register.h"
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls {

PackageInterfaceProto ExtractPackageInterface(
    Package* package) {
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
  auto add_common = [&](PackageInterfaceProto::FunctionBase* f,
                        FunctionBase* ir) {
    f->set_name(ir->name());
    f->set_top(package->GetTop() == ir);
  };
  auto add_named = [&](PackageInterfaceProto::NamedValue* n, Node* node) {
    *n->mutable_type() = node->GetType()->ToProto();
    n->set_name(node->GetName());
  };
  for (const auto& f : package->functions()) {
    auto* func = proto.add_functions();
    add_common(func->mutable_base(), f.get());
    for (auto* param : f->params()) {
      add_named(func->add_parameters(), param);
    }
    *func->mutable_result_type() = f->GetType()->return_type()->ToProto();
  }
  for (const auto& p : package->procs()) {
    auto* proc = proto.add_procs();
    add_common(proc->mutable_base(), p.get());
    for (auto* param : p->params()) {
      add_named(proc->add_state(), param);
    }
    for (const auto& c : p->channel_references()) {
      if (c->direction() == Direction::kSend) {
        *proc->add_send_channels() = c->name();
      } else {
        *proc->add_recv_channels() = c->name();
      }
    }
  }
  for (const auto& b : package->blocks()) {
    auto* blk = proto.add_blocks();
    add_common(blk->mutable_base(), b.get());
    for (const Register* reg : b->GetRegisters()) {
      auto* named = blk->add_registers();
      named->set_name(reg->name());
      *named->mutable_type() = reg->type()->ToProto();
    }
    for (Node* port : b->GetInputPorts()) {
      add_named(blk->add_input_ports(), port);
    }
    for (Node* port : b->GetOutputPorts()) {
      add_named(blk->add_output_ports(), port);
    }
  }
  return proto;
}

}  // namespace xls
