// Copyright 2023 The XLS Authors
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

#include "xls/ir/elaboration.h"

#include <optional>
#include <string_view>

#include "absl/base/optimization.h"
#include "xls/common/casts.h"
#include "xls/ir/function.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_instantiation.h"

namespace xls {
template <>
/* static */ std::string_view ProcInstantiationPath::InstantiatedName(
    const ProcInstantiation& inst) {
  return inst.proc()->name();
}

template <>
/* static */ std::optional<Proc*> ProcInstantiationPath::Instantiated(
    const ProcInstantiation& inst) {
  return inst.proc();
}

template <>
/* static */ std::string_view BlockInstantiationPath::InstantiatedName(
    const Instantiation& inst) {
  switch (inst.kind()) {
    case InstantiationKind::kBlock:
      return down_cast<const BlockInstantiation&>(inst)
          .instantiated_block()
          ->name();
    case InstantiationKind::kFifo:
      return "fifo";
    case InstantiationKind::kExtern:
      return down_cast<const ExternInstantiation&>(inst).function()->name();
    case InstantiationKind::kDelayLine:
      return "delay_line";
  }
  ABSL_UNREACHABLE();
}

template <>
/* static */ std::optional<Block*> BlockInstantiationPath::Instantiated(
    const Instantiation& inst) {
  switch (inst.kind()) {
    case InstantiationKind::kBlock:
      return down_cast<const BlockInstantiation&>(inst).instantiated_block();
    default:
      return std::nullopt;
  }
}
}  // namespace xls
