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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_instantiation.h"
#include "xls/ir/value.h"

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
