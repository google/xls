// Copyright 2026 The XLS Authors
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

#include "xls/spin/trace_compare.h"

#include <cstdint>
#include <queue>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "google/protobuf/text_format.h"
#include "re2/re2.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/ir/bits.h"
#include "xls/ir/evaluator_result.pb.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_instantiation.h"

namespace xls::spin {

namespace {

// Extracts the DSLX proc name from a mangled IR proc name.
// Format: __{module}__{proc_name}_0{optional_param_suffix}_next
std::string ExtractProcName(std::string_view ir_name,
                            std::string_view module) {
  std::string prefix =
      absl::StrCat("__", absl::StrReplaceAll(module, {{".", "_"}}), "__");
  std::string_view s = absl::StripPrefix(ir_name, prefix);
  s = absl::StripSuffix(s, "_next");
  std::string name(s);
  RE2::Replace(&name, "_0(__.*)?$", "");
  return name;
}

constexpr std::string_view kArrow = "->";
constexpr std::string_view kHash = "#";
constexpr std::string_view kSep = "::";
constexpr std::string_view kSend = "SEND";
constexpr std::string_view kRecv = "RECV";

// [^}]* absorbs any extra JSON fields without breaking.
const RE2 kSpinLineRe(
    "\\{\"channel_name\":\"([^\"]*)\","
    "\"direction\":\"([^\"]*)\","
    "\"value\":(-?[0-9]+)"
    "(?:,\"proctype\":\"([^\"]*)\",\"pid\":([0-9]+))?[^}]*\\}");

// Rewrites the bare-name portion via (leaf_proc_type, instance_index, bare) lookup.
// Returns channel_name unchanged when no "::" separator exists or key not found.
std::string ResolveDslxChannelName(std::string_view channel_name,
                               const DslxChannelNameMap& map) {
  // Split "ProcPath::var_name" at the last "::".
  size_t pos = channel_name.rfind(kSep);
  if (pos == std::string_view::npos || map.empty()) {
    return std::string(channel_name);
  }
  std::string_view path = channel_name.substr(0, pos);
  std::string_view bare = channel_name.substr(pos + kSep.size());

  // Extract leaf proc type and instance: "A->Counter#2" -> ("Counter", 2).
  size_t arrow = path.rfind(kArrow);
  std::string_view leaf =
      (arrow == std::string_view::npos) ? path : path.substr(arrow + kArrow.size());
  size_t hash = leaf.find(kHash);
  std::string_view proc_name = leaf.substr(0, hash);  // npos gives full leaf
  int64_t instance = 0;
  if (hash != std::string_view::npos) {
    (void)absl::SimpleAtoi(leaf.substr(hash + kHash.size()), &instance);
  }

  // Replace var_name with the ChannelDecl string from the map.
  auto it = map.find(
      std::make_tuple(std::string(proc_name), instance, std::string(bare)));
  if (it == map.end()) {
    return std::string(channel_name);
  }
  return absl::StrCat(channel_name.substr(0, pos + kSep.size()), it->second);
}

// BFS work item: a proc instance with its resolved channel bindings.
struct ProcWork {
  const dslx::Proc* proc;
  int64_t instance_idx;
  absl::flat_hash_map<std::string, std::string> bindings;
};

// Walks `node` (a proc's config body) in source order. Adds ChannelDecl
// let-bindings to `local` and enqueues spawned children into `work_queue`.
void ProcessConfigBody(
    const dslx::AstNode* node, const dslx::Module& module,
    absl::flat_hash_map<std::string, std::string>& local,
    absl::flat_hash_map<std::string, int64_t>& instance_counts,
    std::queue<ProcWork>& work_queue) {
  if (node == nullptr) {
    return;
  }
  if (node->kind() == dslx::AstNodeKind::kLet) {
    const auto* let = static_cast<const dslx::Let*>(node);
    if (let->rhs() != nullptr &&
        let->rhs()->kind() == dslx::AstNodeKind::kChannelDecl) {
      const auto* decl = static_cast<const dslx::ChannelDecl*>(let->rhs());
      if (decl->channel_name_expr().kind() == dslx::AstNodeKind::kString) {
        std::string_view ch_name =
            static_cast<const dslx::String&>(decl->channel_name_expr()).text();
        for (dslx::NameDef* nd : let->name_def_tree()->GetNameDefs()) {
          local[nd->identifier()] = std::string(ch_name);
        }
      }
    }
  }
  if (node->kind() == dslx::AstNodeKind::kSpawn) {
    const auto* spawn = static_cast<const dslx::Spawn*>(node);
    // spawn->callee() gives the NameRef to the proc name; config()->callee()
    // would give "ProcName.config" which does not match module lookups.
    if (spawn->callee()->kind() == dslx::AstNodeKind::kNameRef) {
      const std::string& callee_name =
          static_cast<const dslx::NameRef*>(spawn->callee())->identifier();
      std::optional<dslx::Proc*> callee =
          module.GetMember<dslx::Proc>(callee_name);
      if (callee.has_value()) {
        int64_t child_idx = instance_counts[callee_name]++;
        absl::flat_hash_map<std::string, std::string> child_bindings;
        absl::Span<dslx::Param* const> params = (*callee)->config().params();
        absl::Span<dslx::Expr* const> args = spawn->config()->args();
        for (size_t i = 0; i < std::min(args.size(), params.size()); ++i) {
          if (args[i]->kind() != dslx::AstNodeKind::kNameRef) {
            continue;
          }
          const std::string& arg_name =
              static_cast<const dslx::NameRef*>(args[i])->identifier();
          auto it = local.find(arg_name);
          if (it != local.end()) {
            child_bindings[params[i]->identifier()] = it->second;
          }
        }
        work_queue.push({.proc = *callee,
                         .instance_idx = child_idx,
                         .bindings = std::move(child_bindings)});
      }
    }
  }
  for (const dslx::AstNode* child : node->GetChildren(/*want_types=*/false)) {
    ProcessConfigBody(child, module, local, instance_counts, work_queue);
  }
}

}  // namespace

// Builds a DslxChannelNameMap via BFS; instance indices match DSLX spawn ordering
// so bindings are correct when a proc type is spawned multiple times.
DslxChannelNameMap BuildDslxChannelNameMap(const dslx::Module& module) {
  DslxChannelNameMap result;
  absl::flat_hash_map<std::string, int64_t> instance_counts;
  std::queue<ProcWork> q;

  for (const dslx::TestProc* tp : module.GetTestProcs()) {
    const dslx::Proc* proc = tp->proc();
    int64_t idx = instance_counts[proc->identifier()]++;
    q.push({.proc = proc, .instance_idx = idx, .bindings = {}});
  }

  while (!q.empty()) {
    ProcWork work = std::move(q.front());
    q.pop();
    const std::string proc_name = work.proc->identifier();
    absl::flat_hash_map<std::string, std::string> local =
        std::move(work.bindings);
    ProcessConfigBody(&work.proc->config(), module, local, instance_counts, q);
    for (const auto& [var, ch] : local) {
      result[std::make_tuple(proc_name, work.instance_idx, var)] = ch;
    }
  }
  return result;
}

absl::StatusOr<ProcInstPaths> BuildProcInstPathsForSpin(Package* package) {
  XLS_ASSIGN_OR_RETURN(Proc * top, package->GetTopAsProc());
  std::string module(package->name());
  ProcInstPaths paths;

  std::queue<std::pair<Proc*, std::string>> worklist;
  worklist.push({top, ExtractProcName(top->name(), module)});

  while (!worklist.empty()) {
    auto [proc, proc_path] = std::move(worklist.front());
    worklist.pop();
    paths[proc->name()].push_back(proc_path);

    absl::flat_hash_map<Proc*, int64_t> child_counts;
    for (const auto& inst : proc->proc_instantiations()) {
      Proc* child = inst->proc();
      int64_t n = child_counts[child]++;
      worklist.push(
          {child, absl::StrCat(proc_path, kArrow,
                               ExtractProcName(child->name(), module), kHash, n)});
    }
  }
  return paths;
}

bool SpinTraceHasTerminator(std::string_view json,
                            std::string_view terminator_channel) {
  if (terminator_channel.empty()) {
    return false;
  }
  for (std::string_view line : absl::StrSplit(json, '\n')) {
    std::string channel_name, direction_str, proctype, pid_str;
    int64_t value = 0;
    if (!RE2::FullMatch(line, kSpinLineRe, &channel_name, &direction_str,
                        &value, &proctype, &pid_str)) {
      continue;
    }
    if (direction_str == kSend &&
        absl::StripPrefix(channel_name, "_") == terminator_channel) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<TraceMap> ParseSpinTrace(std::string_view json,
                                        const ProcInstPaths& proc_paths,
                                        std::string_view terminator_channel) {
  // Per-proctype: pids in first-seen order; index = instance number.
  absl::flat_hash_map<std::string, std::vector<int64_t>> proctype_pids;
  TraceMap events;

  for (std::string_view line : absl::StrSplit(json, '\n')) {
    if (line.empty()) {
      continue;
    }
    std::string channel_name, direction_str, proctype, pid_str;
    int64_t value = 0;
    if (!RE2::FullMatch(line, kSpinLineRe, &channel_name, &direction_str,
                        &value, &proctype, &pid_str)) {
      LOG(WARNING) << "SPIN trace: unrecognized line: " << line;
      continue;
    }
    Direction dir;
    if (direction_str == kSend) {
      dir = Direction::kSend;
    } else if (direction_str == kRecv) {
      dir = Direction::kRecv;
    } else {
      LOG(WARNING) << "SPIN trace: unknown direction: " << direction_str;
      continue;
    }
    // SPIN models channels as 32-bit signed; reinterpret as uint32 to match
    // the DSLX interpreter which uses unsigned values.
    if (value < 0) {
      value &= 0xFFFFFFFFULL;
    }
    // Promela channel names are prefixed with "_"; strip it to get the bare name.
    std::string bare(absl::StripPrefix(channel_name, "_"));
    if (!terminator_channel.empty() && bare == terminator_channel &&
        dir == Direction::kSend) {
      break;
    }

    // Map pid to instance index via first-seen order in proctype_pids; build key.
    // Falls back to bare channel name when proctype is absent or unknown.
    std::string key = bare;
    if (!proctype.empty()) {
      int64_t pid;
      if (absl::SimpleAtoi(pid_str, &pid)) {
        auto it = proc_paths.find(proctype);
        if (it != proc_paths.end()) {
          std::vector<int64_t>& pids = proctype_pids[proctype];
          auto pids_it = absl::c_find(pids, pid);
          if (pids_it == pids.end()) {
            pids.push_back(pid);
            pids_it = std::prev(pids.end());
          }
          size_t instance = pids_it - pids.begin();
          if (instance < it->second.size()) {
            key = absl::StrCat(it->second[instance], kSep, bare);
          }
        }
      }
    }
    events[{key, dir}].push_back(value);
  }
  return events;
}

absl::StatusOr<TraceMap> ParseDslxTrace(
    std::string_view textproto, std::string_view terminator_channel,
    const DslxChannelNameMap& channel_name_map) {
  EvaluatorResultsProto proto;
  if (!google::protobuf::TextFormat::ParseFromString(std::string(textproto),
                                                     &proto)) {
    return absl::InvalidArgumentError("DSLX trace text proto parse error");
  }
  TraceMap events;
  for (const EvaluatorResultProto& eval_result : proto.results()) {
    for (const TraceMessageProto& msg : eval_result.events().trace_msgs()) {
      if (!msg.has_channel()) {
        continue;
      }
      const TraceChannelProto& ch = msg.channel();
      Direction dir;
      switch (ch.direction()) {
        case TraceChannelProto::SEND:
          dir = Direction::kSend;
          break;
        case TraceChannelProto::RECV:
          dir = Direction::kRecv;
          break;
        default:
          continue;
      }
      int64_t value = 0;
      if (ch.has_value() && ch.value().has_bits()) {
        value = Bits::FromBytes(ch.value().bits().data())
                    .UnsignedToInt64()
                    .value_or(0);
      }
      std::string_view ch_name = ch.channel_name();
      size_t sep = ch_name.rfind(kSep);
      std::string_view bare =
          (sep == std::string_view::npos) ? ch_name : ch_name.substr(sep + kSep.size());
      if (!terminator_channel.empty() && bare == terminator_channel &&
          dir == Direction::kSend) {
        return events;
      }
      std::string key = ResolveDslxChannelName(ch.channel_name(), channel_name_map);
      events[{key, dir}].push_back(value);
    }
  }
  return events;
}

absl::Status CompareTraces(const TraceMap& spin, const TraceMap& dslx) {
  absl::btree_set<std::pair<std::string, Direction>> all_keys;
  for (const auto& [key, _] : spin) {
    all_keys.insert(key);
  }
  for (const auto& [key, _] : dslx) {
    all_keys.insert(key);
  }

  std::string mismatches;
  for (const auto& key : all_keys) {
    const auto& [channel_name, dir] = key;
    std::string_view direction = (dir == Direction::kSend) ? kSend : kRecv;
    auto spin_it = spin.find(key);
    auto dslx_it = dslx.find(key);
    if (spin_it == spin.end()) {
      absl::StrAppendFormat(&mismatches,
                            "  channel=%s dir=%s: present in dslx only\n",
                            channel_name, direction);
    } else if (dslx_it == dslx.end()) {
      absl::StrAppendFormat(&mismatches,
                            "  channel=%s dir=%s: present in spin only\n",
                            channel_name, direction);
    } else if (spin_it->second != dslx_it->second) {
      absl::StrAppendFormat(
          &mismatches, "  channel=%s dir=%s: spin=%zu dslx=%zu events differ\n",
          channel_name, direction, spin_it->second.size(),
          dslx_it->second.size());
    }
  }
  if (!mismatches.empty()) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Trace mismatch:\n%s", mismatches));
  }
  return absl::OkStatus();
}

}  // namespace xls::spin
