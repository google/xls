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

#ifndef XLS_PASSES_PASSES_H_
#define XLS_PASSES_PASSES_H_

#include <stdio.h>

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Defines the pass types for passes which operate strictly on XLS IR (i.e.,
// xls::Package).
using Pass = PassBase<Package>;
using CompoundPass = CompoundPassBase<Package>;
using FixedPointCompoundPass = FixedPointCompoundPassBase<Package>;
using InvariantChecker = CompoundPass::InvariantChecker;

static constexpr int64_t kMaxOptLevel = 3;

// Whether optimizations which split operations into multiple pieces should be
// performed at the given optimization level.
inline bool SplitsEnabled(int64_t opt_level) { return opt_level >= 3; }
inline bool NarrowingEnabled(int64_t opt_level) { return opt_level >= 2; }

// Abstract base class for passes operate at function/proc scope. The derived
// class must define RunOnFunctionBaseInternal.
class FunctionBasePass : public Pass {
 public:
  FunctionBasePass(std::string_view short_name, std::string_view long_name)
      : Pass(short_name, long_name) {}

  // Runs the pass on a single function/proc.
  absl::StatusOr<bool> RunOnFunctionBase(FunctionBase* f,
                                         const PassOptions& options,
                                         PassResults* results) const;

 protected:
  // Iterates over each function and proc in the package calling
  // RunOnFunctionBase.
  absl::StatusOr<bool> RunInternal(Package* p, const PassOptions& options,
                                   PassResults* results) const override;

  virtual absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const PassOptions& options,
      PassResults* results) const = 0;

  // Calls the given function for every node in the graph in a loop until no
  // further simplifications are possible.  simplify_f should return true if the
  // IR was modified. simplify_f can add or remove nodes including the node
  // passed to it.
  //
  // TransformNodesToFixedPoint returns true iff any invocations of simplify_f
  // returned true.
  absl::StatusOr<bool> TransformNodesToFixedPoint(
      FunctionBase* f,
      std::function<absl::StatusOr<bool>(Node*)> simplify_f) const;
};

// Abstract base class for passes operate on procs. The derived
// class must define RunOnProcInternal.
class ProcPass : public Pass {
 public:
  ProcPass(std::string_view short_name, std::string_view long_name)
      : Pass(short_name, long_name) {}

  // Proc the pass on a single proc.
  absl::StatusOr<bool> RunOnProc(Proc* proc, const PassOptions& options,
                                 PassResults* results) const;

 protected:
  // Iterates over each proc in the package calling RunOnProc.
  absl::StatusOr<bool> RunInternal(Package* p, const PassOptions& options,
                                   PassResults* results) const override;

  virtual absl::StatusOr<bool> RunOnProcInternal(
      Proc* proc, const PassOptions& options, PassResults* results) const = 0;
};

}  // namespace xls

#endif  // XLS_PASSES_PASSES_H_
