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

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "benchmark/benchmark.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/topo_sort.h"
#include "xls/solvers/solver.h"

namespace xls::solvers {
namespace {

struct BenchmarkInput {
  std::string name;
  std::string path;
  bool translate_only = false;
  bool skip_z3_solve = false;
  bool skip_z3_ne_zero = false;
};

const std::vector<BenchmarkInput>& GetInputs() {
  static const std::vector<BenchmarkInput>* inputs =
      new std::vector<BenchmarkInput>{
          {.name = "fp32_fmac", .path = "xls/examples/fp32_fmac.opt.ir"},
          {.name = "bitonic_sort", .path = "xls/examples/bitonic_sort.opt.ir"},
          {.name = "riscv_simple", .path = "xls/examples/riscv_simple.opt.ir"},
          {.name = "sha256_scaled",
           .path = "xls/examples/sha256_scaled.opt.ir",
           .skip_z3_ne_zero = true},
          {.name = "sha256_full",
           .path = "xls/examples/sha256.opt.ir",
           .translate_only = true}};
  return *inputs;
}

FunctionBase* GetBenchmarkSubject(Package* package) {
  std::optional<FunctionBase*> top = package->GetTop();
  if (top.has_value() && (*top)->node_count() > 10) {
    return *top;
  }
  FunctionBase* best = nullptr;
  int64_t max_nodes = -1;
  for (const auto& f : package->functions()) {
    if (f->node_count() > max_nodes) {
      max_nodes = f->node_count();
      best = f.get();
    }
  }
  for (const auto& p : package->procs()) {
    if (p->node_count() > max_nodes) {
      max_nodes = p->node_count();
      best = p.get();
    }
  }
  return best;
}

absl::StatusOr<Node*> FindSubjectNode(FunctionBase* f, SolverKind solver_kind) {
  if (f->IsFunction()) {
    Node* rv = f->AsFunctionOrDie()->return_value();
    if (rv->GetType()->IsBits()) {
      return rv;
    }
  }
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> nodes, TopoSort(f));
  for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
    if ((*it)->GetType()->IsBits()) {
      return *it;
    }
  }
  return absl::NotFoundError("No suitable subject node found");
}

void RunTranslateBenchmark(::testing::benchmark::State& state,
                           std::string_view ir_relative_path,
                           SolverKind solver_kind) {
  absl::StatusOr<std::filesystem::path> runfile_path =
      GetXlsRunfilePath(ir_relative_path);
  CHECK(runfile_path.ok()) << runfile_path.status().message();

  absl::StatusOr<std::string> contents = GetFileContents(*runfile_path);
  CHECK(contents.ok()) << contents.status().message();

  absl::StatusOr<std::unique_ptr<Package>> package =
      Parser::ParsePackage(*contents, ir_relative_path);
  CHECK(package.ok()) << package.status().message();

  FunctionBase* subject_entity = GetBenchmarkSubject(package->get());
  CHECK(subject_entity != nullptr) << "No benchmark subject found in package";

  absl::StatusOr<std::unique_ptr<Solver>> solver = CreateSolver(solver_kind);
  CHECK(solver.ok()) << solver.status().message();

  for (auto _ : state) {
    auto instance = (*solver)->CreateSolverInstance(subject_entity,
                                                    /*allow_unsupported=*/true);
    CHECK(instance.ok()) << instance.status().message();
    ::testing::DoNotOptimize(instance);
  }
}

enum class SolverTestKind {
  kEqZero,
  kNeZero,
};

void RunSolveBenchmark(::testing::benchmark::State& state,
                       std::string_view ir_relative_path,
                       SolverKind solver_kind, SolverTestKind test_kind) {
  absl::StatusOr<std::filesystem::path> runfile_path =
      GetXlsRunfilePath(ir_relative_path);
  CHECK(runfile_path.ok()) << runfile_path.status().message();

  absl::StatusOr<std::string> contents = GetFileContents(*runfile_path);
  CHECK(contents.ok()) << contents.status().message();

  absl::StatusOr<std::unique_ptr<Package>> package =
      Parser::ParsePackage(*contents, ir_relative_path);
  CHECK(package.ok()) << package.status().message();

  FunctionBase* subject_entity = GetBenchmarkSubject(package->get());
  CHECK(subject_entity != nullptr) << "No benchmark subject found in package";

  absl::StatusOr<Node*> subject = FindSubjectNode(subject_entity, solver_kind);
  if (!subject.ok()) {
    std::cerr << "Skipping solve benchmark for " << ir_relative_path
              << " because no suitable subject node was found: "
              << subject.status().message() << "\n";
    return;
  }

  absl::StatusOr<std::unique_ptr<Solver>> solver = CreateSolver(solver_kind);
  CHECK(solver.ok()) << solver.status().message();

  absl::StatusOr<std::unique_ptr<SolverInstance>> instance =
      (*solver)->CreateSolverInstance(subject_entity,
                                      /*allow_unsupported=*/true);
  CHECK(instance.ok()) << instance.status().message();

  Predicate predicate = (test_kind == SolverTestKind::kEqZero)
                            ? Predicate::EqualToZero()
                            : Predicate::NotEqualToZero();

  for (auto _ : state) {
    auto result_or = (*instance)->TryProve(*subject, predicate);
    CHECK(result_or.ok()) << result_or.status().message();
    ::testing::DoNotOptimize(result_or);
  }
}

void BM_Translate(::testing::benchmark::State& state) {
  SolverKind kind = static_cast<SolverKind>(state.range(0));
  const BenchmarkInput& input = GetInputs()[state.range(1)];
  RunTranslateBenchmark(state, input.path, kind);
}

void BM_Solve(::testing::benchmark::State& state) {
  SolverKind kind = static_cast<SolverKind>(state.range(0));
  const BenchmarkInput& input = GetInputs()[state.range(1)];
  SolverTestKind test_kind = static_cast<SolverTestKind>(state.range(2));
  RunSolveBenchmark(state, input.path, kind, test_kind);
}

void RegisterBenchmarks(::testing::Benchmark* b, bool translate) {
  const auto& inputs = GetInputs();
  for (SolverKind kind : {SolverKind::kZ3, SolverKind::kBitwuzla}) {
    std::string kind_str = (kind == SolverKind::kZ3) ? "Z3" : "Bitwuzla";
    for (int i = 0; i < inputs.size(); ++i) {
      if ((!translate && inputs[i].translate_only) ||
          (kind == SolverKind::kZ3 && inputs[i].skip_z3_solve)) {
        continue;
      }
      if (translate) {
        b->Args({static_cast<int>(kind), i},
                absl::StrCat("solver=", kind_str, "/input=", inputs[i].name));
      } else {
        for (SolverTestKind test_kind :
             {SolverTestKind::kEqZero, SolverTestKind::kNeZero}) {
          if (kind == SolverKind::kZ3 && test_kind == SolverTestKind::kNeZero &&
              inputs[i].skip_z3_ne_zero) {
            continue;
          }
          std::string test_kind_str =
              (test_kind == SolverTestKind::kEqZero) ? "eq_zero" : "ne_zero";
          b->Args({static_cast<int>(kind), i, static_cast<int>(test_kind)},
                  absl::StrCat("solver=", kind_str, "/input=", inputs[i].name,
                               "/test=", test_kind_str));
        }
      }
    }
  }
}

BENCHMARK(BM_Translate)->Apply([](::testing::Benchmark* b) {
  RegisterBenchmarks(b, /*translate=*/true);
});
BENCHMARK(BM_Solve)->Apply([](::testing::Benchmark* b) {
  RegisterBenchmarks(b, /*translate=*/false);
});

}  // namespace
}  // namespace xls::solvers

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  RunSpecifiedBenchmarks();
  return 0;
}
