// Copyright 2022 The XLS Authors
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

#include "xls/jit/jit_proc_runtime.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "llvm/include/llvm/IR/Module.h"
#include "llvm/include/llvm/Support/Error.h"
#include "llvm/include/llvm/Target/TargetMachine.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/value.h"
#include "xls/jit/aot_compiler.h"
#include "xls/jit/aot_entrypoint.pb.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/llvm_compiler.h"
#include "xls/jit/proc_jit.h"

namespace xls {
namespace {

absl::Status InsertInitialChannelValues(const ProcElaboration& elaboration,
                                        ChannelQueueManager& queue_mgr) {
  // Inject initial values into channel queues.
  for (ChannelInstance* channel_instance : elaboration.channel_instances()) {
    Channel* channel = channel_instance->channel;
    ChannelQueue& queue = queue_mgr.GetQueue(channel_instance);
    for (const Value& value : channel->initial_values()) {
      XLS_RETURN_IF_ERROR(queue.Write(value));
    }
  }
  return absl::OkStatus();
}

// Wrapper compiler which just shares a single llvm::Module with multiple
// targets.
class SharedCompiler final : public LlvmCompiler {
 public:
  explicit SharedCompiler(std::string_view name, AotCompiler* underlying,
                          std::unique_ptr<llvm::TargetMachine> target,
                          llvm::DataLayout data_layout)
      : LlvmCompiler(std::move(target), std::move(data_layout),
                     underlying->opt_level(), underlying->include_msan()),
        underlying_(underlying),
        the_module_(underlying_->NewModule(
            absl::StrFormat("__shared_module_for_%s", name))) {}

  bool IsSharedCompilation() const override { return true; }

  // Share around the same module.
  std::unique_ptr<llvm::Module> NewModule(std::string_view ignored) override {
    CHECK(the_module_) << "no module to give out!";
    auto res = std::move(*the_module_);
    the_module_.reset();
    return res;
  }

  absl::Status CompileModule(std::unique_ptr<llvm::Module>&& module) override {
    XLS_RET_CHECK(!the_module_) << "Already took back module.";
    the_module_ = std::move(module);
    return absl::OkStatus();
  }

  // Return the underlying LLVM context.
  llvm::LLVMContext* GetContext() override { return underlying_->GetContext(); }
  absl::StatusOr<std::unique_ptr<llvm::TargetMachine>> CreateTargetMachine()
      override {
    return underlying_->CreateTargetMachine();
  }

  std::unique_ptr<llvm::Module> TakeModule() && {
    CHECK(the_module_) << "no module to give out!";
    auto res = std::move(*the_module_);
    the_module_.reset();
    return res;
  }

 protected:
  absl::Status InitInternal() override {
    return absl::InternalError("Should not be called");
  }

 private:
  AotCompiler* underlying_;
  std::optional<std::unique_ptr<llvm::Module>> the_module_;
};

absl::StatusOr<JitObjectCode> GetAotObjectCode(ProcElaboration elaboration,
                                               bool with_msan) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<AotCompiler> compiler,
                       AotCompiler::Create(with_msan));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<llvm::TargetMachine> target,
                       compiler->CreateTargetMachine());
  llvm::DataLayout layout = target->createDataLayout();
  SharedCompiler sc(elaboration.top()
                        ? elaboration.top()->GetName()
                        : elaboration.procs().front()->package()->name(),
                    compiler.get(), std::move(target), layout);
  std::vector<FunctionEntrypoint> entrypoints;
  entrypoints.reserve(elaboration.procs().size());
  for (Proc* p : elaboration.procs()) {
    entrypoints.push_back({.function = p});
    XLS_ASSIGN_OR_RETURN(entrypoints.back().jit_info,
                         JittedFunctionBase::Build(p, sc));
  }
  XLS_RETURN_IF_ERROR(compiler->CompileModule(std::move(sc).TakeModule()));
  XLS_ASSIGN_OR_RETURN(std::vector<uint8_t> object_code,
                       std::move(compiler)->GetObjectCode());
  return JitObjectCode{
      .object_code = std::move(object_code),
      .entrypoints = std::move(entrypoints),
      .data_layout = layout,
  };
}

namespace {
struct AotProcJitArgs {
  AotEntrypointProto entrypoint;
  Proc* proc;
  JitFunctionType unpacked;
  std::optional<JitFunctionType> packed;
};
}  // namespace

absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateAotRuntime(
    ProcElaboration elaboration, const AotPackageEntrypointsProto& entrypoints,
    absl::Span<ProcAotEntrypoints const> impls) {
  XLS_RET_CHECK_EQ(elaboration.procs().size(), entrypoints.entrypoint_size());
  XLS_RET_CHECK_EQ(elaboration.procs().size(), impls.size());
  absl::flat_hash_map<std::string, AotProcJitArgs> procs_by_name;
  for (const auto& entrypoint : entrypoints.entrypoint()) {
    XLS_RET_CHECK(!procs_by_name.contains(entrypoint.xls_function_identifier()))
        << "Multiple definitions for " << entrypoint.xls_function_identifier();
    procs_by_name[entrypoint.xls_function_identifier()] = {
        .entrypoint = entrypoint,
        .proc = nullptr,
        .unpacked = nullptr,
        .packed = std::nullopt};
  }
  for (const auto& impl : impls) {
    XLS_RET_CHECK(procs_by_name.contains(impl.proc->name()))
        << "Unknown implementation of " << impl.proc->name();
    AotProcJitArgs& args = procs_by_name[impl.proc->name()];
    XLS_RET_CHECK(args.proc == nullptr)
        << "Multiple copies of impl for " << impl.proc->name();
    args.proc = impl.proc;
    args.unpacked = impl.unpacked;
    args.packed = impl.packed;
  }
  XLS_RET_CHECK(absl::c_all_of(elaboration.procs(), [&](Proc* p) {
    return procs_by_name.contains(p->name()) &&
           procs_by_name[p->name()].proc == p;
  })) << "Elaboration has unknown procs";
  XLS_RET_CHECK(entrypoints.has_data_layout())
      << "Data layout required to create an aot runtime";
  llvm::Expected<llvm::DataLayout> layout =
      llvm::DataLayout::parse(entrypoints.data_layout());
  XLS_RET_CHECK(layout) << "Unable to parse '" << entrypoints.data_layout()
                        << "' to an llvm data-layout.";
  // Create a queue manager for the queues. This factory verifies that there an
  // receive only queue for every receive only channel.
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<JitChannelQueueManager> queue_manager,
      JitChannelQueueManager::CreateThreadSafe(
          std::move(elaboration), std::make_unique<JitRuntime>(*layout)));
  // Create a ProcJit for each Proc.
  std::vector<std::unique_ptr<ProcEvaluator>> proc_jits;
  for (const auto& [_, jit_args] : procs_by_name) {
    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<ProcJit> proc_jit,
        ProcJit::CreateFromAot(jit_args.proc, &queue_manager->runtime(),
                               queue_manager.get(), jit_args.entrypoint,
                               jit_args.unpacked, jit_args.packed));
    proc_jits.push_back(std::move(proc_jit));
  }

  // Create a runtime.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<SerialProcRuntime> proc_runtime,
                       SerialProcRuntime::Create(std::move(proc_jits),
                                                 std::move(queue_manager)));

  XLS_RETURN_IF_ERROR(InsertInitialChannelValues(
      proc_runtime->elaboration(), proc_runtime->queue_manager()));
  return std::move(proc_runtime);
}

absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateRuntime(
    ProcElaboration elaboration) {
  // We use the compiler to know the data layout.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<OrcJit> comp, OrcJit::Create());
  XLS_ASSIGN_OR_RETURN(llvm::DataLayout layout, comp->CreateDataLayout());
  // Create a queue manager for the queues. This factory verifies that there an
  // receive only queue for every receive only channel.
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<JitChannelQueueManager> queue_manager,
      JitChannelQueueManager::CreateThreadSafe(
          std::move(elaboration), std::make_unique<JitRuntime>(layout)));

  // Create a ProcJit for each Proc.
  std::vector<std::unique_ptr<ProcEvaluator>> proc_jits;
  for (Proc* proc : queue_manager->elaboration().procs()) {
    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<ProcJit> proc_jit,
        ProcJit::Create(proc, &queue_manager->runtime(), queue_manager.get()));
    proc_jits.push_back(std::move(proc_jit));
  }

  // Create a runtime.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<SerialProcRuntime> proc_runtime,
                       SerialProcRuntime::Create(std::move(proc_jits),
                                                 std::move(queue_manager)));

  XLS_RETURN_IF_ERROR(InsertInitialChannelValues(
      proc_runtime->elaboration(), proc_runtime->queue_manager()));
  return std::move(proc_runtime);
}

}  // namespace

absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateJitSerialProcRuntime(
    Package* package) {
  XLS_ASSIGN_OR_RETURN(ProcElaboration elaboration,
                       ProcElaboration::ElaborateOldStylePackage(package));
  return CreateRuntime(std::move(elaboration));
}

absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateJitSerialProcRuntime(
    Proc* top) {
  XLS_ASSIGN_OR_RETURN(ProcElaboration elaboration,
                       ProcElaboration::Elaborate(top));
  return CreateRuntime(std::move(elaboration));
}

absl::StatusOr<JitObjectCode> CreateProcAotObjectCode(Package* package,
                                                      bool with_msan) {
  XLS_ASSIGN_OR_RETURN(ProcElaboration elaboration,
                       ProcElaboration::ElaborateOldStylePackage(package));
  return GetAotObjectCode(std::move(elaboration), with_msan);
}
absl::StatusOr<JitObjectCode> CreateProcAotObjectCode(Proc* top,
                                                      bool with_msan) {
  XLS_ASSIGN_OR_RETURN(ProcElaboration elaboration,
                       ProcElaboration::Elaborate(top));
  return GetAotObjectCode(std::move(elaboration), with_msan);
}

// Create a SerialProcRuntime composed of ProcJits. Constructed from the
// elaboration of the given proc using the given impls. All procs in the
// elaboration must have an associated entry in the entrypoints and impls lists.
absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateAotSerialProcRuntime(
    Proc* top, const AotPackageEntrypointsProto& entrypoints,
    absl::Span<ProcAotEntrypoints const> impls) {
  XLS_ASSIGN_OR_RETURN(ProcElaboration elaboration,
                       ProcElaboration::Elaborate(top));
  return CreateAotRuntime(std::move(elaboration), entrypoints, impls);
}

// Create a SerialProcRuntime composed of ProcJits. Constructed from the
// elaboration of the given package using the given impls. All procs in the
// elaboration must have an associated entry in the entrypoints and impls lists.
absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateAotSerialProcRuntime(
    Package* package, const AotPackageEntrypointsProto& entrypoints,
    absl::Span<ProcAotEntrypoints const> impls) {
  XLS_ASSIGN_OR_RETURN(ProcElaboration elaboration,
                       ProcElaboration::ElaborateOldStylePackage(package));
  return CreateAotRuntime(std::move(elaboration), entrypoints, impls);
}

}  // namespace xls
