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

#include "xls/dslx/run_routines/ir_test_runner.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/bytecode/bytecode_interpreter_options.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/run_routines/run_routines.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/warning_kind.h"
#include "xls/interpreter/evaluator_options.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/interpreter/interpreter_proc_runtime.h"
#include "xls/interpreter/proc_runtime.h"
#include "xls/ir/channel.h"
#include "xls/ir/events.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/value.h"
#include "xls/jit/function_jit.h"
#include "xls/jit/jit_proc_runtime.h"
#include "xls/passes/dfe_pass.h"
#include "xls/passes/pass_base.h"

namespace xls::dslx {
namespace {
class IrRunner : public AbstractParsedTestRunner {
 public:
  IrRunner(
      absl::flat_hash_map<std::string, std::unique_ptr<Package>>&& packages,
      absl::flat_hash_map<std::string, std::string>&& finish_chan_names,
      std::function<absl::StatusOr<std::unique_ptr<ProcRuntime>>(xls::Package*)>
          proc_runner,
      std::function<absl::StatusOr<InterpreterResult<Value>>(
          xls::Function* f, absl::Span<Value const>)>
          func_runner,
      ImportData* import_data)
      : packages_(std::move(packages)),
        finish_chan_names_(std::move(finish_chan_names)),
        proc_runner_(std::move(proc_runner)),
        func_runner_(std::move(func_runner)),
        import_data_(import_data) {}

  // TODO need to move to having each test proc have its own package from
  // ir_convert.
  absl::StatusOr<RunResult> RunTestSingleProc(
      xls::Proc* proc, Channel* terminate,
      const BytecodeInterpreterOptions& options) {
    XLS_RET_CHECK(options.post_fn_eval_hook() == nullptr)
        << "hooks not supported using non-dslx interpreters";
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ProcRuntime> rt,
                         proc_runner_(proc->package()));
    rt->ResetState();
    VLOG(2) << "Running proc " << proc->name() << " waiting on " << terminate;
    absl::StatusOr<int64_t> result =
        rt->TickUntilOutput({{terminate, 1}}, options.max_ticks());
    // NB Since the setup is done any failure now is in the tested code. EG
    // deadlock, assert, tick timeout etc.
    if (!result.ok()) {
      return RunResult{.result = result.status()};
    }
    VLOG(2) << "Test finished in " << result.value() << " ticks";
    std::vector<std::string> asserts;
    for (xls::ProcInstance* instance : rt->elaboration().proc_instances()) {
      const InterpreterEvents& events = rt->GetInterpreterEvents(instance);
      for (const auto& trace : events.trace_msgs) {
        options.trace_hook()(
            Span::FromString(trace.message, file_table()).value_or(Span{}),
            trace.message);
      }
      absl::c_copy(events.assert_msgs, std::back_inserter(asserts));
    }
    if (asserts.empty()) {
      std::optional<Value> proc_result =
          rt->queue_manager().GetQueue(terminate).Read();
      if (!proc_result || !proc_result->bits().IsOne()) {
        return RunResult{.result = FailureErrorStatus(
                             Span::Fake(),
                             "Proc terminated without a true result!",
                             file_table())};
      }
      return RunResult{.result = absl::OkStatus()};
    }
    return RunResult{.result =
                         absl::AbortedError(absl::StrJoin(asserts, "\n"))};
  }

  absl::StatusOr<RunResult> RunTestProc(
      std::string_view name,
      const BytecodeInterpreterOptions& options) override {
    if (options.trace_channels()) {
      LOG(WARNING) << "Unable to trace channels with IR testing";
    }
    if (options.format_preference() != FormatPreference::kDefault) {
      LOG(WARNING) << "Unable to set custom format preferences with IR testing";
    }
    XLS_RET_CHECK(options.post_fn_eval_hook() == nullptr)
        << "hooks not supported using non-dslx interpreters";
    XLS_RET_CHECK(finish_chan_names_.contains(name)) << name << " not found.";
    std::string_view finish_name = finish_chan_names_.at(name);
    auto package = packages_.at(name).get();
    // TODO(https://github.com/google/xls/issues/1592) To avoid any issues with
    // empty-procs or unrelated making deadlock detection not work due to
    // entering livelock we run DFE on the package.
    DeadFunctionEliminationPass dfe;
    PassResults pr;
    XLS_RETURN_IF_ERROR(dfe.Run(package, {}, &pr).status());
    // Get the corresponding entries.
    XLS_ASSIGN_OR_RETURN(auto* top, package->GetTopAsProc());
    XLS_ASSIGN_OR_RETURN(
        auto* chan, package->GetChannel(finish_name),
        _ << "available: ["
          << absl::StrJoin(
                 package->channels(), ", ",
                 [](std::string* ret, xls::Channel* c) { *ret = c->name(); })
          << "]");
    return RunTestSingleProc(top, chan, options);
  }
  absl::StatusOr<RunResult> RunTestFunction(
      std::string_view name,
      const BytecodeInterpreterOptions& options) override {
    auto* func_package = packages_.at(name).get();
    XLS_ASSIGN_OR_RETURN(xls::Function * f, func_package->GetTopAsFunction());
    XLS_RET_CHECK(f->GetType()->return_type()->IsTuple()) << f->GetType();
    XLS_RET_CHECK_EQ(f->GetType()->return_type()->AsTupleOrDie()->size(), 0)
        << f->GetType();
    XLS_ASSIGN_OR_RETURN(InterpreterResult<Value> v, this->func_runner_(f, {}));
    for (const TraceMessage& trace : v.events.trace_msgs) {
      options.trace_hook()(
          Span::FromString(trace.message, file_table()).value_or(Span{}),
          trace.message);
    }
    if (v.events.assert_msgs.empty()) {
      return RunResult{.result = absl::OkStatus()};
    }
    return RunResult{.result = absl::AbortedError(
                         absl::StrJoin(v.events.assert_msgs, "\n"))};
  }

 private:
  FileTable& file_table() { return import_data_->file_table(); }

  absl::flat_hash_map<std::string, std::unique_ptr<Package>> packages_;
  absl::flat_hash_map<std::string, std::string> finish_chan_names_;
  std::function<absl::StatusOr<std::unique_ptr<ProcRuntime>>(xls::Package*)>
      proc_runner_;
  std::function<absl::StatusOr<InterpreterResult<Value>>(
      xls::Function* f, absl::Span<Value const>)>
      func_runner_;
  ImportData* import_data_;
};

absl::StatusOr<std::unique_ptr<AbstractParsedTestRunner>> MakeRunner(
    ImportData* import_data, TypeInfo* type_info, Module* module,
    std::function<absl::StatusOr<InterpreterResult<Value>>(
        xls::Function* f, absl::Span<Value const>)>
        func,
    std::function<absl::StatusOr<std::unique_ptr<ProcRuntime>>(xls::Package*)>
        proc) {
  ConvertOptions base_option{
      .emit_fail_as_assert = true,
      .verify_ir = false,
      .warnings_as_errors = false,
      .enabled_warnings = kNoWarningsSet,
      .convert_tests = true,
  };
  absl::flat_hash_map<std::string, std::unique_ptr<Package>> packages;
  absl::flat_hash_map<std::string, std::string> finish_chan_names;
  for (std::string_view name : module->GetTestNames()) {
    std::optional<ModuleMember*> maybe_member =
        module->FindMemberWithName(name);
    XLS_RET_CHECK(maybe_member);
    auto member = *maybe_member;
    PackageConversionData package_data{
        .package = std::make_unique<Package>(
            absl::StrFormat("%s_test_for_package_%s", name, module->name()))};
    XLS_RETURN_IF_ERROR(ConvertOneFunctionIntoPackage(
        module, name, import_data, nullptr, base_option, &package_data));
    if (std::holds_alternative<TestProc*>(*member)) {
      TestProc* tp = std::get<TestProc*>(*member);
      std::string dslx_chan_name =
          tp->proc()->config().params()[0]->identifier();
      // TODO(allight): This duplicates code in the ir_convert/channel_scope.cc
      finish_chan_names[name] =
          absl::StrCat(package_data.package->name(), "__", dslx_chan_name);
    }
    packages[name] = std::move(package_data.package);
  }
  return std::make_unique<IrRunner>(
      std::move(packages), std::move(finish_chan_names), std::move(proc),
      std::move(func), import_data);
}
}  // namespace

absl::StatusOr<std::unique_ptr<AbstractParsedTestRunner>>
IrJitTestRunner::CreateTestRunner(ImportData* import_data, TypeInfo* type_info,
                                  Module* module) const {
  return MakeRunner(
      import_data, type_info, module,
      [](xls::Function* f,
         auto args) -> absl::StatusOr<InterpreterResult<Value>> {
        XLS_ASSIGN_OR_RETURN(auto jit, FunctionJit::Create(f));
        return jit->Run(args);
      },
      [](xls::Package* p) -> absl::StatusOr<std::unique_ptr<ProcRuntime>> {
        return CreateJitSerialProcRuntime(p, EvaluatorOptions());
      });
}

absl::StatusOr<std::unique_ptr<AbstractParsedTestRunner>>
IrInterpreterTestRunner::CreateTestRunner(ImportData* import_data,
                                          TypeInfo* type_info,
                                          Module* module) const {
  return MakeRunner(
      import_data, type_info, module, InterpretFunction,
      [](xls::Package* p) -> absl::StatusOr<std::unique_ptr<ProcRuntime>> {
        return CreateInterpreterSerialProcRuntime(p, EvaluatorOptions());
      });
}

}  // namespace xls::dslx
