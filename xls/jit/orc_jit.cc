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

#include "xls/jit/orc_jit.h"

#include <cerrno>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>  // NOLINT
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/include/llvm-c/Target.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "llvm/include/llvm/ADT/StringExtras.h"
#include "llvm/include/llvm/Analysis/CGSCCPassManager.h"
#include "llvm/include/llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/include/llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/include/llvm/IR/Argument.h"
#include "llvm/include/llvm/IR/BasicBlock.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/IR/Instruction.h"
#include "llvm/include/llvm/IR/LegacyPassManager.h"
#include "llvm/include/llvm/IR/Module.h"
#include "llvm/include/llvm/IR/PassManager.h"
#include "llvm/include/llvm/IR/Use.h"
#include "llvm/include/llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/include/llvm/Passes/OptimizationLevel.h"
#include "llvm/include/llvm/Passes/PassBuilder.h"
#include "llvm/include/llvm/Support/Casting.h"
#include "llvm/include/llvm/Support/CodeGen.h"
#include "llvm/include/llvm/Support/Error.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "llvm/include/llvm/TargetParser/SubtargetFeature.h"
#include "llvm/include/llvm/TargetParser/X86TargetParser.h"
#include "llvm/include/llvm/Transforms/Instrumentation/MemorySanitizer.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/jit/observer.h"

namespace xls {
namespace {

absl::once_flag once;
void OnceInit() {
  LLVMInitializeNativeTarget();
  LLVMInitializeNativeAsmPrinter();
  LLVMInitializeNativeAsmParser();
}

std::string DumpLlvmModuleToString(const llvm::Module* module) {
  std::string buffer;
  llvm::raw_string_ostream ostream(buffer);
  module->print(ostream, nullptr);
  ostream.flush();
  return buffer;
}

// Used for reporting a bad optimization level specification to LLVM internals.
class BadOptLevelError : public llvm::ErrorInfo<BadOptLevelError> {
 public:
  explicit BadOptLevelError(int opt_level) : opt_level_(opt_level) {}

  void log(llvm::raw_ostream& os) const override {
    os << "Invalid opt level: " << opt_level_;
  }

  std::error_code convertToErrorCode() const override {
    return std::error_code(EINVAL, std::system_category());
  }

  static char ID;

 private:
  int opt_level_;
};

char BadOptLevelError::ID;

}  // namespace

OrcJit::OrcJit(int64_t opt_level, bool emit_object_code, bool include_msan)
    : context_(std::make_unique<llvm::LLVMContext>()),
      execution_session_(
          std::make_unique<llvm::orc::UnsupportedExecutorProcessControl>()),
      object_layer_(
          execution_session_,
          []() { return std::make_unique<llvm::SectionMemoryManager>(); }),
      dylib_(execution_session_.createBareJITDylib("main")),
      opt_level_(opt_level),
      emit_object_code_(emit_object_code),
      data_layout_(""),
      include_msan_(include_msan) {}

OrcJit::~OrcJit() {
  if (auto err = execution_session_.endSession()) {
    execution_session_.reportError(std::move(err));
  }
}

llvm::Expected<llvm::orc::ThreadSafeModule> OrcJit::Optimizer(
    llvm::orc::ThreadSafeModule module,
    const llvm::orc::MaterializationResponsibility& responsibility) {
  llvm::Module* bare_module = module.getModuleUnlocked();

  XLS_VLOG(2) << "Unoptimized module IR:";
  XLS_VLOG_LINES(2, DumpLlvmModuleToString(bare_module));
  if (jit_observer_ != nullptr &&
      jit_observer_->GetNotificationOptions().unoptimized_module) {
    jit_observer_->UnoptimizedModule(bare_module);
  }

  llvm::CGSCCAnalysisManager cgam;
  llvm::FunctionAnalysisManager fam;
  llvm::LoopAnalysisManager lam;
  llvm::ModuleAnalysisManager mam;
  llvm::PassBuilder pass_builder;

  if (include_msan_) {
    pass_builder.registerPipelineStartEPCallback(
        [](llvm::ModulePassManager& mpm, llvm::OptimizationLevel) -> void {
          mpm.addPass(
              llvm::MemorySanitizerPass(llvm::MemorySanitizerOptions()));
        });
  }
  pass_builder.registerModuleAnalyses(mam);
  pass_builder.registerCGSCCAnalyses(cgam);
  pass_builder.registerFunctionAnalyses(fam);
  pass_builder.registerLoopAnalyses(lam);
  pass_builder.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::OptimizationLevel llvm_opt_level;
  switch (opt_level_) {
    case 0:
      llvm_opt_level = llvm::OptimizationLevel::O0;
      break;
    case 1:
      llvm_opt_level = llvm::OptimizationLevel::O1;
      break;
    case 2:
      llvm_opt_level = llvm::OptimizationLevel::O2;
      break;
    case 3:
      llvm_opt_level = llvm::OptimizationLevel::O3;
      break;
    default:
      return llvm::Expected<llvm::orc::ThreadSafeModule>(
          llvm::Error(std::make_unique<BadOptLevelError>(opt_level_)));
  }
  llvm::ModulePassManager mpm;
  if (llvm_opt_level == llvm::OptimizationLevel::O0) {
    mpm = pass_builder.buildO0DefaultPipeline(llvm_opt_level);
  } else {
    mpm = pass_builder.buildPerModuleDefaultPipeline(llvm_opt_level);
  }
  mpm.run(*bare_module, mam);

  XLS_VLOG(2) << "Optimized module IR:";
  XLS_VLOG_LINES(2, DumpLlvmModuleToString(bare_module));
  if (jit_observer_ != nullptr &&
      jit_observer_->GetNotificationOptions().optimized_module) {
    jit_observer_->OptimizedModule(bare_module);
  }

  bool observe_asm_code =
      (jit_observer_ != nullptr &&
       jit_observer_->GetNotificationOptions().assembly_code_str);
  if (VLOG_IS_ON(3) || observe_asm_code) {
    // The ostream and its buffer must be declared before the
    // module_pass_manager because the destrutor of the pass manager calls flush
    // on the ostream so these must be destructed *after* the pass manager. C++
    // guarantees that the destructors are called in reverse order the objects
    // are declared.
    llvm::SmallVector<char, 0> stream_buffer;
    llvm::raw_svector_ostream ostream(stream_buffer);
    llvm::legacy::PassManager mpm;
    if (target_machine_->addPassesToEmitFile(
            mpm, ostream, nullptr, llvm::CodeGenFileType::AssemblyFile)) {
      XLS_VLOG(3) << "Could not create ASM generation pass!";
    }
    mpm.run(*bare_module);
    XLS_VLOG(3) << "Generated ASM:";
    std::string asm_code(stream_buffer.begin(), stream_buffer.end());
    XLS_VLOG_LINES(3, asm_code);
    if (observe_asm_code) {
      jit_observer_->AssemblyCodeString(bare_module, asm_code);
    }
  }

  return module;
}

absl::StatusOr<std::unique_ptr<OrcJit>> OrcJit::Create(
    int64_t opt_level, bool emit_object_code, std::optional<bool> emit_msan,
    JitObserver* observer) {
  absl::call_once(once, OnceInit);
#ifdef ABSL_HAVE_MEMORY_SANITIZER
  constexpr bool kHasMsan = true;
#else
  constexpr bool kHasMsan = false;
#endif
  std::unique_ptr<OrcJit> jit = absl::WrapUnique(
      new OrcJit(opt_level, emit_object_code, emit_msan.value_or(kHasMsan)));
  jit->SetJitObserver(observer);
  XLS_RETURN_IF_ERROR(jit->Init());
  return std::move(jit);
}

/* static */ absl::StatusOr<llvm::DataLayout> OrcJit::CreateDataLayout(
    bool aot_specification) {
  absl::call_once(once, OnceInit);
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<llvm::TargetMachine> target_machine,
                       CreateTargetMachine(aot_specification));
  return target_machine->createDataLayout();
}

/* static */ absl::StatusOr<std::unique_ptr<llvm::TargetMachine>>
OrcJit::CreateTargetMachine(bool aot_specification) {
  auto error_or_target_builder =
      llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!error_or_target_builder) {
    return absl::InternalError(
        absl::StrCat("Unable to detect host: ",
                     llvm::toString(error_or_target_builder.takeError())));
  }

  error_or_target_builder->setRelocationModel(llvm::Reloc::Model::PIC_);
  if (aot_specification) {
    // In ahead-of-time compilation we're compiling on machines we are not
    // immediately about to run on, where runtime machines may have
    // heterogeneous specifications vs the compilation machine. We assume a
    // baseline level of compatibility for all machines XLS compilations might
    // run on.
    switch (error_or_target_builder->getTargetTriple().getArch()) {
      case llvm::Triple::x86_64: {
        const std::string kBaselineCpu = "haswell";
        error_or_target_builder->setCPU(kBaselineCpu);
        // Clear out the existing features.
        error_or_target_builder->getFeatures() = llvm::SubtargetFeatures();
        // Add in features available on our "baseline" CPU.
        llvm::SmallVector<llvm::StringRef, 32> target_cpu_features;
        llvm::X86::getFeaturesForCPU(kBaselineCpu, target_cpu_features,
                                     /*NeedPlus=*/true);
        std::vector<std::string> features(target_cpu_features.begin(),
                                          target_cpu_features.end());
        error_or_target_builder->addFeatures(features);
        break;
      }
      case llvm::Triple::aarch64:
        return absl::UnimplementedError(
            "Support AOT feature flag setting for aarch64 compilation.");
      default:
        return absl::InvalidArgumentError(
            "Compiling on unrecognized host architecture for AOT "
            "compilation: " +
            std::string{
                error_or_target_builder->getTargetTriple().getArchName()});
    }
  }
  auto error_or_target_machine = error_or_target_builder->createTargetMachine();
  if (!error_or_target_machine) {
    return absl::InternalError(
        absl::StrCat("Unable to create target machine: ",
                     llvm::toString(error_or_target_machine.takeError())));
  }
  return std::move(error_or_target_machine.get());
}

namespace {
// Based on https://github.com/google/sanitizers/wiki/MemorySanitizerJIT
// tutorial.

// Identifiers we use to pick out the actual thread-local buffers shared with
// host msan. Basically the msan ABI is:
// %x = load-symbol __emutls_v.__msan_param_tls
// %y = load-symbol __emutls_get_address
// %tls_slot = invoke %y (%x)
#ifdef ABSL_HAVE_MEMORY_SANITIZER
static constexpr uintptr_t kParamTlsEntry = 1;
static constexpr uintptr_t kRetvalTlsEntry = 2;
// TODO(allight): Technically if we want to support origin-tracking we could but
// we'd need to add more of the locals from MSan.cpp here.
extern "C" __thread unsigned long long  // NOLINT(runtime/int)
    __msan_param_tls[];
extern "C" __thread unsigned long long  // NOLINT(runtime/int)
    __msan_retval_tls[];
void* GetMsanTLSAddr(void* ctx) {
  switch (absl::bit_cast<uintptr_t>(ctx)) {
    case kParamTlsEntry:
      return absl::bit_cast<void*>(&__msan_param_tls);
    case kRetvalTlsEntry:
      return absl::bit_cast<void*>(&__msan_retval_tls);
    default:
      LOG(ERROR) << "Unexpected TLS addr request: " << ctx;
      return nullptr;
  }
}
#endif

class MsanHostEmuTls : public llvm::orc::DefinitionGenerator {
 public:
  llvm::Error tryToGenerate(llvm::orc::LookupState&, llvm::orc::LookupKind,
                            llvm::orc::JITDylib& dylib,
                            llvm::orc::JITDylibLookupFlags,
                            const llvm::orc::SymbolLookupSet& targets) final {
#ifndef ABSL_HAVE_MEMORY_SANITIZER
    // No actual msan so don't do anything.
    return llvm::Error::success();
#else
    llvm::orc::SymbolMap result;
    for (auto& kv : targets) {
      auto name = (*kv.first).str();
      if (name == "__emutls_get_address") {
        result[kv.first] = llvm::orc::ExecutorSymbolDef(
            {llvm::orc::ExecutorAddr::fromPtr(GetMsanTLSAddr),
             llvm::JITSymbolFlags::Exported});
      }
      if (name == "__emutls_v.__msan_param_tls") {
        result[kv.first] = llvm::orc::ExecutorSymbolDef(
            {llvm::orc::ExecutorAddr::fromPtr(
                 absl::bit_cast<void*>(kParamTlsEntry)),
             llvm::JITSymbolFlags::Exported});
      }
      if (name == "__emutls_v.__msan_retval_tls") {
        result[kv.first] = llvm::orc::ExecutorSymbolDef(
            {llvm::orc::ExecutorAddr::fromPtr(
                 absl::bit_cast<void*>(kRetvalTlsEntry)),
             llvm::JITSymbolFlags::Exported});
      }
    }
    if (result.empty()) {
      return llvm::Error::success();
    }
    return dylib.define(llvm::orc::absoluteSymbols(std::move(result)));
#endif
  }
};
}  // namespace

absl::Status OrcJit::Init() {
  XLS_ASSIGN_OR_RETURN(target_machine_, CreateTargetMachine(emit_object_code_));
  if (VLOG_IS_ON(1)) {
    std::string triple = target_machine_->getTargetTriple().normalize();
    std::string cpu = target_machine_->getTargetCPU().str();
    std::string feature_string =
        target_machine_->getTargetFeatureString().str();
    XLS_VLOG(1) << "LLVM target triple: " << triple;
    XLS_VLOG(1) << "LLVM target CPU: " << cpu;
    XLS_VLOG(1) << "LLVM target feature string: " << feature_string;
    XLS_VLOG_LINES(
        1,
        absl::StrFormat("llc invocation:\n  llc [input.ll] --filetype=obj "
                        "--relocation-model=pic -mtriple=%s -mcpu=%s -mattr=%s",
                        triple, cpu, feature_string));
  }
  data_layout_ = target_machine_->createDataLayout();

  execution_session_.runSessionLocked([this]() {
    dylib_.addGenerator(std::make_unique<MsanHostEmuTls>());
    dylib_.addGenerator(
        cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            data_layout_.getGlobalPrefix())));
  });

  auto compiler = std::make_unique<llvm::orc::SimpleCompiler>(*target_machine_);
  compile_layer_ = std::make_unique<llvm::orc::IRCompileLayer>(
      execution_session_, object_layer_, std::move(compiler));

  if (emit_object_code_) {
    object_code_layer_ = std::make_unique<llvm::orc::IRTransformLayer>(
        execution_session_, *compile_layer_,
        [this](llvm::orc::ThreadSafeModule module,
               const llvm::orc::MaterializationResponsibility& mr) {
          llvm::SmallVector<char, 0> stream_buffer;
          llvm::raw_svector_ostream ostream(stream_buffer);
          llvm::legacy::PassManager mpm;
          if (target_machine_->addPassesToEmitFile(
                  mpm, ostream, nullptr, llvm::CodeGenFileType::ObjectFile)) {
            XLS_VLOG(3) << "Could not create ASM generation pass!";
          }
          mpm.run(*module.getModuleUnlocked());

          object_code_ =
              std::vector<uint8_t>(stream_buffer.begin(), stream_buffer.end());
          return module;
        });
  }

  llvm::orc::IRLayer* parent_layer =
      emit_object_code_
          ? static_cast<llvm::orc::IRLayer*>(object_code_layer_.get())
          : static_cast<llvm::orc::IRLayer*>(compile_layer_.get());
  transform_layer_ = std::make_unique<llvm::orc::IRTransformLayer>(
      execution_session_, *parent_layer,
      [this](llvm::orc::ThreadSafeModule module,
             const llvm::orc::MaterializationResponsibility& responsibility) {
        return Optimizer(std::move(module), responsibility);
      });

  return absl::OkStatus();
}

std::unique_ptr<llvm::Module> OrcJit::NewModule(std::string_view name) {
  llvm::LLVMContext* bare_context = context_.getContext();
  auto module = std::make_unique<llvm::Module>(name, *bare_context);
  module->setDataLayout(data_layout_);
  return module;
}

namespace {

// Check that every operand of every instruction is in the same function as the
// instruction itself. This is a very common error which is not checked by the
// LLVM verifier(!).
absl::Status VerifyModule(const llvm::Module& module) {
  XLS_VLOG(4) << DumpLlvmModuleToString(&module);
  for (const llvm::Function& function : module.functions()) {
    for (const llvm::BasicBlock& basic_block : function) {
      for (const llvm::Instruction& inst : basic_block) {
        int64_t i = 0;
        for (const llvm::Use& use : inst.operands()) {
          if (const llvm::Instruction* inst_use =
                  llvm::dyn_cast<llvm::Instruction>(&use)) {
            XLS_RET_CHECK_EQ(inst_use->getFunction(), &function)
                << absl::StreamFormat(
                       "In function `%s`, operand %d of this instruction is "
                       "from different function `%s`: %s",
                       function.getName().str(), i,
                       inst_use->getFunction()->getName().str(),
                       DumpLlvmObjectToString(inst));
          } else if (const llvm::Argument* arg_use =
                         llvm::dyn_cast<llvm::Argument>(&use)) {
            XLS_RET_CHECK_EQ(arg_use->getParent(), &function)
                << absl::StreamFormat(
                       "In function `%s`, operand %d of this instruction is "
                       "from different function `%s`: %s",
                       function.getName().str(), i,
                       arg_use->getParent()->getName().str(),
                       DumpLlvmObjectToString(inst));
          }

          ++i;
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status OrcJit::CompileModule(std::unique_ptr<llvm::Module>&& module) {
  XLS_RETURN_IF_ERROR(VerifyModule(*module));
  llvm::Error error = transform_layer_->add(
      dylib_, llvm::orc::ThreadSafeModule(std::move(module), context_));
  if (error) {
    return absl::UnknownError(absl::StrFormat(
        "Error compiling converted IR: %s", llvm::toString(std::move(error))));
  }
  return absl::OkStatus();
}

absl::StatusOr<llvm::orc::ExecutorAddr> OrcJit::LoadSymbol(
    std::string_view function_name) {
#ifdef __APPLE__
  // On Apple systems, symbols are still prepended with an underscore.
  std::string function_name_with_underscore = absl::StrCat("_", function_name);
  function_name = function_name_with_underscore;
#endif /* __APPLE__ */
  llvm::Expected<llvm::orc::ExecutorSymbolDef> symbol =
      execution_session_.lookup(&dylib_, function_name);
  if (!symbol) {
    return absl::InternalError(
        absl::StrFormat("Could not find start symbol \"%s\": %s", function_name,
                        llvm::toString(symbol.takeError())));
  }
  return symbol->getAddress();
}

std::string DumpLlvmModuleToString(const llvm::Module& module) {
  std::string buffer;
  llvm::raw_string_ostream ostream(buffer);
  module.print(ostream, nullptr);
  ostream.flush();
  return buffer;
}

}  // namespace xls
