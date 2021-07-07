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

#include "xls/jit/ir_jit.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/include/llvm-c/Target.h"
#include "llvm/include/llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/include/llvm/Analysis/TargetTransformInfo.h"
#include "llvm/include/llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/include/llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/include/llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/include/llvm/IR/BasicBlock.h"
#include "llvm/include/llvm/IR/Constants.h"
#include "llvm/include/llvm/IR/DerivedTypes.h"
#include "llvm/include/llvm/IR/Instructions.h"
#include "llvm/include/llvm/IR/Intrinsics.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "llvm/include/llvm/IR/LegacyPassManager.h"
#include "llvm/include/llvm/IR/Module.h"
#include "llvm/include/llvm/IR/Value.h"
#include "llvm/include/llvm/Support/CodeGen.h"
#include "llvm/include/llvm/Support/DynamicLibrary.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "llvm/include/llvm/Target/TargetMachine.h"
#include "llvm/include/llvm/Transforms/IPO/PassManagerBuilder.h"
#include "xls/codegen/vast.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/logging/vlog_is_on.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/keyword_args.h"
#include "xls/ir/proc.h"
#include "xls/ir/random_value.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_helpers.h"
#include "xls/jit/function_builder_visitor.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/llvm_type_converter.h"

namespace xls {
namespace {

absl::once_flag once;
void OnceInit() {
  LLVMInitializeNativeTarget();
  LLVMInitializeNativeAsmPrinter();
  LLVMInitializeNativeAsmParser();
}

}  // namespace

IrJit::~IrJit() {
  if (auto err = execution_session_.endSession()) {
    execution_session_.reportError(std::move(err));
  }
}

absl::StatusOr<std::unique_ptr<IrJit>> IrJit::Create(Function* xls_function,
                                                     int64_t opt_level) {
  absl::call_once(once, OnceInit);

  auto jit = absl::WrapUnique(new IrJit(xls_function, opt_level));
  XLS_RETURN_IF_ERROR(jit->Init());
  auto visit_fn = [&jit](llvm::Module* module, llvm::Function* llvm_function,
                         bool generate_packed) {
    return FunctionBuilderVisitor::Visit(
        module, llvm_function, jit->xls_function_, jit->type_converter_.get(),
        /*is_top=*/true, generate_packed);
  };
  XLS_RETURN_IF_ERROR(jit->Compile(visit_fn));
  return jit;
}

absl::StatusOr<std::unique_ptr<IrJit>> IrJit::CreateProc(
    Proc* proc, JitChannelQueueManager* queue_mgr,
    ProcBuilderVisitor::RecvFnT recv_fn, ProcBuilderVisitor::SendFnT send_fn,
    int64_t opt_level) {
  absl::call_once(once, OnceInit);

  auto jit = absl::WrapUnique(new IrJit(proc, opt_level));
  XLS_RETURN_IF_ERROR(jit->Init());
  auto visit_fn = [&jit, queue_mgr, recv_fn, send_fn](
                      llvm::Module* module, llvm::Function* llvm_function,
                      bool generate_packed) {
    return ProcBuilderVisitor::Visit(
        module, llvm_function, jit->xls_function_, jit->type_converter_.get(),
        /*is_top=*/true, generate_packed, queue_mgr, recv_fn, send_fn);
  };
  XLS_RETURN_IF_ERROR(jit->Compile(visit_fn));
  return jit;
}

absl::Status IrJit::Compile(VisitFn visit_fn) {
  llvm::LLVMContext* bare_context = context_.getContext();
  auto module = std::make_unique<llvm::Module>("the_module", *bare_context);
  module->setDataLayout(data_layout_);
  XLS_RETURN_IF_ERROR(CompileFunction(visit_fn, module.get()));
  XLS_RETURN_IF_ERROR(CompilePackedViewFunction(visit_fn, module.get()));
  llvm::Error error = transform_layer_->add(
      dylib_, llvm::orc::ThreadSafeModule(std::move(module), context_));
  if (error) {
    return absl::UnknownError(absl::StrFormat(
        "Error compiling converted IR: %s", llvm::toString(std::move(error))));
  }

  auto load_symbol = [this](const std::string& function_name)
      -> absl::StatusOr<llvm::JITTargetAddress> {
    llvm::Expected<llvm::JITEvaluatedSymbol> symbol =
        execution_session_.lookup(&dylib_, function_name);
    if (!symbol) {
      return absl::InternalError(
          absl::StrFormat("Could not find start symbol \"%s\": %s",
                          function_name, llvm::toString(symbol.takeError())));
    }
    return symbol->getAddress();
  };

  std::string function_name = absl::StrFormat(
      "%s::%s", xls_function_->package()->name(), xls_function_->name());
  XLS_ASSIGN_OR_RETURN(auto fn_address, load_symbol(function_name));
  invoker_ = absl::bit_cast<JitFunctionType>(fn_address);

  absl::StrAppend(&function_name, "_packed");
  XLS_ASSIGN_OR_RETURN(fn_address, load_symbol(function_name));
  packed_invoker_ = absl::bit_cast<PackedJitFunctionType>(fn_address);

  return absl::OkStatus();
}

IrJit::IrJit(FunctionBase* xls_function, int64_t opt_level)
    : context_(std::make_unique<llvm::LLVMContext>()),
      object_layer_(
          execution_session_,
          []() { return std::make_unique<llvm::SectionMemoryManager>(); }),
      dylib_(execution_session_.createBareJITDylib("main")),
      data_layout_(""),
      xls_function_(xls_function),
      opt_level_(opt_level),
      invoker_(nullptr) {}

llvm::Expected<llvm::orc::ThreadSafeModule> IrJit::Optimizer(
    llvm::orc::ThreadSafeModule module,
    const llvm::orc::MaterializationResponsibility& responsibility) {
  llvm::Module* bare_module = module.getModuleUnlocked();

  XLS_VLOG(2) << "Unoptimized module IR:";
  XLS_VLOG(2).NoPrefix() << ir_runtime_->DumpToString(*bare_module);

  llvm::TargetLibraryInfoImpl library_info(target_machine_->getTargetTriple());
  llvm::PassManagerBuilder builder;
  builder.OptLevel = opt_level_;
  builder.LibraryInfo =
      new llvm::TargetLibraryInfoImpl(target_machine_->getTargetTriple());

  // The ostream and its buffer must be declared before the module_pass_manager
  // because the destrutor of the pass manager calls flush on the ostream so
  // these must be destructed *after* the pass manager. C++ guarantees that the
  // destructors are called in reverse order the obects are declared.
  llvm::SmallVector<char, 0> stream_buffer;
  llvm::raw_svector_ostream ostream(stream_buffer);

  llvm::legacy::PassManager module_pass_manager;
  builder.populateModulePassManager(module_pass_manager);
  module_pass_manager.add(llvm::createTargetTransformInfoWrapperPass(
      target_machine_->getTargetIRAnalysis()));

  llvm::legacy::FunctionPassManager function_pass_manager(bare_module);
  builder.populateFunctionPassManager(function_pass_manager);
  function_pass_manager.doInitialization();
  for (auto& function : *bare_module) {
    function_pass_manager.run(function);
  }
  function_pass_manager.doFinalization();

  bool dump_asm = false;
  if (XLS_VLOG_IS_ON(3)) {
    dump_asm = true;
    if (target_machine_->addPassesToEmitFile(
            module_pass_manager, ostream, nullptr, llvm::CGFT_AssemblyFile)) {
      XLS_VLOG(3) << "Could not create ASM generation pass!";
      dump_asm = false;
    }
  }

  module_pass_manager.run(*bare_module);

  XLS_VLOG(2) << "Optimized module IR:";
  XLS_VLOG(2).NoPrefix() << ir_runtime_->DumpToString(*bare_module);

  if (dump_asm) {
    XLS_VLOG(3) << "Generated ASM:";
    XLS_VLOG_LINES(3, std::string(stream_buffer.begin(), stream_buffer.end()));
  }
  return module;
}

absl::Status IrJit::Init() {
  auto error_or_target_builder =
      llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!error_or_target_builder) {
    return absl::InternalError(
        absl::StrCat("Unable to detect host: ",
                     llvm::toString(error_or_target_builder.takeError())));
  }

  auto error_or_target_machine = error_or_target_builder->createTargetMachine();
  if (!error_or_target_machine) {
    return absl::InternalError(
        absl::StrCat("Unable to create target machine: ",
                     llvm::toString(error_or_target_machine.takeError())));
  }
  target_machine_ = std::move(error_or_target_machine.get());
  data_layout_ = target_machine_->createDataLayout();
  type_converter_ =
      std::make_unique<LlvmTypeConverter>(context_.getContext(), data_layout_);

  execution_session_.runSessionLocked([this]() {
    dylib_.addGenerator(
        cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            data_layout_.getGlobalPrefix())));
  });

  auto compiler = std::make_unique<llvm::orc::SimpleCompiler>(*target_machine_);
  compile_layer_ = std::make_unique<llvm::orc::IRCompileLayer>(
      execution_session_, object_layer_, std::move(compiler));

  transform_layer_ = std::make_unique<llvm::orc::IRTransformLayer>(
      execution_session_, *compile_layer_,
      [this](llvm::orc::ThreadSafeModule module,
             const llvm::orc::MaterializationResponsibility& responsibility) {
        return Optimizer(std::move(module), responsibility);
      });

  ir_runtime_ =
      std::make_unique<JitRuntime>(data_layout_, type_converter_.get());

  return absl::OkStatus();
}

absl::Status IrJit::CompileFunction(VisitFn visit_fn, llvm::Module* module) {
  llvm::LLVMContext* bare_context = context_.getContext();

  // To return values > 64b in size, we need to copy them into a result buffer,
  // instead of returning a fixed-size result element.
  // To do this, we need to construct the function type, adding a result buffer
  // arg (and setting the result type to void) and then storing the computation
  // result therein.
  std::vector<llvm::Type*> param_types;
  // Represent the input args as char/i8 pointers to their data.
  param_types.push_back(llvm::PointerType::get(
      llvm::ArrayType::get(
          llvm::PointerType::get(llvm::Type::getInt8Ty(*bare_context),
                                 /*AddressSpace=*/0),
          xls_function_->params().size()),
      /*AddressSpace=*/0));

  for (const Param* param : xls_function_->params()) {
    arg_type_bytes_.push_back(
        type_converter_->GetTypeByteSize(param->GetType()));
  }

  // Pass the last param as a pointer to the actual return type.
  Type* return_type =
      FunctionBuilderVisitor::GetEffectiveReturnValue(xls_function_)->GetType();
  llvm::Type* llvm_return_type =
      type_converter_->ConvertToLlvmType(return_type);
  param_types.push_back(
      llvm::PointerType::get(llvm_return_type, /*AddressSpace=*/0));

  // Treat void pointers as int64_t values at the LLVM IR level.
  // Using an actual pointer type triggers LLVM asserts when compiling
  // in debug mode.
  // TODO(amfv): 2021-04-05 Figure out why and fix void pointer handling.
  llvm::Type* void_ptr_type = llvm::Type::getInt64Ty(*bare_context);

  // assertion status argument
  param_types.push_back(void_ptr_type);
  // user data argument
  param_types.push_back(void_ptr_type);

  llvm::FunctionType* function_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(*bare_context),
      llvm::ArrayRef<llvm::Type*>(param_types.data(), param_types.size()),
      /*isVarArg=*/false);

  Package* xls_package = xls_function_->package();
  std::string function_name =
      absl::StrFormat("%s::%s", xls_package->name(), xls_function_->name());
  llvm::Function* llvm_function = llvm::cast<llvm::Function>(
      module->getOrInsertFunction(function_name, function_type).getCallee());
  return_type_bytes_ = type_converter_->GetTypeByteSize(return_type);
  XLS_RETURN_IF_ERROR(
      visit_fn(module, llvm_function, /*generate_packed=*/false));

  return absl::OkStatus();
}

absl::StatusOr<Value> IrJit::Run(absl::Span<const Value> args,
                                 void* user_data) {
  absl::Span<Param* const> params = xls_function_->params();
  if (args.size() != params.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Arg list to '%s' has the wrong size: %d vs expected %d.",
        xls_function_->name(), args.size(), xls_function_->params().size()));
  }

  for (int i = 0; i < params.size(); i++) {
    if (!ValueConformsToType(args[i], params[i]->GetType())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Got argument %s for parameter %d which is not of type %s",
          args[i].ToString(), i, params[i]->GetType()->ToString()));
    }
  }

  std::vector<std::unique_ptr<uint8_t[]>> unique_arg_buffers;
  std::vector<uint8_t*> arg_buffers;
  unique_arg_buffers.reserve(xls_function_->params().size());
  arg_buffers.reserve(unique_arg_buffers.size());
  std::vector<Type*> param_types;
  for (const Param* param : xls_function_->params()) {
    unique_arg_buffers.push_back(std::make_unique<uint8_t[]>(
        type_converter_->GetTypeByteSize(param->GetType())));
    arg_buffers.push_back(unique_arg_buffers.back().get());
    param_types.push_back(param->GetType());
  }

  XLS_RETURN_IF_ERROR(
      ir_runtime_->PackArgs(args, param_types, absl::MakeSpan(arg_buffers)));

  absl::Status assert_status = absl::OkStatus();

  absl::InlinedVector<uint8_t, 16> outputs(return_type_bytes_);
  invoker_(arg_buffers.data(), outputs.data(), &assert_status, user_data);

  if (!assert_status.ok()) {
    return assert_status;
  }

  return ir_runtime_->UnpackBuffer(
      outputs.data(),
      FunctionBuilderVisitor::GetEffectiveReturnValue(xls_function_)
          ->GetType());
}

absl::StatusOr<Value> IrJit::Run(
    const absl::flat_hash_map<std::string, Value>& kwargs, void* user_data) {
  XLS_ASSIGN_OR_RETURN(std::vector<Value> positional_args,
                       KeywordArgsToPositional(*xls_function_, kwargs));
  return Run(positional_args, user_data);
}

absl::Status IrJit::RunWithViews(absl::Span<uint8_t*> args,
                                 absl::Span<uint8_t> result_buffer,
                                 void* user_data) {
  absl::Span<Param* const> params = xls_function_->params();
  if (args.size() != params.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Arg list has the wrong size: %d vs expected %d.",
                        args.size(), xls_function_->params().size()));
  }

  if (result_buffer.size() < return_type_bytes_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Result buffer too small - must be at least %d bytes!",
                     return_type_bytes_));
  }

  absl::Status assert_status = absl::OkStatus();

  invoker_(args.data(), result_buffer.data(), &assert_status, user_data);
  return assert_status;
}

absl::StatusOr<Value> CreateAndRun(Function* xls_function,
                                   absl::Span<const Value> args) {
  // No proc support from Python yet.
  XLS_ASSIGN_OR_RETURN(auto jit, IrJit::Create(xls_function));
  XLS_ASSIGN_OR_RETURN(auto result, jit->Run(args));
  return result;
}

// Much of the core here is the same as in CompileFunction() - refer there for
// general comments.
absl::Status IrJit::CompilePackedViewFunction(VisitFn visit_fn,
                                              llvm::Module* module) {
  llvm::LLVMContext* bare_context = context_.getContext();
  llvm::Type* i8_type = llvm::Type::getInt8Ty(*bare_context);

  // Create arg packing/unpacking buffers as in CompileFunction().
  std::vector<llvm::Type*> param_types;
  llvm::FunctionType* function_type;
  // Represent the input args as char/i8 pointers to their data.
  param_types.push_back(llvm::PointerType::get(
      llvm::ArrayType::get(llvm::PointerType::get(i8_type, /*AddressSpace=*/0),
                           xls_function_->params().size()),
      /*AddressSpace=*/0));

  int64_t return_width =
      FunctionBuilderVisitor::GetEffectiveReturnValue(xls_function_)
          ->GetType()
          ->GetFlatBitCount();
  if (return_width != 0) {
    // For packed operation, just pass a i8 pointer for the result.
    llvm::Type* return_type =
        llvm::IntegerType::get(*bare_context, return_width);
    param_types.push_back(
        llvm::PointerType::get(return_type, /*AddressSpace=*/0));
  }
  // assertion status
  param_types.push_back(llvm::Type::getInt64Ty(*bare_context));
  // user data
  param_types.push_back(llvm::Type::getInt64Ty(*bare_context));
  function_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(*bare_context),
      llvm::ArrayRef<llvm::Type*>(param_types.data(), param_types.size()),
      /*isVarArg=*/false);

  Package* xls_package = xls_function_->package();
  std::string function_name = absl::StrFormat(
      "%s::%s_packed", xls_package->name(), xls_function_->name());
  llvm::Function* llvm_function = llvm::cast<llvm::Function>(
      module->getOrInsertFunction(function_name, function_type).getCallee());
  XLS_RETURN_IF_ERROR(
      visit_fn(module, llvm_function, /*generate_packed=*/true));

  return absl::OkStatus();
}

}  // namespace xls
