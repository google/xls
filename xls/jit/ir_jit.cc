// Copyright 2020 Google LLC
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
#include <memory>
#include <random>

#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm-c/Target.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "xls/codegen/vast.h"
#include "xls/common/integral_types.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/logging/vlog_is_on.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/keyword_args.h"
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

absl::StatusOr<std::unique_ptr<IrJit>> IrJit::Create(
    Function* xls_function, ChannelQueueManager* queue_mgr, int64 opt_level) {
  absl::call_once(once, OnceInit);

  auto jit = absl::WrapUnique(new IrJit(xls_function, queue_mgr, opt_level));
  XLS_RETURN_IF_ERROR(jit->Init());
  XLS_RETURN_IF_ERROR(jit->Compile(absl::nullopt));
  return jit;
}

absl::Status IrJit::Compile(absl::optional<ChannelQueueManager*> queue_mgr) {
  llvm::LLVMContext* bare_context = context_.getContext();
  auto module = std::make_unique<llvm::Module>("the_module", *bare_context);
  module->setDataLayout(data_layout_);
  XLS_RETURN_IF_ERROR(CompileFunction(module.get()));
  XLS_RETURN_IF_ERROR(CompilePackedViewFunction(module.get()));
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
  invoker_ = reinterpret_cast<JitFunctionType>(fn_address);

  absl::StrAppend(&function_name, "_packed");
  XLS_ASSIGN_OR_RETURN(fn_address, load_symbol(function_name));
  packed_invoker_ = reinterpret_cast<PackedJitFunctionType>(fn_address);

  return absl::OkStatus();
}

IrJit::IrJit(Function* xls_function, ChannelQueueManager* queue_mgr,
             int64 opt_level)
    : context_(std::make_unique<llvm::LLVMContext>()),
      object_layer_(
          execution_session_,
          []() { return std::make_unique<llvm::SectionMemoryManager>(); }),
      dylib_(execution_session_.createBareJITDylib("main")),
      data_layout_(""),
      xls_function_(xls_function),
      xls_function_type_(xls_function_->GetType()),
      queue_mgr_(queue_mgr),
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
  llvm::SmallVector<char, 0> stream_buffer;
  llvm::raw_svector_ostream ostream(stream_buffer);
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

absl::Status IrJit::CompileFunction(llvm::Module* module) {
  llvm::LLVMContext* bare_context = context_.getContext();

  // To return values > 64b in size, we need to copy them into a result buffer,
  // instead of returning a fixed-size result element.
  // To do this, we need to construct the function type, adding a result buffer
  // arg (and setting the result type to void) and then storing the computation
  // result therein.
  std::vector<llvm::Type*> param_types;
  llvm::FunctionType* function_type;
  // Represent the input args as char/i8 pointers to their data.
  param_types.push_back(llvm::PointerType::get(
      llvm::ArrayType::get(
          llvm::PointerType::get(llvm::Type::getInt8Ty(*bare_context),
                                 /*AddressSpace=*/0),
          xls_function_type_->parameter_count()),
      /*AddressSpace=*/0));

  for (const Type* type : xls_function_type_->parameters()) {
    arg_type_bytes_.push_back(type_converter_->GetTypeByteSize(*type));
  }

  // Pass the last param as a pointer to the actual return type.
  Type* return_type = xls_function_type_->return_type();
  llvm::Type* llvm_return_type =
      type_converter_->ConvertToLlvmType(*return_type);
  param_types.push_back(
      llvm::PointerType::get(llvm_return_type, /*AddressSpace=*/0));
  function_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(*bare_context),
      llvm::ArrayRef<llvm::Type*>(param_types.data(), param_types.size()),
      /*isVarArg=*/false);

  Package* xls_package = xls_function_->package();
  std::string function_name =
      absl::StrFormat("%s::%s", xls_package->name(), xls_function_->name());
  llvm::Function* llvm_function = llvm::cast<llvm::Function>(
      module->getOrInsertFunction(function_name, function_type).getCallee());
  return_type_bytes_ = type_converter_->GetTypeByteSize(*return_type);
  XLS_RETURN_IF_ERROR(
      FunctionBuilderVisitor::Build(module, llvm_function, xls_function_,
                                    type_converter_.get(),
                                    /*is_top=*/true, /*generate_packed=*/false)
          .status());

  return absl::OkStatus();
}

absl::StatusOr<Value> IrJit::Run(absl::Span<const Value> args) {
  absl::Span<Param* const> params = xls_function_->params();
  if (args.size() != params.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Arg list has the wrong size: %d vs expected %d.",
                        args.size(), xls_function_->params().size()));
  }

  for (int i = 0; i < params.size(); i++) {
    if (!ValueConformsToType(args[i], params[i]->GetType())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Got argument %s for parameter %d which is not of type %s",
          args[i].ToString(), i, params[i]->GetType()->ToString()));
    }
  }

  std::vector<std::unique_ptr<uint8[]>> unique_arg_buffers;
  std::vector<uint8*> arg_buffers;
  unique_arg_buffers.reserve(xls_function_type_->parameters().size());
  arg_buffers.reserve(unique_arg_buffers.size());
  for (const Type* type : xls_function_type_->parameters()) {
    unique_arg_buffers.push_back(
        std::make_unique<uint8[]>(type_converter_->GetTypeByteSize(*type)));
    arg_buffers.push_back(unique_arg_buffers.back().get());
  }

  XLS_RETURN_IF_ERROR(ir_runtime_->PackArgs(
      args, xls_function_type_->parameters(), absl::MakeSpan(arg_buffers)));

  absl::InlinedVector<uint8, 16> outputs(return_type_bytes_);
  invoker_(arg_buffers.data(), outputs.data());

  return ir_runtime_->UnpackBuffer(outputs.data(),
                                   xls_function_type_->return_type());
}

absl::StatusOr<Value> IrJit::Run(
    const absl::flat_hash_map<std::string, Value>& kwargs) {
  XLS_ASSIGN_OR_RETURN(std::vector<Value> positional_args,
                       KeywordArgsToPositional(*xls_function_, kwargs));
  return Run(positional_args);
}

absl::Status IrJit::RunWithViews(absl::Span<const uint8*> args,
                                 absl::Span<uint8> result_buffer) {
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

  invoker_(args.data(), result_buffer.data());
  return absl::OkStatus();
}

absl::StatusOr<Value> CreateAndRun(Function* xls_function,
                                   absl::Span<const Value> args) {
  // No proc support from Python yet.
  XLS_ASSIGN_OR_RETURN(auto jit,
                       IrJit::Create(xls_function, /*queue_mgr=*/nullptr));
  XLS_ASSIGN_OR_RETURN(auto result, jit->Run(args));
  return result;
}

absl::StatusOr<std::pair<std::vector<std::vector<Value>>, std::vector<Value>>>
CreateAndQuickCheck(Function* xls_function, int64 seed, int64 num_tests) {
  // No proc support from Python yet.
  XLS_ASSIGN_OR_RETURN(auto jit,
                       IrJit::Create(xls_function, /*queue_mgr=*/nullptr));
  std::vector<Value> results;
  std::vector<std::vector<Value>> argsets;
  std::minstd_rand rng_engine(seed);

  for (int i = 0; i < num_tests; i++) {
    argsets.push_back(RandomFunctionArguments(xls_function, &rng_engine));
    XLS_ASSIGN_OR_RETURN(auto result, jit->Run(argsets[i]));
    results.push_back(result);
    if (result.IsAllZeros())
      // We were able to falsify the xls_function (predicate), bail out early
      // and present this evidence.
      break;
  }

  return std::make_pair(argsets, results);
}

// Much of the core here is the same as in CompileFunction() - refer there for
// general comments.
absl::Status IrJit::CompilePackedViewFunction(llvm::Module* module) {
  llvm::LLVMContext* bare_context = context_.getContext();
  llvm::Type* i8_type = llvm::Type::getInt8Ty(*bare_context);

  // Create arg packing/unpacking buffers as in CompileFunction().
  std::vector<llvm::Type*> param_types;
  llvm::FunctionType* function_type;
  // Represent the input args as char/i8 pointers to their data.
  param_types.push_back(llvm::PointerType::get(
      llvm::ArrayType::get(llvm::PointerType::get(i8_type, /*AddressSpace=*/0),
                           xls_function_type_->parameter_count()),
      /*AddressSpace=*/0));

  int64 return_width =
      xls_function_->return_value()->GetType()->GetFlatBitCount();
  if (return_width != 0) {
    // For packed operation, just pass a i8 pointer for the result.
    llvm::Type* return_type =
        llvm::IntegerType::get(*bare_context, return_width);
    param_types.push_back(
        llvm::PointerType::get(return_type, /*AddressSpace=*/0));
  }
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
      FunctionBuilderVisitor::Build(module, llvm_function, xls_function_,
                                    type_converter_.get(),
                                    /*is_top=*/true, /*generate_packed=*/true)
          .status());

  return absl::OkStatus();
}

}  // namespace xls
