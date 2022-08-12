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

#include "xls/jit/function_jit.h"

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "llvm/include/llvm/IR/Module.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/logging/vlog_is_on.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/keyword_args.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_helpers.h"
#include "xls/jit/ir_builder_visitor.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/llvm_type_converter.h"

namespace xls {
namespace {

// An IR builder visitor which constructs llvm::Functions built from XLS
// functions.
class FunctionBuilderVisitor : public IrBuilderVisitor {
 public:
  FunctionBuilderVisitor(
      llvm::Function* llvm_fn, Function* xls_fn,
      LlvmTypeConverter* type_converter,
      std::function<absl::StatusOr<llvm::Function*>(Function*)>
          function_builder)
      : IrBuilderVisitor(llvm_fn, xls_fn, type_converter, function_builder) {}

  absl::Status HandleParam(Param* param) override {
    // Params are passed by value as the first arguments to the function.
    XLS_ASSIGN_OR_RETURN(int index,
                         param->function_base()->GetParamIndex(param));
    llvm::Function* llvm_function =
        dispatch_builder()->GetInsertBlock()->getParent();
    llvm::Value* result = type_converter()->ClearPaddingBits(
        llvm_function->getArg(index), param->GetType(), *dispatch_builder());
    return StoreResult(param, result);
  }

  // Finishes building the function by adding a return statement.
  absl::Status Finalize() {
    Node* return_value = xls_fn_->AsFunctionOrDie()->return_value();
    dispatch_builder()->CreateRet(node_map().at(return_value));
    return absl::OkStatus();
  }
};

// Recursive helper for packing LLVM values representing structured XLS types
// into a flat bit vector.
absl::StatusOr<llvm::Value*> PackValueHelper(llvm::Value* element,
                                             Type* element_type,
                                             llvm::Value* buffer,
                                             int64_t bit_offset,
                                             llvm::IRBuilder<>* builder) {
  switch (element_type->kind()) {
    case TypeKind::kBits:
      if (element->getType() != buffer->getType()) {
        if (element->getType()->getIntegerBitWidth() >
            buffer->getType()->getIntegerBitWidth()) {
          // The LLVM type of the subelement is wider than the packed value of
          // the entire type. This can happen because bits types are padded up
          // to powers of two.
          element = builder->CreateTrunc(element, buffer->getType());
        } else {
          element = builder->CreateZExt(element, buffer->getType());
        }
      }
      element = builder->CreateShl(element, bit_offset);
      return builder->CreateOr(buffer, element);
    case TypeKind::kArray: {
      ArrayType* array_type = element_type->AsArrayOrDie();
      Type* array_element_type = array_type->element_type();
      for (uint32_t i = 0; i < array_type->size(); i++) {
        XLS_ASSIGN_OR_RETURN(
            buffer,
            PackValueHelper(
                builder->CreateExtractValue(element, {i}), array_element_type,
                buffer, bit_offset + i * array_element_type->GetFlatBitCount(),
                builder));
      }
      return buffer;
    }
    case TypeKind::kTuple: {
      // Reverse tuple packing order to match native layout.
      TupleType* tuple_type = element_type->AsTupleOrDie();
      for (int64_t i = tuple_type->size() - 1; i >= 0; i--) {
        XLS_ASSIGN_OR_RETURN(
            buffer, PackValueHelper(builder->CreateExtractValue(
                                        element, {static_cast<uint32_t>(i)}),
                                    tuple_type->element_type(i), buffer,
                                    bit_offset, builder));
        bit_offset += tuple_type->element_type(i)->GetFlatBitCount();
      }
      return buffer;
    }
    case TypeKind::kToken: {
      // Tokens are zero-bit constructs, so there's nothing to do!
      return buffer;
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unhandled element kind: ", TypeKindToString(element_type->kind())));
  }
}

// Packs the given `value` in LLVM native data layout for XLS type
// `xls_type` into a flat bit vector and returns it.
absl::StatusOr<llvm::Value*> PackValue(llvm::Value* value, Type* xls_type,
                                       const LlvmTypeConverter& type_converter,
                                       llvm::IRBuilder<>* builder) {
  llvm::Value* packed_buffer = llvm::ConstantInt::get(
      type_converter.ConvertToPackedLlvmType(xls_type), 0);
  return PackValueHelper(value, xls_type, packed_buffer, 0, builder);
}

// Unpacks the packed bit vector representation in `packed_value` into a
// LLVM native data layout.
absl::StatusOr<llvm::Value*> UnpackValue(
    Type* param_type, llvm::Value* packed_value,
    const LlvmTypeConverter& type_converter, llvm::IRBuilder<>* builder) {
  switch (param_type->kind()) {
    case TypeKind::kBits:
      return builder->CreateZExt(
          builder->CreateTrunc(
              packed_value, type_converter.ConvertToPackedLlvmType(param_type)),
          type_converter.ConvertToLlvmType(param_type));
    case TypeKind::kArray: {
      // Create an empty array and plop in every element.
      ArrayType* array_type = param_type->AsArrayOrDie();
      Type* element_type = array_type->element_type();

      llvm::Value* array = IrBuilderVisitor::CreateTypedZeroValue(
          type_converter.ConvertToLlvmType(array_type));
      for (uint32_t i = 0; i < array_type->size(); i++) {
        XLS_ASSIGN_OR_RETURN(
            llvm::Value * element,
            UnpackValue(element_type, packed_value, type_converter, builder));
        array = builder->CreateInsertValue(array, element, {i});
        packed_value =
            builder->CreateLShr(packed_value, element_type->GetFlatBitCount());
      }
      return array;
    }
    case TypeKind::kTuple: {
      // Create an empty tuple and plop in every element.
      TupleType* tuple_type = param_type->AsTupleOrDie();
      llvm::Value* tuple = IrBuilderVisitor::CreateTypedZeroValue(
          type_converter.ConvertToLlvmType(tuple_type));
      for (int32_t i = tuple_type->size() - 1; i >= 0; i--) {
        // Tuple elements are stored MSB -> LSB, so we need to extract in
        // reverse order to match native layout.
        Type* element_type = tuple_type->element_type(i);
        XLS_ASSIGN_OR_RETURN(
            llvm::Value * element,
            UnpackValue(element_type, packed_value, type_converter, builder));
        tuple = builder->CreateInsertValue(tuple, element,
                                           {static_cast<uint32_t>(i)});
        packed_value =
            builder->CreateLShr(packed_value, element_type->GetFlatBitCount());
      }
      return tuple;
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unhandled type kind: ", TypeKindToString(param_type->kind())));
  }
}

// Load the `index`-th packed argument from the given arg array. The argument is
// unpacked into the LLVM native data layout and returned.
absl::StatusOr<llvm::Value*> LoadAndUnpackArgument(
    int64_t arg_index, Type* xls_type, llvm::Value* arg_array,
    int64_t arg_array_size, const LlvmTypeConverter& type_converter,
    llvm::IRBuilder<>* builder) {
  if (xls_type->GetFlatBitCount() == 0) {
    // Create an empty structure, etc.
    return IrBuilderVisitor::CreateTypedZeroValue(
        type_converter.ConvertToLlvmType(xls_type));
  }

  llvm::Type* packed_type = type_converter.ConvertToPackedLlvmType(xls_type);
  llvm::Value* packed_value = IrBuilderVisitor::LoadFromPointerArray(
      arg_index, packed_type, arg_array, arg_array_size, builder);

  // Now populate an Value of Param's type with the packed buffer contents.
  XLS_ASSIGN_OR_RETURN(
      llvm::Value * unpacked_param,
      UnpackValue(xls_type, packed_value, type_converter, builder));
  return unpacked_param;
}

// Return the LLVM function type corresponding to the given XLS function. The
// arguments and return value are described in BuildFunction.
llvm::FunctionType* GetFunctionType(Function* xls_function, OrcJit& jit) {
  llvm::LLVMContext& context = *jit.GetContext();

  std::vector<llvm::Type*> param_types;
  for (Param* param : xls_function->params()) {
    param_types.push_back(
        jit.GetTypeConverter().ConvertToLlvmType(param->GetType()));
  }

  llvm::Type* ptr_type = llvm::PointerType::get(context, 0);

  // After the XLS function parameters are:
  //   events pointer, user data, jit runtime
  param_types.push_back(ptr_type);
  param_types.push_back(ptr_type);
  param_types.push_back(ptr_type);

  llvm::Type* llvm_return_type = jit.GetTypeConverter().ConvertToLlvmType(
      xls_function->return_value()->GetType());

  return llvm::FunctionType::get(
      llvm_return_type,
      llvm::ArrayRef<llvm::Type*>(param_types.data(), param_types.size()),
      /*isVarArg=*/false);
}

// Return the LLVM function type corresponding to the wrapper functions (around
// functions built by BuildFunction) which take inputs and outputs as pointers
// to buffers.
llvm::FunctionType* GetWrapperFunctionType(int64_t param_count,
                                           llvm::Type* return_value_type,
                                           OrcJit& jit) {
  llvm::LLVMContext& context = *jit.GetContext();

  // The parameters are:
  //   (input arg ptrs, return value ptr, events, user data, JIT runtime)
  // All are opaque pointer types in llvm IR.
  llvm::Type* ptr_type = llvm::PointerType::get(context, 0);
  std::vector<llvm::Type*> param_types(5, ptr_type);

  return llvm::FunctionType::get(
      llvm::Type::getVoidTy(context),
      llvm::ArrayRef<llvm::Type*>(param_types.data(), param_types.size()),
      /*isVarArg=*/false);
}

}  // namespace

absl::StatusOr<llvm::Function*> BuildFunction(Function* xls_function,
                                              llvm::Module* module,
                                              OrcJit& jit) {
  llvm::FunctionType* function_type = GetFunctionType(xls_function, jit);
  llvm::Function* llvm_function = llvm::cast<llvm::Function>(
      module->getOrInsertFunction(xls_function->qualified_name(), function_type)
          .getCallee());

  // Give the function parameters meaningful names.
  int64_t argc = 0;
  for (Param* param : xls_function->params()) {
    llvm_function->getArg(argc++)->setName(param->GetName());
  }
  llvm_function->getArg(argc++)->setName("__events");
  llvm_function->getArg(argc++)->setName("__user_data");
  llvm_function->getArg(argc++)->setName("__jit_runtime");

  FunctionBuilderVisitor visitor(
      llvm_function, xls_function, &jit.GetTypeConverter(),
      [&](Function* f) { return BuildFunction(f, module, jit); });
  XLS_RETURN_IF_ERROR(xls_function->Accept(&visitor));
  XLS_RETURN_IF_ERROR(visitor.Finalize());

  XLS_VLOG(3) << "BuildFunction(" << xls_function->name() << ")";
  XLS_VLOG_LINES(3, DumpLlvmObjectToString(*llvm_function));

  return llvm_function;
}

absl::StatusOr<std::unique_ptr<FunctionJit>> FunctionJit::Create(
    Function* xls_function, int64_t opt_level) {
  return CreateInternal(xls_function, opt_level, /*emit_object_code=*/false);
}

absl::StatusOr<std::vector<char>> FunctionJit::CreateObjectFile(
    Function* xls_function, int64_t opt_level) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<FunctionJit> jit,
      CreateInternal(xls_function, opt_level, /*emit_object_code=*/true));
  return jit->orc_jit_->GetObjectCode();
}

absl::StatusOr<std::unique_ptr<FunctionJit>> FunctionJit::CreateInternal(
    Function* xls_function, int64_t opt_level, bool emit_object_code) {
  auto jit = absl::WrapUnique(new FunctionJit(xls_function));
  XLS_ASSIGN_OR_RETURN(jit->orc_jit_,
                       OrcJit::Create(opt_level, emit_object_code));

  jit->ir_runtime_ = std::make_unique<JitRuntime>(
      jit->orc_jit_->GetDataLayout(), &jit->orc_jit_->GetTypeConverter());
  std::unique_ptr<llvm::Module> module =
      jit->GetOrcJit().NewModule(xls_function->name());
  XLS_ASSIGN_OR_RETURN(
      llvm::Function * llvm_function,
      BuildFunction(xls_function, module.get(), jit->GetOrcJit()));

  // Set the argument sizes.
  for (const Param* param : xls_function->params()) {
    jit->arg_type_bytes_.push_back(
        jit->orc_jit_->GetTypeConverter().GetTypeByteSize(param->GetType()));
  }

  XLS_ASSIGN_OR_RETURN(llvm::Function * wrapper_function,
                       jit->BuildWrapper(llvm_function));
  std::string function_name = wrapper_function->getName().str();
  XLS_ASSIGN_OR_RETURN(llvm::Function * packed_wrapper_function,
                       jit->BuildPackedWrapper(llvm_function));
  std::string packed_wrapper_name = packed_wrapper_function->getName().str();

  XLS_VLOG(3) << "Module for " << xls_function->name() << ":";
  XLS_VLOG_LINES(3, DumpLlvmModuleToString(*module));

  XLS_RETURN_IF_ERROR(jit->orc_jit_->CompileModule(std::move(module)));

  XLS_ASSIGN_OR_RETURN(auto fn_address,
                       jit->orc_jit_->LoadSymbol(function_name));
  jit->invoker_ = absl::bit_cast<JitFunctionType>(fn_address);

  XLS_ASSIGN_OR_RETURN(fn_address,
                       jit->orc_jit_->LoadSymbol(packed_wrapper_name));
  jit->packed_invoker_ = absl::bit_cast<PackedJitFunctionType>(fn_address);
  return jit;
}

FunctionJit::FunctionJit(Function* xls_function)
    : xls_function_(xls_function), invoker_(nullptr) {}

absl::StatusOr<InterpreterResult<Value>> FunctionJit::Run(
    absl::Span<const Value> args) {
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
        orc_jit_->GetTypeConverter().GetTypeByteSize(param->GetType())));
    arg_buffers.push_back(unique_arg_buffers.back().get());
    param_types.push_back(param->GetType());
  }

  XLS_RETURN_IF_ERROR(
      ir_runtime_->PackArgs(args, param_types, absl::MakeSpan(arg_buffers)));

  InterpreterEvents events;

  absl::InlinedVector<uint8_t, 16> result_buffer(return_type_bytes_);
  invoker_(arg_buffers.data(), result_buffer.data(), &events,
           /*user_data=*/nullptr, runtime());

  Value result = ir_runtime_->UnpackBuffer(
      result_buffer.data(), xls_function_->return_value()->GetType());

  return InterpreterResult<Value>{std::move(result), std::move(events)};
}

absl::StatusOr<InterpreterResult<Value>> FunctionJit::Run(
    const absl::flat_hash_map<std::string, Value>& kwargs) {
  XLS_ASSIGN_OR_RETURN(std::vector<Value> positional_args,
                       KeywordArgsToPositional(*xls_function_, kwargs));
  return Run(positional_args);
}

absl::Status FunctionJit::RunWithViews(absl::Span<uint8_t*> args,
                                       absl::Span<uint8_t> result_buffer) {
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

  InterpreterEvents events;

  invoker_(args.data(), result_buffer.data(), &events, /*user_data=*/nullptr,
           runtime());

  return InterpreterEventsToStatus(events);
}

absl::StatusOr<llvm::Function*> FunctionJit::BuildWrapper(
    llvm::Function* callee) {
  llvm::Module* module = callee->getParent();
  llvm::LLVMContext* bare_context = orc_jit_->GetContext();

  Type* xls_return_type = xls_function_->return_value()->GetType();
  llvm::Type* llvm_return_type =
      orc_jit_->GetTypeConverter().ConvertToLlvmType(xls_return_type);
  llvm::FunctionType* function_type = GetWrapperFunctionType(
      xls_function_->params().size(), llvm_return_type, GetOrcJit());

  std::string function_name = absl::StrFormat("%s", xls_function_->name());
  llvm::Function* wrapper_function = llvm::cast<llvm::Function>(
      module->getOrInsertFunction(function_name, function_type).getCallee());

  // Give the arguments meaningful names so debugging the LLVM IR is easier.
  int64_t argc = 0;
  wrapper_function->getArg(argc++)->setName("inputs");
  wrapper_function->getArg(argc++)->setName("output");
  wrapper_function->getArg(argc++)->setName("interpreter_events");
  wrapper_function->getArg(argc++)->setName("user_data");
  wrapper_function->getArg(argc++)->setName("jit_runtime");

  auto basic_block =
      llvm::BasicBlock::Create(*bare_context, "entry", wrapper_function,
                               /*InsertBefore=*/nullptr);
  llvm::IRBuilder<> builder(basic_block);

  // Read in the arguments and add them to the list of arguments to pass to the
  // wrapped function.
  llvm::Value* arg_array = wrapper_function->getArg(0);
  std::vector<llvm::Value*> args;
  for (int64_t i = 0; i < xls_function_->params().size(); ++i) {
    Param* param = xls_function_->param(i);
    args.push_back(IrBuilderVisitor::LoadFromPointerArray(
        i, type_converter()->ConvertToLlvmType(param->GetType()), arg_array,
        xls_function_->params().size(), &builder));
  }

  // Pass through the final three arguments:
  //   interpreter events, user data, JIT runtime pointer
  args.push_back(wrapper_function->getArg(wrapper_function->arg_size() - 3));
  args.push_back(wrapper_function->getArg(wrapper_function->arg_size() - 2));
  args.push_back(wrapper_function->getArg(wrapper_function->arg_size() - 1));

  llvm::Value* result = builder.CreateCall(callee, args);

  return_type_bytes_ =
      orc_jit_->GetTypeConverter().GetTypeByteSize(xls_return_type);
  IrBuilderVisitor::UnpoisonBuffer(wrapper_function->getArg(1),
                                   return_type_bytes_, &builder);

  builder.CreateStore(result, wrapper_function->getArg(1));
  builder.CreateRetVoid();

  return wrapper_function;
}

absl::StatusOr<llvm::Function*> FunctionJit::BuildPackedWrapper(
    llvm::Function* callee) {
  llvm::Module* module = callee->getParent();
  llvm::LLVMContext* bare_context = orc_jit_->GetContext();

  Type* xls_return_type = xls_function_->return_value()->GetType();
  llvm::FunctionType* function_type = GetWrapperFunctionType(
      xls_function_->params().size(),
      type_converter()->ConvertToPackedLlvmType(xls_return_type), GetOrcJit());

  std::string function_name =
      absl::StrFormat("%s_packed", xls_function_->name());
  llvm::Function* wrapper_function = llvm::cast<llvm::Function>(
      module->getOrInsertFunction(function_name, function_type).getCallee());

  // Give the arguments meaningful names so debugging the LLVM IR is easier.
  int64_t argc = 0;
  wrapper_function->getArg(argc++)->setName("packed_inputs");
  wrapper_function->getArg(argc++)->setName("packed_output");
  wrapper_function->getArg(argc++)->setName("interpreter_events");
  wrapper_function->getArg(argc++)->setName("user_data");
  wrapper_function->getArg(argc++)->setName("jit_runtime");

  auto basic_block =
      llvm::BasicBlock::Create(*bare_context, "entry", wrapper_function,
                               /*InsertBefore=*/nullptr);

  llvm::IRBuilder<> builder(basic_block);

  // First load and unpack the arguments then store them in LLVM native data
  // layout. These unpacked values are pointed to by an array of pointers passed
  // on to the wrapped function.
  std::vector<llvm::Value*> args;
  for (int64_t i = 0; i < xls_function_->params().size(); ++i) {
    Param* param = xls_function_->param(i);

    XLS_ASSIGN_OR_RETURN(
        llvm::Value * unpacked_arg,
        LoadAndUnpackArgument(i, param->GetType(), wrapper_function->getArg(0),
                              xls_function_->params().size(),
                              orc_jit_->GetTypeConverter(), &builder));
    args.push_back(unpacked_arg);
  }

  // Pass through the final three arguments:
  //   interpreter events, user data, JIT runtime pointer
  args.push_back(wrapper_function->getArg(wrapper_function->arg_size() - 3));
  args.push_back(wrapper_function->getArg(wrapper_function->arg_size() - 2));
  args.push_back(wrapper_function->getArg(wrapper_function->arg_size() - 1));

  llvm::Value* return_value = builder.CreateCall(callee, args);

  // After returning, pack the value into the return value buffer.
  if (xls_return_type->GetFlatBitCount() != 0) {
    // Declare the return argument as an iX, and pack the actual data as such
    // an integer.
    XLS_ASSIGN_OR_RETURN(
        llvm::Value * packed_return,
        PackValue(return_value, xls_return_type, *type_converter(), &builder));
    llvm::Value* output_arg =
        wrapper_function->getArg(wrapper_function->arg_size() - 4);
    builder.CreateStore(packed_return, output_arg);

    IrBuilderVisitor::UnpoisonBuffer(output_arg, return_type_bytes_, &builder);
  }

  builder.CreateRetVoid();

  return wrapper_function;
}

absl::StatusOr<InterpreterResult<Value>> CreateAndRun(
    Function* xls_function, absl::Span<const Value> args) {
  // No proc support from Python yet.
  XLS_ASSIGN_OR_RETURN(auto jit, FunctionJit::Create(xls_function));
  XLS_ASSIGN_OR_RETURN(auto result, jit->Run(args));
  return result;
}

}  // namespace xls
