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

#include "absl/flags/flag.h"
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
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/keyword_args.h"
#include "xls/ir/proc.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_helpers.h"
#include "xls/jit/function_builder_visitor.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/llvm_type_converter.h"

namespace xls {
namespace {

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
        element = builder->CreateZExt(element, buffer->getType());
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
                                       llvm::IRBuilder<>* builder) {
  llvm::Value* packed_buffer = llvm::ConstantInt::get(
      llvm::IntegerType::get(builder->getContext(),
                             xls_type->GetFlatBitCount()),
      0);
  return PackValueHelper(value, xls_type, packed_buffer, 0, builder);
}

// Unpacks the packed bit vector representation in `packed_value` into a
// LLVM native data layout.
absl::StatusOr<llvm::Value*> UnpackValue(
    Type* param_type, llvm::Value* packed_value,
    const LlvmTypeConverter& type_converter, llvm::IRBuilder<>* builder) {
  llvm::LLVMContext& context = builder->getContext();
  switch (param_type->kind()) {
    case TypeKind::kBits:
      return builder->CreateTrunc(
          packed_value,
          llvm::IntegerType::get(context, param_type->GetFlatBitCount()));
    case TypeKind::kArray: {
      // Create an empty array and plop in every element.
      ArrayType* array_type = param_type->AsArrayOrDie();
      Type* element_type = array_type->element_type();

      llvm::Value* array = FunctionBuilderVisitor::CreateTypedZeroValue(
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
      llvm::Value* tuple = FunctionBuilderVisitor::CreateTypedZeroValue(
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
    return FunctionBuilderVisitor::CreateTypedZeroValue(
        type_converter.ConvertToLlvmType(xls_type));
  }

  llvm::LLVMContext& context = builder->getContext();
  llvm::Type* packed_type =
      llvm::IntegerType::get(context, xls_type->GetFlatBitCount());
  llvm::Value* packed_value = FunctionBuilderVisitor::LoadFromPointerArray(
      arg_index, packed_type, arg_array, arg_array_size, builder);

  // Now populate an Value of Param's type with the packed buffer contents.
  XLS_ASSIGN_OR_RETURN(
      llvm::Value * unpacked_param,
      UnpackValue(xls_type, packed_value, type_converter, builder));
  return unpacked_param;
}

template <typename T>
std::string DumpLlvmToString(const T& llvm_object) {
  std::string buffer;
  llvm::raw_string_ostream ostream(buffer);
  llvm_object.print(ostream);
  ostream.flush();
  return buffer;
}

}  // namespace

absl::StatusOr<std::unique_ptr<IrJit>> IrJit::Create(Function* xls_function,
                                                     int64_t opt_level) {
  auto jit = absl::WrapUnique(new IrJit(xls_function));
  XLS_RETURN_IF_ERROR(jit->Init(opt_level));
  auto visit_fn = [&jit](llvm::Module* module, llvm::Function* llvm_function) {
    return FunctionBuilderVisitor::Visit(
        module, llvm_function, jit->xls_function_, jit->type_converter(),
        /*is_top=*/true);
  };
  XLS_RETURN_IF_ERROR(jit->Compile(visit_fn));
  return jit;
}

absl::StatusOr<std::unique_ptr<IrJit>> IrJit::CreateProc(
    Proc* proc, JitChannelQueueManager* queue_mgr,
    ProcBuilderVisitor::RecvFnT recv_fn, ProcBuilderVisitor::SendFnT send_fn,
    int64_t opt_level) {
  auto jit = absl::WrapUnique(new IrJit(proc));
  XLS_RETURN_IF_ERROR(jit->Init(opt_level));
  auto visit_fn = [&jit, queue_mgr, recv_fn, send_fn](
                      llvm::Module* module, llvm::Function* llvm_function) {
    return ProcBuilderVisitor::Visit(
        module, llvm_function, jit->xls_function_, jit->type_converter(),
        /*is_top=*/true, queue_mgr, recv_fn, send_fn);
  };
  XLS_RETURN_IF_ERROR(jit->Compile(visit_fn));
  return jit;
}

absl::StatusOr<std::vector<char>> IrJit::CreateObjectFile(
    Function* xls_function, int64_t opt_level) {
  auto jit = absl::WrapUnique(new IrJit(xls_function));
  XLS_RETURN_IF_ERROR(jit->Init(opt_level, /*emit_object_code=*/true));
  auto visit_fn = [&jit](llvm::Module* module, llvm::Function* llvm_function) {
    return FunctionBuilderVisitor::Visit(
        module, llvm_function, jit->xls_function_, jit->type_converter(),
        /*is_top=*/true);
  };
  XLS_RETURN_IF_ERROR(jit->Compile(visit_fn));
  return jit->orc_jit_->GetObjectCode();
}

absl::Status IrJit::Compile(VisitFn visit_fn) {
  std::unique_ptr<llvm::Module> module = orc_jit_->NewModule("the_module");
  XLS_ASSIGN_OR_RETURN(llvm::Function * llvm_function,
                       BuildFunction(visit_fn, module.get()));
  std::string function_name = llvm_function->getName().str();
  XLS_ASSIGN_OR_RETURN(
      llvm::Function * packed_wrapper_function,
      BuildPackedWrapper(visit_fn, llvm_function, module.get()));
  std::string packed_wrapper_name = packed_wrapper_function->getName().str();

  XLS_RETURN_IF_ERROR(orc_jit_->CompileModule(std::move(module)));

  XLS_ASSIGN_OR_RETURN(auto fn_address, orc_jit_->LoadSymbol(function_name));
  invoker_ = absl::bit_cast<JitFunctionType>(fn_address);

  XLS_ASSIGN_OR_RETURN(fn_address, orc_jit_->LoadSymbol(packed_wrapper_name));
  packed_invoker_ = absl::bit_cast<PackedJitFunctionType>(fn_address);

  return absl::OkStatus();
}

IrJit::IrJit(FunctionBase* xls_function)
    : xls_function_(xls_function), invoker_(nullptr) {}

absl::Status IrJit::Init(int64_t opt_level, bool emit_object_code) {
  XLS_ASSIGN_OR_RETURN(orc_jit_, OrcJit::Create(opt_level, emit_object_code));

  ir_runtime_ = std::make_unique<JitRuntime>(orc_jit_->GetDataLayout(),
                                             &orc_jit_->GetTypeConverter());

  return absl::OkStatus();
}

absl::StatusOr<llvm::Function*> IrJit::BuildFunction(VisitFn visit_fn,
                                                     llvm::Module* module) {
  llvm::LLVMContext* bare_context = orc_jit_->GetContext();

  // To return values > 64b in size, we need to copy them into a result buffer,
  // instead of returning a fixed-size result element.
  // To do this, we need to construct the function type, adding a result buffer
  // arg (and setting the result type to void) and then storing the computation
  // result therein.
  std::vector<llvm::Type*> param_types;
  // Represent the input args as char/i8 pointers to their data.
  param_types.push_back(llvm::PointerType::get(
      llvm::ArrayType::get(llvm::Type::getInt8PtrTy(*bare_context),
                           xls_function_->params().size()),
      /*AddressSpace=*/0));

  for (const Param* param : xls_function_->params()) {
    arg_type_bytes_.push_back(
        orc_jit_->GetTypeConverter().GetTypeByteSize(param->GetType()));
  }

  // Pass the last param as a pointer to the actual return type.
  Type* return_type =
      FunctionBuilderVisitor::GetEffectiveReturnValue(xls_function_)->GetType();
  llvm::Type* llvm_return_type =
      orc_jit_->GetTypeConverter().ConvertToLlvmType(return_type);
  param_types.push_back(
      llvm::PointerType::get(llvm_return_type, /*AddressSpace=*/0));

  // Treat void pointers as int64_t values at the LLVM IR level.
  // Using an actual pointer type triggers LLVM asserts when compiling
  // in debug mode.
  // TODO(amfv): 2021-04-05 Figure out why and fix void pointer handling.
  llvm::Type* void_ptr_type = llvm::Type::getInt64Ty(*bare_context);

  // interpreter events argument
  param_types.push_back(void_ptr_type);
  // user data argument
  param_types.push_back(void_ptr_type);
  // JIT runtime argument
  param_types.push_back(void_ptr_type);

  llvm::FunctionType* function_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(*bare_context),
      llvm::ArrayRef<llvm::Type*>(param_types.data(), param_types.size()),
      /*isVarArg=*/false);

  std::string function_name = absl::StrFormat("%s", xls_function_->name());
  llvm::Function* llvm_function = llvm::cast<llvm::Function>(
      module->getOrInsertFunction(function_name, function_type).getCallee());
  return_type_bytes_ =
      orc_jit_->GetTypeConverter().GetTypeByteSize(return_type);
  XLS_RETURN_IF_ERROR(visit_fn(module, llvm_function));

  XLS_VLOG(3) << absl::StrFormat("LLVM function for %s:",
                                 xls_function_->name());
  XLS_VLOG(3) << DumpLlvmToString(*llvm_function);

  return llvm_function;
}

absl::StatusOr<InterpreterResult<Value>> IrJit::Run(
    absl::Span<const Value> args, void* user_data) {
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
  invoker_(arg_buffers.data(), result_buffer.data(), &events, user_data,
           runtime());

  Value result = ir_runtime_->UnpackBuffer(
      result_buffer.data(),
      FunctionBuilderVisitor::GetEffectiveReturnValue(xls_function_)
          ->GetType());

  return InterpreterResult<Value>{std::move(result), std::move(events)};
}

absl::StatusOr<InterpreterResult<Value>> IrJit::Run(
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

  InterpreterEvents events;

  invoker_(args.data(), result_buffer.data(), &events, user_data, runtime());

  return InterpreterEventsToStatus(events);
}

absl::StatusOr<llvm::Function*> IrJit::BuildPackedWrapper(
    VisitFn visit_fn, llvm::Function* subfunction, llvm::Module* module) {
  llvm::LLVMContext* bare_context = orc_jit_->GetContext();
  llvm::Type* i8_type = llvm::Type::getInt8Ty(*bare_context);

  // Create arg packing/unpacking buffers as in BuildFunction().
  std::vector<llvm::Type*> param_types;
  llvm::FunctionType* function_type;
  // Represent the input args as char/i8 pointers to their data.
  param_types.push_back(llvm::PointerType::get(
      llvm::ArrayType::get(llvm::PointerType::get(i8_type, /*AddressSpace=*/0),
                           xls_function_->params().size()),
      /*AddressSpace=*/0));

  Node* return_node =
      FunctionBuilderVisitor::GetEffectiveReturnValue(xls_function_);
  int64_t return_width = return_node->GetType()->GetFlatBitCount();
  llvm::Type* return_type =
      orc_jit_->GetTypeConverter().ConvertToLlvmType(return_node->GetType());

  if (return_width != 0) {
    // For packed operation, just pass a i8 pointer for the result.
    param_types.push_back(llvm::PointerType::get(
        llvm::IntegerType::get(*bare_context, return_width),
        /*AddressSpace=*/0));
  }
  // The final three parameters are:
  //   interpreter events, user data, JIT runtime pointer
  param_types.push_back(llvm::Type::getInt64Ty(*bare_context));
  param_types.push_back(llvm::Type::getInt64Ty(*bare_context));
  param_types.push_back(llvm::Type::getInt64Ty(*bare_context));

  function_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(*bare_context),
      llvm::ArrayRef<llvm::Type*>(param_types.data(), param_types.size()),
      /*isVarArg=*/false);

  std::string function_name =
      absl::StrFormat("%s_packed", xls_function_->name());
  llvm::Function* wrapper_function = llvm::cast<llvm::Function>(
      module->getOrInsertFunction(function_name, function_type).getCallee());

  // Give the arguments meaningful names so debugging the LLVM IR is easier.
  int64_t argc = 0;
  wrapper_function->getArg(argc++)->setName("packed_inputs");
  if (return_width != 0) {
    wrapper_function->getArg(argc++)->setName("packed_output");
  }
  wrapper_function->getArg(argc++)->setName("interpreter_events");
  wrapper_function->getArg(argc++)->setName("user_data");
  wrapper_function->getArg(argc++)->setName("jit_runtime");

  auto basic_block =
      llvm::BasicBlock::Create(*bare_context, "entry", wrapper_function,
                               /*InsertBefore=*/nullptr);

  std::vector<llvm::Value*> args;
  llvm::IRBuilder<> builder(basic_block);

  // First load and unpack the arguments then store them in LLVM native data
  // layout. These unpacked values are pointed to by an array of pointers passed
  // on to the wrapped function.
  llvm::Type* input_pointers_type = llvm::ArrayType::get(
      llvm::Type::getInt8PtrTy(*bare_context), xls_function_->params().size());
  llvm::AllocaInst* input_pointers = builder.CreateAlloca(input_pointers_type);
  input_pointers->setName("unpacked_input_buffers");
  for (int64_t i = 0; i < xls_function_->params().size(); ++i) {
    Param* param = xls_function_->param(i);

    XLS_ASSIGN_OR_RETURN(
        llvm::Value * unpacked_arg,
        LoadAndUnpackArgument(i, param->GetType(), wrapper_function->getArg(0),
                              xls_function_->params().size(),
                              orc_jit_->GetTypeConverter(), &builder));
    unpacked_arg->setName(absl::StrFormat("unpacked_%s", param->GetName()));

    llvm::AllocaInst* input_buffer =
        builder.CreateAlloca(unpacked_arg->getType());
    input_buffer->setName(
        absl::StrFormat("unpacked_%s_buffer", param->GetName()));
    builder.CreateStore(unpacked_arg, input_buffer);
    llvm::Value* input_pointer = builder.CreateBitCast(
        input_buffer, llvm::Type::getInt8PtrTy(*bare_context));
    llvm::Value* gep = builder.CreateGEP(
        input_pointers_type, input_pointers,
        {
            llvm::ConstantInt::get(llvm::Type::getInt64Ty(*bare_context), 0),
            llvm::ConstantInt::get(llvm::Type::getInt64Ty(*bare_context), i),
        });
    builder.CreateStore(input_pointer, gep);
  }
  args.push_back(input_pointers);

  // Allocate space for the unpacked return value.
  llvm::AllocaInst* return_buffer = builder.CreateAlloca(return_type);
  return_buffer->setName("unpacked_output_buffer");

  args.push_back(return_buffer);
  // Pass through the final three arguments:
  //   interpreter events, user data, JIT runtime pointer
  args.push_back(wrapper_function->getArg(wrapper_function->arg_size() - 3));
  args.push_back(wrapper_function->getArg(wrapper_function->arg_size() - 2));
  args.push_back(wrapper_function->getArg(wrapper_function->arg_size() - 1));

  builder.CreateCall(subfunction, args);

  // After returning, pack the value into the return value buffer.
  if (return_width != 0) {
    // Declare the return argument as an iX, and pack the actual data as such
    // an integer.
    llvm::Value* return_value = builder.CreateLoad(return_type, return_buffer);
    XLS_ASSIGN_OR_RETURN(
        llvm::Value * packed_return,
        PackValue(return_value, return_node->GetType(), &builder));
    llvm::Value* output_arg =
        wrapper_function->getArg(wrapper_function->arg_size() - 4);
    builder.CreateStore(packed_return, output_arg);

    FunctionBuilderVisitor::UnpoisonBuffer(output_arg, return_type_bytes_,
                                           &builder);
  }

  builder.CreateRetVoid();

  XLS_VLOG(3) << "Packed wrapper function:";
  XLS_VLOG(3) << DumpLlvmToString(*wrapper_function);

  return wrapper_function;
}

absl::StatusOr<InterpreterResult<Value>> CreateAndRun(
    Function* xls_function, absl::Span<const Value> args) {
  // No proc support from Python yet.
  XLS_ASSIGN_OR_RETURN(auto jit, IrJit::Create(xls_function));
  XLS_ASSIGN_OR_RETURN(auto result, jit->Run(args));
  return result;
}

}  // namespace xls
