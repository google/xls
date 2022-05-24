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

#include "xls/jit/proc_jit.h"

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
#include "xls/jit/function_jit.h"
#include "xls/jit/ir_builder_visitor.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/llvm_type_converter.h"

namespace xls {
namespace {

// Specialization of IrBuilderVisitor for translating Procs. Handles
// proc-specific operands such has sends and receives.
class ProcBuilderVisitor : public IrBuilderVisitor {
 public:
  ProcBuilderVisitor(llvm::Function* llvm_fn, Proc* proc,
                     LlvmTypeConverter* type_converter,
                     JitChannelQueueManager* queue_mgr,
                     ProcJit::RecvFnT recv_fn, ProcJit::SendFnT send_fn,
                     std::function<absl::StatusOr<llvm::Function*>(Function*)>
                         function_builder)
      : IrBuilderVisitor(llvm_fn, proc, type_converter, function_builder),
        queue_mgr_(queue_mgr),
        recv_fn_(recv_fn),
        send_fn_(send_fn) {}

  absl::Status HandleReceive(Receive* recv) override;
  absl::Status HandleSend(Send* send) override;

  // The first argument to the proc function is the proc state.
  absl::Status HandleParam(Param* param) override {
    if (param == proc()->GetUniqueStateParam()) {
      return StoreResult(param, llvm_fn_->getArg(0));
    }
    XLS_RET_CHECK_EQ(param, proc()->TokenParam());

    return StoreResult(param, type_converter()->GetToken());
  }

  // Builds the IR to return the next state value.
  absl::Status Finalize() {
    Node* next_state = xls_fn_->AsProcOrDie()->GetUniqueNextState();
    Type* xls_state_type = next_state->GetType();
    llvm::Type* llvm_state_type =
        type_converter()->ConvertToLlvmType(xls_state_type);
    if (llvm_state_type->isVoidTy()) {
      builder()->CreateRetVoid();
    } else {
      builder()->CreateRet(node_map().at(next_state));
    }
    return absl::OkStatus();
  }

  Proc* proc() const { return xls_fn_->AsProcOrDie(); }

 private:
  absl::StatusOr<llvm::Value*> InvokeRecvCallback(llvm::IRBuilder<>* builder,
                                                  JitChannelQueue* queue,
                                                  Receive* receive);

  absl::Status InvokeSendCallback(llvm::IRBuilder<>* builder,
                                  JitChannelQueue* queue, Send* send,
                                  Node* data);

 public:
  JitChannelQueueManager* queue_mgr_;
  ProcJit::RecvFnT recv_fn_;
  ProcJit::SendFnT send_fn_;
};

absl::StatusOr<llvm::Value*> ProcBuilderVisitor::InvokeRecvCallback(
    llvm::IRBuilder<>* builder, JitChannelQueue* queue, Receive* receive) {
  llvm::Type* void_type = llvm::Type::getVoidTy(ctx());
  llvm::Type* int64_type = llvm::Type::getInt64Ty(ctx());
  llvm::Type* int8_ptr_type = llvm::Type::getInt8PtrTy(ctx(), 0);

  // Treat void pointers as int64_t values at the LLVM IR level.
  // Using an actual pointer type triggers LLVM asserts when compiling
  // in debug mode.
  // TODO(amfv): 2021-04-05 Figure out why and fix void pointer handling.
  llvm::Type* void_ptr_type = int64_type;

  // Call the user-provided function of type ProcJit::RecvFnT to receive the
  // value.
  std::vector<llvm::Type*> params = {int64_type, int64_type, int8_ptr_type,
                                     int64_type, void_ptr_type};
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  llvm::Type* recv_type =
      type_converter()->ConvertToLlvmType(receive->GetType());
  int64_t recv_bytes = type_converter()->GetTypeByteSize(receive->GetType());
  llvm::AllocaInst* alloca = builder->CreateAlloca(recv_type);

  // Call the user-provided receive function.
  std::vector<llvm::Value*> args = {
      llvm::ConstantInt::get(int64_type, absl::bit_cast<uint64_t>(queue)),
      llvm::ConstantInt::get(int64_type, absl::bit_cast<uint64_t>(receive)),
      builder->CreatePointerCast(alloca, int8_ptr_type),
      llvm::ConstantInt::get(int64_type, recv_bytes),
      GetUserDataPtr(),
  };
  llvm::ConstantInt* fn_addr = llvm::ConstantInt::get(
      llvm::Type::getInt64Ty(ctx()), absl::bit_cast<uint64_t>(recv_fn_));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  builder->CreateCall(fn_type, fn_ptr, args);

  // Load the reveive data from the bounce buffer.
  return builder->CreateLoad(recv_type, alloca);
}

absl::Status ProcBuilderVisitor::HandleReceive(Receive* recv) {
  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue,
                       queue_mgr_->GetQueueById(recv->channel_id()));
  llvm::Value* result;
  if (recv->predicate().has_value()) {
    // First, declare the join block (so the case blocks can refer to it).
    llvm::BasicBlock* join_block = llvm::BasicBlock::Create(
        ctx(), absl::StrCat(recv->GetName(), "_join"), llvm_fn());

    // Create a block/branch for the true predicate case.
    llvm::BasicBlock* true_block = llvm::BasicBlock::Create(
        ctx(), absl::StrCat(recv->GetName(), "_true"), llvm_fn(), join_block);
    llvm::IRBuilder<> true_builder(true_block);
    XLS_ASSIGN_OR_RETURN(llvm::Value * true_result,
                         InvokeRecvCallback(&true_builder, queue, recv));
    true_builder.CreateBr(join_block);

    // And the same for a false predicate - this will return an empty/zero
    // value. Creating an empty struct emits ops, so it needs a builder.
    llvm::BasicBlock* false_block = llvm::BasicBlock::Create(
        ctx(), absl::StrCat(recv->GetName(), "_false"), llvm_fn(), join_block);
    llvm::IRBuilder<> false_builder(false_block);
    llvm::Type* result_type =
        type_converter()->ConvertToLlvmType(recv->GetType());
    llvm::Value* false_result = CreateTypedZeroValue(result_type);
    false_builder.CreateBr(join_block);

    // Next, create a branch op w/the original builder,
    builder()->CreateCondBr(node_map().at(recv->predicate().value()),
                            true_block, false_block);

    // then join the two branches back together.
    auto join_builder = std::make_unique<llvm::IRBuilder<>>(join_block);

    llvm::PHINode* phi =
        join_builder->CreatePHI(result_type, /*NumReservedValues=*/2);
    phi->addIncoming(true_result, true_block);
    phi->addIncoming(false_result, false_block);
    result =
        join_builder->CreateInsertValue(phi, type_converter()->GetToken(), {0});

    // Finally, set this's IRBuilder to be the output block's (since that's
    // where the Function continues).
    set_builder(std::move(join_builder));
  } else {
    XLS_ASSIGN_OR_RETURN(llvm::Value * invoke,
                         InvokeRecvCallback(builder(), queue, recv));
    result =
        builder()->CreateInsertValue(invoke, type_converter()->GetToken(), {0});
  }
  return StoreResult(recv, result);
}

absl::Status ProcBuilderVisitor::InvokeSendCallback(llvm::IRBuilder<>* builder,
                                                    JitChannelQueue* queue,
                                                    Send* send, Node* data) {
  llvm::Type* void_type = llvm::Type::getVoidTy(ctx());
  llvm::Type* int64_type = llvm::Type::getInt64Ty(ctx());
  llvm::Type* int8_ptr_type = llvm::Type::getInt8PtrTy(ctx(), 0);

  // Treat void pointers as int64_t values at the LLVM IR level.
  // Using an actual pointer type triggers LLVM asserts when compiling
  // in debug mode.
  // TODO(amfv): 2021-04-05 Figure out why and fix void pointer handling.
  llvm::Type* void_ptr_type = int64_type;

  // We do the same for sending/enqueuing as we do for receiving/dequeueing
  // above (set up and call an external function).
  std::vector<llvm::Type*> params = {
      int64_type, int64_type, int8_ptr_type, int64_type, void_ptr_type,
  };
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  // Pack all the data to be sent into a contiguous tuple (we'll need to pack
  // the data anyway, since llvm::Values don't automatically correspond to
  // pointer-referencable storage; that's what allocas are for).
  std::vector<Type*> tuple_elems;
  tuple_elems.push_back(data->GetType());
  TupleType tuple_type(tuple_elems);
  llvm::Type* send_op_types = type_converter()->ConvertToLlvmType(&tuple_type);
  int64_t send_type_size = type_converter()->GetTypeByteSize(&tuple_type);
  llvm::Value* tuple = CreateTypedZeroValue(send_op_types);
  tuple = builder->CreateInsertValue(tuple, node_map().at(data), {0u});
  llvm::AllocaInst* alloca = builder->CreateAlloca(send_op_types);
  builder->CreateStore(tuple, alloca);

  std::vector<llvm::Value*> args = {
      llvm::ConstantInt::get(int64_type, absl::bit_cast<uint64_t>(queue)),
      llvm::ConstantInt::get(int64_type, absl::bit_cast<uint64_t>(send)),
      builder->CreatePointerCast(alloca, int8_ptr_type),
      llvm::ConstantInt::get(int64_type, send_type_size),
      GetUserDataPtr(),
  };

  llvm::ConstantInt* fn_addr = llvm::ConstantInt::get(
      llvm::Type::getInt64Ty(ctx()), absl::bit_cast<uint64_t>(send_fn_));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  builder->CreateCall(fn_type, fn_ptr, args);
  return absl::OkStatus();
}

absl::Status ProcBuilderVisitor::HandleSend(Send* send) {
  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue,
                       queue_mgr_->GetQueueById(send->channel_id()));
  if (send->predicate().has_value()) {
    // First, declare the join block (so the case blocks can refer to it).
    llvm::BasicBlock* join_block = llvm::BasicBlock::Create(
        ctx(), absl::StrCat(send->GetName(), "_join"), llvm_fn());

    llvm::BasicBlock* true_block = llvm::BasicBlock::Create(
        ctx(), absl::StrCat(send->GetName(), "_true"), llvm_fn(), join_block);
    llvm::IRBuilder<> true_builder(true_block);
    XLS_RETURN_IF_ERROR(
        InvokeSendCallback(&true_builder, queue, send, send->data()));
    true_builder.CreateBr(join_block);

    llvm::BasicBlock* false_block = llvm::BasicBlock::Create(
        ctx(), absl::StrCat(send->GetName(), "_false"), llvm_fn(), join_block);
    llvm::IRBuilder<> false_builder(false_block);
    false_builder.CreateBr(join_block);

    builder()->CreateCondBr(node_map().at(send->predicate().value()),
                            true_block, false_block);

    auto join_builder = std::make_unique<llvm::IRBuilder<>>(join_block);

    set_builder(std::move(join_builder));

  } else {
    XLS_RETURN_IF_ERROR(
        InvokeSendCallback(builder(), queue, send, send->data()));
  }
  return StoreResult(send, type_converter()->GetToken());
}

// Builds and returns and LLVM function which executes a single tick of the
// given proc. The LLVM function has the following signature:
//
//   stateT f(stateT state,
//            void* events, void* user_data, void* jit_runtime)
//
//   events      : a pointer to an InterpreterEvents object
//   user_data   : opaque pointer passed on to send and receive functions.
//   jit_runtime : a pointer to a JitRuntime
//
// The function returns the next state value.
absl::StatusOr<llvm::Function*> BuildProcFunction(
    Proc* proc, llvm::Module* module, JitChannelQueueManager* queue_mgr,
    ProcJit::RecvFnT recv_fn, ProcJit::SendFnT send_fn, OrcJit& jit) {
  llvm::LLVMContext& context = *jit.GetContext();

  Type* xls_state_type = proc->GetUniqueStateType();
  llvm::Type* llvm_state_type =
      jit.GetTypeConverter().ConvertToLlvmType(xls_state_type);

  std::vector<llvm::Type*> param_types;
  param_types.push_back(
      jit.GetTypeConverter().ConvertToLlvmType(xls_state_type));

  // Treat void pointers as int64_t values at the LLVM IR level.
  // Using an actual pointer type triggers LLVM asserts when compiling
  // in debug mode.
  // TODO(amfv): 2021-04-05 Figure out why and fix void pointer handling.
  llvm::Type* void_ptr_type = llvm::Type::getInt64Ty(context);

  // After the XLS function parameters are:
  //   events pointer, user data, jit runtime
  // All are void pointer type.
  param_types.push_back(void_ptr_type);
  param_types.push_back(void_ptr_type);
  param_types.push_back(void_ptr_type);

  llvm::FunctionType* function_type = llvm::FunctionType::get(
      llvm_state_type,
      llvm::ArrayRef<llvm::Type*>(param_types.data(), param_types.size()),
      /*isVarArg=*/false);
  llvm::Function* llvm_function = llvm::cast<llvm::Function>(
      module->getOrInsertFunction(proc->qualified_name(), function_type)
          .getCallee());

  // Give the function parameters meaningful names.
  int64_t argc = 0;
  llvm_function->getArg(argc++)->setName(
      proc->GetUniqueStateParam()->GetName());
  llvm_function->getArg(argc++)->setName("__events");
  llvm_function->getArg(argc++)->setName("__user_data");
  llvm_function->getArg(argc++)->setName("__jit_runtime");

  ProcBuilderVisitor visitor(
      llvm_function, proc, &jit.GetTypeConverter(), queue_mgr, recv_fn, send_fn,
      [&](Function* f) { return BuildFunction(f, module, jit); });
  XLS_RETURN_IF_ERROR(proc->Accept(&visitor));
  XLS_RETURN_IF_ERROR(visitor.Finalize());

  return llvm_function;
}

}  // namespace

absl::StatusOr<std::unique_ptr<ProcJit>> ProcJit::Create(
    Proc* proc, JitChannelQueueManager* queue_mgr, ProcJit::RecvFnT recv_fn,
    ProcJit::SendFnT send_fn, int64_t opt_level) {
  auto jit = absl::WrapUnique(new ProcJit(proc));
  XLS_ASSIGN_OR_RETURN(jit->orc_jit_,
                       OrcJit::Create(opt_level, /*emit_object_code=*/false));
  jit->ir_runtime_ = std::make_unique<JitRuntime>(
      jit->orc_jit_->GetDataLayout(), &jit->orc_jit_->GetTypeConverter());
  std::unique_ptr<llvm::Module> module =
      jit->GetOrcJit().NewModule(proc->name());
  XLS_ASSIGN_OR_RETURN(llvm::Function * llvm_function,
                       BuildProcFunction(proc, module.get(), queue_mgr, recv_fn,
                                         send_fn, jit->GetOrcJit()));
  XLS_ASSIGN_OR_RETURN(llvm::Function * wrapper_function,
                       jit->BuildWrapper(llvm_function));
  std::string function_name = wrapper_function->getName().str();

  XLS_VLOG(3) << "Module for " << proc->name() << ":";
  XLS_VLOG_LINES(3, DumpLlvmModuleToString(*module));

  XLS_RETURN_IF_ERROR(jit->orc_jit_->CompileModule(std::move(module)));
  XLS_ASSIGN_OR_RETURN(auto fn_address,
                       jit->orc_jit_->LoadSymbol(function_name));
  jit->invoker_ = absl::bit_cast<JitFunctionType>(fn_address);

  return jit;
}

ProcJit::ProcJit(Proc* proc) : proc_(proc), invoker_(nullptr) {}

absl::StatusOr<llvm::Function*> ProcJit::BuildWrapper(llvm::Function* callee) {
  llvm::Module* module = callee->getParent();
  llvm::LLVMContext* bare_context = orc_jit_->GetContext();

  // Gather the parameter types to build the function type. Signature:
  //
  //   void f(uint8_t*[] state, uint8_t* next_state,
  //          void* events, void* user_data, void* jit_runtime)
  std::vector<llvm::Type*> param_types;
  param_types.push_back(llvm::PointerType::get(
      llvm::ArrayType::get(llvm::Type::getInt8PtrTy(*bare_context), 1),
      /*AddressSpace=*/0));
  Type* xls_state_type = proc_->GetUniqueStateType();
  llvm::Type* llvm_state_type =
      orc_jit_->GetTypeConverter().ConvertToLlvmType(xls_state_type);
  param_types.push_back(
      llvm::PointerType::get(llvm_state_type, /*AddressSpace=*/0));

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

  llvm::Function* wrapper_function = llvm::cast<llvm::Function>(
      module->getOrInsertFunction(proc_->name(), function_type).getCallee());

  // Give the arguments meaningful names so debugging the LLVM IR is easier.
  int64_t argc = 0;
  wrapper_function->getArg(argc++)->setName("state");
  wrapper_function->getArg(argc++)->setName("next_state");
  wrapper_function->getArg(argc++)->setName("interpreter_events");
  wrapper_function->getArg(argc++)->setName("user_data");
  wrapper_function->getArg(argc++)->setName("jit_runtime");

  auto basic_block =
      llvm::BasicBlock::Create(*bare_context, "entry", wrapper_function,
                               /*InsertBefore=*/nullptr);
  llvm::IRBuilder<> builder(basic_block);

  // Read in the state value the arguments and add them to the list of arguments
  // to pass to the wrapped function.
  llvm::Value* arg_array = wrapper_function->getArg(0);
  llvm::Value* state_value = IrBuilderVisitor::LoadFromPointerArray(
      0, llvm_state_type, arg_array, 1, &builder);

  std::vector<llvm::Value*> args;
  args.push_back(state_value);

  // Pass through the final three arguments:
  //   interpreter events, user data, JIT runtime pointer
  args.push_back(wrapper_function->getArg(wrapper_function->arg_size() - 3));
  args.push_back(wrapper_function->getArg(wrapper_function->arg_size() - 2));
  args.push_back(wrapper_function->getArg(wrapper_function->arg_size() - 1));

  llvm::Value* next_state = builder.CreateCall(callee, args);

  state_type_bytes_ =
      orc_jit_->GetTypeConverter().GetTypeByteSize(xls_state_type);
  IrBuilderVisitor::UnpoisonBuffer(wrapper_function->getArg(1),
                                   state_type_bytes_, &builder);

  builder.CreateStore(next_state, wrapper_function->getArg(1));
  builder.CreateRetVoid();

  XLS_VLOG(3) << absl::StrFormat("LLVM function for %s:", proc_->name());
  XLS_VLOG(3) << DumpLlvmObjectToString(*wrapper_function);

  return wrapper_function;
}

absl::StatusOr<InterpreterResult<Value>> ProcJit::Run(const Value& state,
                                                      void* user_data) {
  Type* state_type = proc_->GetUniqueStateType();
  if (!ValueConformsToType(state, state_type)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected state argument %s to be of type %s",
                        state.ToString(), state_type->ToString()));
  }
  std::unique_ptr<uint8_t[]> arg_buffer = std::make_unique<uint8_t[]>(
      orc_jit_->GetTypeConverter().GetTypeByteSize(state_type));
  ir_runtime_->BlitValueToBuffer(
      state, state_type,
      absl::MakeSpan(arg_buffer.get(),
                     type_converter()->GetTypeByteSize(state_type)));

  InterpreterEvents events;

  std::vector<uint8_t*> arg_buffers = {arg_buffer.get()};
  absl::InlinedVector<uint8_t, 16> result_buffer(state_type_bytes_);
  invoker_(arg_buffers.data(), result_buffer.data(), &events, user_data,
           runtime());

  Value result = ir_runtime_->UnpackBuffer(result_buffer.data(), state_type);

  return InterpreterResult<Value>{std::move(result), std::move(events)};
}

}  // namespace xls
