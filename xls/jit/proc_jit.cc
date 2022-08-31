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

  // The first arguments to the proc function are the existing proc
  // state. Function signature is:
  //
  //   void f(state_0_T state_0, .., state_n_T state_n,
  //          state_0_t* next_state_0, ..., state_n_T* next_state_n,
  //          void* events, void* user_data, void* jit_runtime)
  absl::Status HandleParam(Param* param) override {
    llvm::Function* llvm_function =
        dispatch_builder()->GetInsertBlock()->getParent();

    llvm::Value* value;
    if (param == proc()->TokenParam()) {
      value = type_converter()->GetToken();
    } else {
      XLS_ASSIGN_OR_RETURN(int64_t index, proc()->GetStateParamIndex(param));
      value = llvm_function->getArg(index);
    }

    return StoreResult(param, value);
  }

  // Builds the IR to write the next state value into the appropriate buffer
  // (passed in as an argument). Function signature is:
  //
  //   void f(state_0_T state_0, .., state_n_T state_n,
  //          state_0_t* next_state_0, ..., state_n_T* next_state_n,
  //          void* events, void* user_data, void* jit_runtime)
  absl::Status Finalize() {
    llvm::Function* llvm_function =
        dispatch_builder()->GetInsertBlock()->getParent();
    for (int64_t i = 0; i < proc()->GetStateElementCount(); ++i) {
      llvm::Value* next_state_ptr =
          llvm_function->getArg(proc()->GetStateElementCount() + i);
      dispatch_builder()->CreateStore(
          node_map().at(proc()->GetNextStateElement(i)), next_state_ptr);
    }
    dispatch_builder()->CreateRetVoid();

    return absl::OkStatus();
  }

  Proc* proc() const { return xls_fn_->AsProcOrDie(); }

 private:
  absl::StatusOr<llvm::Value*> InvokeRecvCallback(llvm::IRBuilder<>* builder,
                                                  JitChannelQueue* queue,
                                                  Receive* receive,
                                                  llvm::Value* user_data);

  absl::Status InvokeSendCallback(llvm::IRBuilder<>* builder,
                                  JitChannelQueue* queue, Send* send,
                                  llvm::Value* llvm_data,
                                  llvm::Value* user_data);

 public:
  JitChannelQueueManager* queue_mgr_;
  ProcJit::RecvFnT recv_fn_;
  ProcJit::SendFnT send_fn_;
};

absl::StatusOr<llvm::Value*> ProcBuilderVisitor::InvokeRecvCallback(
    llvm::IRBuilder<>* builder, JitChannelQueue* queue, Receive* receive,
    llvm::Value* user_data) {
  llvm::Type* bool_type = llvm::Type::getInt1Ty(ctx());
  llvm::Type* int64_type = llvm::Type::getInt64Ty(ctx());
  llvm::Type* int8_ptr_type = llvm::Type::getInt8PtrTy(ctx(), 0);

  llvm::Type* ptr_type = llvm::PointerType::get(ctx(), 0);

  // Call the user-provided function of type ProcJit::RecvFnT to receive the
  // value.
  std::vector<llvm::Type*> params = {int64_type, int64_type, int8_ptr_type,
                                     int64_type, ptr_type};
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(bool_type, params, /*isVarArg=*/false);

  // recv_type is the full type of the receive.
  //   1. If blocking, then it is a tuple of (token, payload).
  //   2. If non-blocking, then it is a tuple of (token, payload, bool).
  llvm::Type* recv_type =
      type_converter()->ConvertToLlvmType(receive->GetType());
  llvm::AllocaInst* alloca = builder->CreateAlloca(recv_type);

  // recv_payload_bytes is just the size of the payload.
  //
  // As token is zero size, it can also be considered the size of the
  // token + payload.
  int64_t recv_payload_bytes =
      type_converter()->GetTypeByteSize(receive->GetPayloadType());

  // Call the user-provided receive function.
  std::vector<llvm::Value*> args = {
      llvm::ConstantInt::get(int64_type, absl::bit_cast<uint64_t>(queue)),
      llvm::ConstantInt::get(int64_type, absl::bit_cast<uint64_t>(receive)),
      builder->CreatePointerCast(alloca, int8_ptr_type),
      llvm::ConstantInt::get(int64_type, recv_payload_bytes),
      user_data,
  };
  llvm::ConstantInt* fn_addr = llvm::ConstantInt::get(
      llvm::Type::getInt64Ty(ctx()), absl::bit_cast<uint64_t>(recv_fn_));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  llvm::Value* data_valid = builder->CreateCall(fn_type, fn_ptr, args);

  // Load the receive data from the bounce buffer.
  llvm::Value* data = builder->CreateLoad(recv_type, alloca);

  if (receive->is_blocking()) {
    return data;
  }

  // For non-blocking receives, add data_valid as the last entry in the
  // return tuple.
  return builder->CreateInsertValue(data, data_valid, {2});
}

absl::Status ProcBuilderVisitor::HandleReceive(Receive* recv) {
  std::vector<std::string> operand_names = {"tkn"};
  if (recv->predicate().has_value()) {
    operand_names.push_back("predicate");
  }
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(recv, operand_names,
                                        /*include_context_args=*/true));
  llvm::IRBuilder<>& b = node_context.builder();
  llvm::Value* user_data = node_context.GetUserData();

  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue,
                       queue_mgr_->GetQueueById(recv->channel_id()));
  if (recv->predicate().has_value()) {
    llvm::Value* predicate = node_context.operand(1);

    // First, declare the join block (so the case blocks can refer to it).
    llvm::BasicBlock* join_block =
        llvm::BasicBlock::Create(ctx(), absl::StrCat(recv->GetName(), "_join"),
                                 node_context.llvm_function());

    // Create a block/branch for the true predicate case.
    llvm::BasicBlock* true_block =
        llvm::BasicBlock::Create(ctx(), absl::StrCat(recv->GetName(), "_true"),
                                 node_context.llvm_function(), join_block);
    llvm::IRBuilder<> true_builder(true_block);
    XLS_ASSIGN_OR_RETURN(
        llvm::Value * true_result,
        InvokeRecvCallback(&true_builder, queue, recv, user_data));
    true_builder.CreateBr(join_block);

    // And the same for a false predicate - this will return an empty/zero
    // value. Creating an empty struct emits ops, so it needs a builder.
    llvm::BasicBlock* false_block =
        llvm::BasicBlock::Create(ctx(), absl::StrCat(recv->GetName(), "_false"),
                                 node_context.llvm_function(), join_block);
    llvm::IRBuilder<> false_builder(false_block);
    llvm::Type* result_type =
        type_converter()->ConvertToLlvmType(recv->GetType());
    llvm::Value* false_result = CreateTypedZeroValue(result_type);
    false_builder.CreateBr(join_block);

    // Next, create a branch op w/the original builder,
    b.CreateCondBr(predicate, true_block, false_block);

    // then join the two branches back together.
    auto join_builder = std::make_unique<llvm::IRBuilder<>>(join_block);

    llvm::PHINode* phi =
        join_builder->CreatePHI(result_type, /*NumReservedValues=*/2);
    phi->addIncoming(true_result, true_block);
    phi->addIncoming(false_result, false_block);
    llvm::Value* result =
        join_builder->CreateInsertValue(phi, type_converter()->GetToken(), {0});

    return FinalizeNodeIrContext(node_context, result,
                                 /*exit_builder=*/std::move(join_builder));
  }
  XLS_ASSIGN_OR_RETURN(llvm::Value * invoke,
                       InvokeRecvCallback(&b, queue, recv, user_data));
  llvm::Value* result =
      b.CreateInsertValue(invoke, type_converter()->GetToken(), {0});

  return FinalizeNodeIrContext(node_context, result);
}

absl::Status ProcBuilderVisitor::InvokeSendCallback(llvm::IRBuilder<>* builder,
                                                    JitChannelQueue* queue,
                                                    Send* send,
                                                    llvm::Value* llvm_data,
                                                    llvm::Value* user_data) {
  llvm::Type* void_type = llvm::Type::getVoidTy(ctx());
  llvm::Type* int64_type = llvm::Type::getInt64Ty(ctx());
  llvm::Type* int8_ptr_type = llvm::Type::getInt8PtrTy(ctx(), 0);

  llvm::Type* ptr_type = llvm::PointerType::get(ctx(), 0);

  // We do the same for sending/enqueuing as we do for receiving/dequeueing
  // above (set up and call an external function).
  std::vector<llvm::Type*> params = {
      int64_type, int64_type, int8_ptr_type, int64_type, ptr_type,
  };
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  // Pack all the data to be sent into a contiguous tuple (we'll need to pack
  // the data anyway, since llvm::Values don't automatically correspond to
  // pointer-referencable storage; that's what allocas are for).
  std::vector<Type*> tuple_elems;
  tuple_elems.push_back(send->data()->GetType());
  TupleType* tuple_type = proc()->package()->GetTupleType(tuple_elems);
  llvm::Type* send_op_types = type_converter()->ConvertToLlvmType(tuple_type);
  int64_t send_type_size = type_converter()->GetTypeByteSize(tuple_type);
  llvm::Value* tuple = CreateTypedZeroValue(send_op_types);
  tuple = builder->CreateInsertValue(tuple, llvm_data, {0u});
  llvm::AllocaInst* alloca = builder->CreateAlloca(send_op_types);
  builder->CreateStore(tuple, alloca);

  std::vector<llvm::Value*> args = {
      llvm::ConstantInt::get(int64_type, absl::bit_cast<uint64_t>(queue)),
      llvm::ConstantInt::get(int64_type, absl::bit_cast<uint64_t>(send)),
      builder->CreatePointerCast(alloca, int8_ptr_type),
      llvm::ConstantInt::get(int64_type, send_type_size),
      user_data,
  };

  llvm::ConstantInt* fn_addr = llvm::ConstantInt::get(
      llvm::Type::getInt64Ty(ctx()), absl::bit_cast<uint64_t>(send_fn_));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  builder->CreateCall(fn_type, fn_ptr, args);
  return absl::OkStatus();
}

absl::Status ProcBuilderVisitor::HandleSend(Send* send) {
  std::vector<std::string> operand_names = {"tkn", "data"};
  if (send->predicate().has_value()) {
    operand_names.push_back("predicate");
  }
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(send, operand_names,
                                        /*include_context_args=*/true));
  llvm::IRBuilder<>& b = node_context.builder();
  llvm::Value* data = node_context.operand(1);
  llvm::Value* user_data = node_context.GetUserData();

  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue,
                       queue_mgr_->GetQueueById(send->channel_id()));
  if (send->predicate().has_value()) {
    llvm::Value* predicate = node_context.operand(2);

    // First, declare the join block (so the case blocks can refer to it).
    llvm::BasicBlock* join_block =
        llvm::BasicBlock::Create(ctx(), absl::StrCat(send->GetName(), "_join"),
                                 node_context.llvm_function());

    llvm::BasicBlock* true_block =
        llvm::BasicBlock::Create(ctx(), absl::StrCat(send->GetName(), "_true"),
                                 node_context.llvm_function(), join_block);
    llvm::IRBuilder<> true_builder(true_block);
    XLS_RETURN_IF_ERROR(
        InvokeSendCallback(&true_builder, queue, send, data, user_data));
    true_builder.CreateBr(join_block);

    llvm::BasicBlock* false_block =
        llvm::BasicBlock::Create(ctx(), absl::StrCat(send->GetName(), "_false"),
                                 node_context.llvm_function(), join_block);
    llvm::IRBuilder<> false_builder(false_block);
    false_builder.CreateBr(join_block);

    b.CreateCondBr(predicate, true_block, false_block);

    auto exit_builder = std::make_unique<llvm::IRBuilder<>>(join_block);
    return FinalizeNodeIrContext(node_context, type_converter_->GetToken(),
                                 std::move(exit_builder));
  }
  // Unconditional send.
  XLS_RETURN_IF_ERROR(InvokeSendCallback(&b, queue, send, data, user_data));

  return FinalizeNodeIrContext(node_context, type_converter_->GetToken());
}

// Builds and returns and LLVM function which executes a single tick of the
// given proc. The LLVM function has the following signature:
//
//   void f(state_0_T state_0, .., state_n_T state_n,
//          state_0_t* next_state_0, ..., state_n_T* next_state_n,
//          void* events, void* user_data, void* jit_runtime)
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

  // Add current state parameters. These are passed by value.
  std::vector<llvm::Type*> param_types;
  for (int64_t i = 0; i < proc->GetStateElementCount(); ++i) {
    llvm::Type* llvm_state_type =
        jit.GetTypeConverter().ConvertToLlvmType(proc->GetStateElementType(i));
    param_types.push_back(llvm_state_type);
  }

  llvm::Type* ptr_type = llvm::PointerType::get(context, 0);

  // Add next state parameters. These are passed by pointer.
  param_types.insert(param_types.end(), proc->GetStateElementCount(), ptr_type);

  // After the XLS function parameters are:
  //   events pointer, user data, jit runtime
  // All are pointer type.
  param_types.push_back(ptr_type);
  param_types.push_back(ptr_type);
  param_types.push_back(ptr_type);

  llvm::FunctionType* function_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context),
      llvm::ArrayRef<llvm::Type*>(param_types.data(), param_types.size()),
      /*isVarArg=*/false);
  llvm::Function* llvm_function = llvm::cast<llvm::Function>(
      module->getOrInsertFunction(proc->qualified_name(), function_type)
          .getCallee());

  // Give the function parameters meaningful names.
  int64_t argc = 0;
  for (int64_t i = 0; i < proc->GetStateElementCount(); ++i) {
    llvm_function->getArg(argc++)->setName(proc->GetStateParam(i)->GetName());
  }
  for (int64_t i = 0; i < proc->GetStateElementCount(); ++i) {
    llvm_function->getArg(argc++)->setName(
        absl::StrFormat("%s_next", proc->GetStateParam(i)->GetName()));
  }
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
  //   void f(uint8_t*[] state, uint8_t*[] next_state,
  //          void* events, void* user_data, void* jit_runtime)
  //
  // At the llvm IR level pointers are untyped.
  llvm::Type* ptr_type = llvm::PointerType::get(*bare_context, 0);
  std::vector<llvm::Type*> param_types(5, ptr_type);

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

  std::vector<llvm::Value*> args;

  // Load the state values and add them to the list of arguments to pass to the
  // wrapped function.
  llvm::Value* state_array = wrapper_function->getArg(0);
  for (int64_t i = 0; i < proc_->GetStateElementCount(); ++i) {
    llvm::Type* llvm_state_type =
        orc_jit_->GetTypeConverter().ConvertToLlvmType(
            proc_->GetStateElementType(i));
    llvm::Value* state_element = IrBuilderVisitor::LoadFromPointerArray(
        i, llvm_state_type, state_array, proc_->GetStateElementCount(),
        &builder);
    state_element->setName(proc_->GetStateParam(i)->GetName());
    args.push_back(state_element);
  }

  // The buffers allocated to hold the next-state values. Allocated by the
  // caller of the wrapper function and passed in as array elements in argument
  // 1.
  std::vector<llvm::Value*> next_state_buffers;

  llvm::Value* next_state_array = wrapper_function->getArg(1);
  for (int64_t i = 0; i < proc_->GetStateElementCount(); ++i) {
    llvm::Type* llvm_state_type =
        orc_jit_->GetTypeConverter().ConvertToLlvmType(
            proc_->GetStateElementType(i));
    llvm::Value* next_state_ptr = IrBuilderVisitor::LoadPointerFromPointerArray(
        i, llvm_state_type, next_state_array, proc_->GetStateElementCount(),
        &builder);
    next_state_ptr->setName(
        absl::StrFormat("next_%s_ptr", proc_->GetStateParam(i)->GetName()));
    args.push_back(next_state_ptr);
    next_state_buffers.push_back(next_state_ptr);
  }

  // Pass through the final three arguments:
  //   interpreter events, user data, JIT runtime pointer
  args.push_back(wrapper_function->getArg(wrapper_function->arg_size() - 3));
  args.push_back(wrapper_function->getArg(wrapper_function->arg_size() - 2));
  args.push_back(wrapper_function->getArg(wrapper_function->arg_size() - 1));

  builder.CreateCall(callee, args);

  // Unpoison the next-state buffers.
  for (int64_t i = 0; i < proc_->GetStateElementCount(); ++i) {
    int64_t state_type_bytes = orc_jit_->GetTypeConverter().GetTypeByteSize(
        proc_->GetStateElementType(i));
    llvm::Value* next_state_buffer = next_state_buffers[i];
    IrBuilderVisitor::UnpoisonBuffer(next_state_buffer, state_type_bytes,
                                     &builder);
  }

  builder.CreateRetVoid();

  XLS_VLOG(3) << absl::StrFormat("LLVM function for %s:", proc_->name());
  XLS_VLOG(3) << DumpLlvmObjectToString(*wrapper_function);

  return wrapper_function;
}

absl::StatusOr<std::vector<std::vector<uint8_t>>> ProcJit::ConvertStateToView(
    absl::Span<const Value> state_value, bool initialize_with_value) {
  std::vector<std::vector<uint8_t>> state_buffers;

  for (int64_t i = 0; i < proc()->GetStateElementCount(); ++i) {
    Type* state_type = proc_->GetStateElementType(i);

    if (!ValueConformsToType(state_value[i], state_type)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Expected state argument %s (%d) to be of type %s, is: %s",
          proc_->GetStateParam(i)->GetName(), i, state_type->ToString(),
          state_value[i].ToString()));
    }

    state_buffers.push_back(std::vector<uint8_t>(
        orc_jit_->GetTypeConverter().GetTypeByteSize(state_type)));

    if (initialize_with_value) {
      ir_runtime_->BlitValueToBuffer(state_value[i], state_type,
                                     absl::MakeSpan(state_buffers.back()));
    }
  }

  return state_buffers;
}

std::vector<Value> ProcJit::ConvertStateViewToValue(
    absl::Span<uint8_t const* const> state_buffers) {
  std::vector<Value> state_values;
  for (int64_t i = 0; i < proc()->GetStateElementCount(); ++i) {
    Type* state_type = proc_->GetStateElementType(i);
    state_values.push_back(
        ir_runtime_->UnpackBuffer(state_buffers[i], state_type));
  }

  return state_values;
}

absl::Status ProcJit::RunWithViews(absl::Span<uint8_t const* const> state,
                                   absl::Span<uint8_t* const> next_state,
                                   void* user_data) {
  InterpreterEvents events;
  invoker_(state.data(), next_state.data(), &events, user_data, runtime());

  return absl::OkStatus();
}

absl::StatusOr<InterpreterResult<std::vector<Value>>> ProcJit::Run(
    absl::Span<const Value> state, void* user_data) {
  int64_t state_element_count = proc()->GetStateElementCount();

  // Buffers for state and next-state values. Allocate as std::unique_ptr.
  XLS_ASSIGN_OR_RETURN(
      std::vector<std::vector<uint8_t>> state_buffers,
      ConvertStateToView(state, /*initialize_with_value=*/true));

  XLS_ASSIGN_OR_RETURN(
      std::vector<std::vector<uint8_t>> next_state_buffers,
      ConvertStateToView(state, /*initialize_with_value=*/false));

  // Copy of `state_buffers` and `next_state_buffers` but as raw pointers for
  // passing to the generated function.
  std::vector<uint8_t*> raw_state_buffers(state_element_count);
  std::vector<uint8_t*> raw_next_state_buffers(state_element_count);

  for (int64_t i = 0; i < state_element_count; ++i) {
    raw_state_buffers.at(i) = state_buffers.at(i).data();
    raw_next_state_buffers.at(i) = next_state_buffers.at(i).data();
  }

  InterpreterEvents events;
  invoker_(raw_state_buffers.data(), raw_next_state_buffers.data(), &events,
           user_data, runtime());

  // Convert from LLVM native type to xls::Value for returning.
  std::vector<Value> next_state_values =
      ConvertStateViewToValue(raw_next_state_buffers);

  return InterpreterResult<std::vector<Value>>{std::move(next_state_values),
                                               std::move(events)};
}

}  // namespace xls
