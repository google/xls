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
#include "xls/jit/proc_builder_visitor.h"

#include "llvm/include/llvm/IR/BasicBlock.h"
#include "llvm/include/llvm/IR/DerivedTypes.h"
#include "llvm/include/llvm/IR/IRBuilder.h"
#include "llvm/include/llvm/IR/Instructions.h"

namespace xls {

absl::Status ProcBuilderVisitor::Visit(
    llvm::Module* module, llvm::Function* llvm_fn, FunctionBase* xls_fn,
    LlvmTypeConverter* type_converter, bool is_top, bool generate_packed,
    JitChannelQueueManager* queue_mgr, RecvFnT recv_fn, SendFnT send_fn) {
  ProcBuilderVisitor visitor(module, llvm_fn, xls_fn, type_converter, is_top,
                             generate_packed, queue_mgr, recv_fn, send_fn);
  return visitor.BuildInternal();
}

ProcBuilderVisitor::ProcBuilderVisitor(
    llvm::Module* module, llvm::Function* llvm_fn, FunctionBase* xls_fn,
    LlvmTypeConverter* type_converter, bool is_top, bool generate_packed,
    JitChannelQueueManager* queue_mgr, RecvFnT recv_fn, SendFnT send_fn)
    : FunctionBuilderVisitor(module, llvm_fn, xls_fn, type_converter, is_top,
                             generate_packed),
      queue_mgr_(queue_mgr),
      recv_fn_(recv_fn),
      send_fn_(send_fn) {}

absl::StatusOr<llvm::Value*> ProcBuilderVisitor::InvokeRecvCallback(
    llvm::IRBuilder<>* builder, JitChannelQueue* queue, Node* node) {
  llvm::Type* void_type = llvm::Type::getVoidTy(ctx());
  llvm::Type* int64_type = llvm::Type::getInt64Ty(ctx());
  llvm::Type* int8_ptr_type = llvm::Type::getInt8PtrTy(ctx(), 0);

  // To actually receive a message, we'll be pulling it from some queue (that
  // will be known at JIT compilation time). Rather than trying to code that
  // up as LLVM IR, we provide the dequeue operation as an external function
  // (currently "DequeueMessage"). To call such a function - being defined
  // outside LLVM - we need to:
  //  1) conceptually add it to our module under construction, which requires
  //     defining its signature to LLVM,
  // LLVM doesn't like void* types, so we use int64_t's instead.
  std::vector<llvm::Type*> params(
      {int64_type, int64_type, int8_ptr_type, int64_type, int64_type});
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  llvm::Type* recv_type = type_converter()->ConvertToLlvmType(node->GetType());
  int64_t recv_bytes = type_converter()->GetTypeByteSize(node->GetType());
  llvm::AllocaInst* alloca = builder->CreateAlloca(recv_type);

  //  2) create the argument list to pass to the function. We use opaque
  //     pointers to our data elements, to avoid recursively defining every
  //     type used by every type and so on.
  static_assert(sizeof(Receive) == sizeof(ReceiveIf));
  std::vector<llvm::Value*> args({
      llvm::ConstantInt::get(int64_type, reinterpret_cast<uint64_t>(queue)),
      // This is sinful, I know, but ReceiveIf and Receive are
      // layout-compatible..._for now_ (hence the static_assert above).
      // TODO(meheff) : Make Receive & ReceiveIf share a common base
      // class (also Send & SendIf).
      llvm::ConstantInt::get(int64_type, reinterpret_cast<uint64_t>(node)),
      builder->CreatePointerCast(alloca, int8_ptr_type),
      llvm::ConstantInt::get(int64_type, recv_bytes),
      GetUserDataPtr(),
  });

  // 3) finally emit the function call,
  llvm::ConstantInt* fn_addr = llvm::ConstantInt::get(
      llvm::Type::getInt64Ty(ctx()), reinterpret_cast<uint64_t>(recv_fn_));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  builder->CreateCall(fn_type, fn_ptr, args);

  // 4) then load its result from the bounce buffer.
  return builder->CreateLoad(alloca);
}

absl::Status ProcBuilderVisitor::HandleReceive(Receive* recv) {
  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue,
                       queue_mgr_->GetQueueById(recv->channel_id()));
  XLS_ASSIGN_OR_RETURN(llvm::Value * invoke,
                       InvokeRecvCallback(builder(), queue, recv));
  llvm::Value* result =
      builder()->CreateInsertValue(invoke, type_converter()->GetToken(), {0});
  return StoreResult(recv, result);
}

absl::Status ProcBuilderVisitor::HandleReceiveIf(ReceiveIf* recv_if) {
  // First, declare the join block (so the case blocks can refer to it).
  llvm::BasicBlock* join_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(recv_if->GetName(), "_join"), llvm_fn());

  // Create a block/branch for the true predicate case.
  llvm::BasicBlock* true_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(recv_if->GetName(), "_true"), llvm_fn(), join_block);
  llvm::IRBuilder<> true_builder(true_block);
  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue,
                       queue_mgr_->GetQueueById(recv_if->channel_id()));
  XLS_ASSIGN_OR_RETURN(llvm::Value * true_result,
                       InvokeRecvCallback(&true_builder, queue, recv_if));
  llvm::Value* true_token = type_converter()->GetToken();
  true_builder.CreateBr(join_block);

  // And the same for a false predicate - this will return an empty/zero value.
  // Creating an empty struct emits ops, so it needs a builder.
  llvm::BasicBlock* false_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(recv_if->GetName(), "_false"), llvm_fn(), join_block);
  llvm::IRBuilder<> false_builder(false_block);
  llvm::Type* result_type =
      type_converter()->ConvertToLlvmType(recv_if->GetType());
  llvm::Value* false_result = CreateTypedZeroValue(result_type);
  llvm::Value* false_token = node_map().at(recv_if->operand(0));
  false_builder.CreateBr(join_block);

  // Next, create a branch op w/the original builder,
  builder()->CreateCondBr(node_map().at(recv_if->predicate()), true_block,
                          false_block);

  // then join the two branches back together.
  auto join_builder = std::make_unique<llvm::IRBuilder<>>(join_block);
  llvm::PHINode* phi =
      join_builder->CreatePHI(result_type, /*NumReservedValues=*/2);
  phi->addIncoming(true_result, true_block);
  phi->addIncoming(false_result, false_block);

  llvm::PHINode* token_phi = join_builder->CreatePHI(
      type_converter()->GetTokenType(), /*NumReservedValues=*/2);
  token_phi->addIncoming(true_token, true_block);
  token_phi->addIncoming(false_token, false_block);
  llvm::Value* result = join_builder->CreateInsertValue(phi, token_phi, {0});

  // Finally, set this's IRBuilder to be the output block's (since that's where
  // the Function continues).
  set_builder(std::move(join_builder));
  return StoreResult(recv_if, result);
}

absl::Status ProcBuilderVisitor::InvokeSendCallback(llvm::IRBuilder<>* builder,
                                                    JitChannelQueue* queue,
                                                    Node* node, Node* data) {
  llvm::Type* void_type = llvm::Type::getVoidTy(ctx());
  llvm::Type* int64_type = llvm::Type::getInt64Ty(ctx());
  llvm::Type* int8_ptr_type = llvm::Type::getInt8PtrTy(ctx(), 0);

  // We do the same for sending/enqueuing as we do for receiving/dequeueing
  // above (set up and call an external function).
  std::vector<llvm::Type*> params({
      int64_type,
      int64_type,
      int8_ptr_type,
      int64_type,
      int64_type,
  });
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

  std::vector<llvm::Value*> args({
      llvm::ConstantInt::get(int64_type, reinterpret_cast<uint64_t>(queue)),
      llvm::ConstantInt::get(int64_type, reinterpret_cast<uint64_t>(node)),
      builder->CreatePointerCast(alloca, int8_ptr_type),
      llvm::ConstantInt::get(int64_type, send_type_size),
      GetUserDataPtr(),
  });

  llvm::ConstantInt* fn_addr = llvm::ConstantInt::get(
      llvm::Type::getInt64Ty(ctx()), reinterpret_cast<uint64_t>(send_fn_));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  builder->CreateCall(fn_type, fn_ptr, args);
  return absl::OkStatus();
}

absl::Status ProcBuilderVisitor::HandleSend(Send* send) {
  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue,
                       queue_mgr_->GetQueueById(send->channel_id()));
  XLS_RETURN_IF_ERROR(InvokeSendCallback(builder(), queue, send, send->data()));
  return StoreResult(send, type_converter()->GetToken());
}

absl::Status ProcBuilderVisitor::HandleSendIf(SendIf* send_if) {
  // First, declare the join block (so the case blocks can refer to it).
  llvm::BasicBlock* join_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(send_if->GetName(), "_join"), llvm_fn());

  llvm::BasicBlock* true_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(send_if->GetName(), "_true"), llvm_fn(), join_block);
  llvm::IRBuilder<> true_builder(true_block);
  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue,
                       queue_mgr_->GetQueueById(send_if->channel_id()));
  XLS_RETURN_IF_ERROR(
      InvokeSendCallback(&true_builder, queue, send_if, send_if->data()));
  llvm::Value* true_token = type_converter()->GetToken();
  true_builder.CreateBr(join_block);

  llvm::BasicBlock* false_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(send_if->GetName(), "_false"), llvm_fn(), join_block);
  llvm::IRBuilder<> false_builder(false_block);
  llvm::Value* false_token = node_map().at(send_if->operand(0));
  false_builder.CreateBr(join_block);

  builder()->CreateCondBr(node_map().at(send_if->predicate()), true_block,
                          false_block);

  auto join_builder = std::make_unique<llvm::IRBuilder<>>(join_block);
  llvm::PHINode* phi = join_builder->CreatePHI(type_converter()->GetTokenType(),
                                               /*NumReservedValues=*/2);
  phi->addIncoming(true_token, true_block);
  phi->addIncoming(false_token, false_block);

  set_builder(std::move(join_builder));
  return StoreResult(send_if, phi);
}

}  // namespace xls
