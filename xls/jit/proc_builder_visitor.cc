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

#include "llvm/IR/DerivedTypes.h"

namespace xls {

absl::Status ProcBuilderVisitor::Visit(
    llvm::Module* module, llvm::Function* llvm_fn, Function* xls_fn,
    LlvmTypeConverter* type_converter, bool is_top, bool generate_packed,
    JitChannelQueueManager* queue_mgr, RecvFnT recv_fn, SendFnT send_fn) {
  ProcBuilderVisitor visitor(module, llvm_fn, xls_fn, type_converter, is_top,
                             generate_packed, queue_mgr, recv_fn, send_fn);
  return visitor.BuildInternal();
}

ProcBuilderVisitor::ProcBuilderVisitor(
    llvm::Module* module, llvm::Function* llvm_fn, Function* xls_fn,
    LlvmTypeConverter* type_converter, bool is_top, bool generate_packed,
    JitChannelQueueManager* queue_mgr, RecvFnT recv_fn, SendFnT send_fn)
    : FunctionBuilderVisitor(module, llvm_fn, xls_fn, type_converter, is_top,
                             generate_packed),
      queue_mgr_(queue_mgr),
      recv_fn_(recv_fn),
      send_fn_(send_fn) {}

absl::Status ProcBuilderVisitor::HandleReceive(Receive* recv) {
  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue,
                       queue_mgr_->GetQueueById(recv->channel_id()));

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
  // LLVM doesn't like void* types, so we use int64s instead.
  std::vector<llvm::Type*> params(
      {int64_type, int64_type, int8_ptr_type, int64_type});
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  llvm::Type* recv_type = type_converter()->ConvertToLlvmType(*recv->GetType());
  int64 recv_bytes = type_converter()->GetTypeByteSize(*recv->GetType());
  llvm::AllocaInst* alloca = builder()->CreateAlloca(recv_type);

  //  2) create the argument list to pass to the function. We use opaque
  //     pointers to our data elements, to avoid recursively defining every
  //     type used by every type and so on.
  std::vector<llvm::Value*> args(
      {llvm::ConstantInt::get(int64_type, reinterpret_cast<uint64>(queue)),
       llvm::ConstantInt::get(int64_type, reinterpret_cast<uint64>(recv)),
       builder()->CreatePointerCast(alloca, int8_ptr_type),
       llvm::ConstantInt::get(int64_type, recv_bytes)});

  // 3) finally emit the function call,
  llvm::ConstantInt* fn_addr = llvm::ConstantInt::get(
      llvm::Type::getInt64Ty(ctx()), reinterpret_cast<uint64>(recv_fn_));
  llvm::Value* fn_ptr =
      builder()->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  builder()->CreateCall(fn_type, fn_ptr, args);

  // 4) then load its result from the bounce buffer.
  llvm::Value* xls_value = builder()->CreateLoad(alloca);
  return StoreResult(recv, xls_value);
}

absl::Status ProcBuilderVisitor::HandleSend(Send* send) {
  llvm::Type* void_type = llvm::Type::getVoidTy(ctx());
  llvm::Type* int64_type = llvm::Type::getInt64Ty(ctx());
  llvm::Type* int8_ptr_type = llvm::Type::getInt8PtrTy(ctx(), 0);

  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue,
                       queue_mgr_->GetQueueById(send->channel_id()));

  // We do the same for sending/enqueuing as we do for receiving/dequeueing
  // above (set up and call an external function).
  std::vector<llvm::Type*> params({
      int64_type,
      int64_type,
      int8_ptr_type,
      int64_type,
  });
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  // Pack all the data to be sent into a contiguous tuple (we'll need to pack
  // the data anyway, since llvm::Values don't automatically correspond to
  // pointer-referencable storage; that's what allocas are for).
  std::vector<Type*> tuple_elems;
  for (const Node* node : send->data_operands()) {
    tuple_elems.push_back(node->GetType());
  }
  TupleType tuple_type(tuple_elems);
  llvm::Type* send_op_types = type_converter()->ConvertToLlvmType(tuple_type);
  int64 send_type_size = type_converter()->GetTypeByteSize(tuple_type);
  llvm::Value* tuple = CreateTypedZeroValue(send_op_types);
  for (int i = 0; i < send->data_operands().size(); i++) {
    tuple = builder()->CreateInsertValue(
        tuple, node_map().at(send->data_operands()[i]), {static_cast<uint>(i)});
  }
  llvm::AllocaInst* alloca = builder()->CreateAlloca(send_op_types);
  builder()->CreateStore(tuple, alloca);

  std::vector<llvm::Value*> args({
      llvm::ConstantInt::get(int64_type, reinterpret_cast<uint64>(queue)),
      llvm::ConstantInt::get(int64_type, reinterpret_cast<uint64>(send)),
      builder()->CreatePointerCast(alloca, int8_ptr_type),
      llvm::ConstantInt::get(int64_type, send_type_size),
  });

  llvm::ConstantInt* fn_addr = llvm::ConstantInt::get(
      llvm::Type::getInt64Ty(ctx()), reinterpret_cast<uint64>(send_fn_));
  llvm::Value* fn_ptr =
      builder()->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  builder()->CreateCall(fn_type, fn_ptr, args);
  return absl::OkStatus();
}

}  // namespace xls
