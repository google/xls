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
#ifndef XLS_JIT_PROC_BUILDER_VISITOR_H_
#define XLS_JIT_PROC_BUILDER_VISITOR_H_

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "xls/ir/function.h"
#include "xls/jit/function_builder_visitor.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/llvm_type_converter.h"

namespace xls {

// ProcBuilderVisitor builds on FunctionBuilderVisitor by adding support for
// Recv/Send (and their related "If") nodes.
// The actual behaviors of Recv/Send nodes are defined in externally-provided
// (i.e., user-defined) functions specified as "plugins" to the visitor in
// Visit(). In this way, a number of "runtimes" may be specified to enable
// different behaviors, depending on user need: debugging, performance,
// scalability, etc.
class ProcBuilderVisitor : public FunctionBuilderVisitor {
 public:
  // The receive function has the following prototype:
  // void recv_fn(uint64 queue_ptr, uint64 recv_ptr, uint8* buffer,
  //              int64 data_sz);
  // where:
  //  - queue_ptr is a pointer to a JitChannelQueue,
  //  - recv_ptr is a pointer to a Receive node,
  //  - buffer is a pointer to the data to fill (with incoming data), and
  //  - data_sz is the size of the receive buffer.
  //
  // The send function has the following prototype:
  // void send_fn(uint64 queue_ptr, uint64 send_ptr, uint8* buffer,
  //              int64 data_sz);
  // where:
  //  - queue_ptr is a pointer to a JitChannelQueue,
  //  - send_ptr is a pointer to a Send node,
  //  - buffer is a pointer to the data to fill (with incoming data), and
  //  - data_sz is the size of the receive buffer.
  //
  // Internally, the JIT looks up functions _by_name_, which is why we have
  // recv_fn_name and send_fn_name instead of function pointers. Needless to
  // say, these functions must be in-process. It's likely helpful to declare
  // them with "C" linkage, as well.

  // Populates llvm_fn with an LLVM IR translation of the given xls_fn, calling
  // out to the specified receive and send functions when encountering their
  // respective nodes.
  using RecvFnT = void (*)(JitChannelQueue*, Receive*, uint8*, int64);
  using SendFnT = void (*)(JitChannelQueue*, Send*, uint8*, int64);
  static absl::Status Visit(llvm::Module* module, llvm::Function* llvm_fn,
                            Function* xls_fn, LlvmTypeConverter* type_converter,
                            bool is_top, bool generate_packed,
                            JitChannelQueueManager* queue_mgr, RecvFnT recv_fn,
                            SendFnT send_fn);

  absl::Status HandleReceive(Receive* recv) override;
  absl::Status HandleReceiveIf(ReceiveIf* recv) override;
  absl::Status HandleSend(Send* send) override;
  absl::Status HandleSendIf(SendIf* send) override;

 private:
  ProcBuilderVisitor(llvm::Module* module, llvm::Function* llvm_fn,
                     Function* xls_fn, LlvmTypeConverter* type_converter,
                     bool is_top, bool generate_packed,
                     JitChannelQueueManager* queue_mgr, RecvFnT recv_fn,
                     SendFnT send_fn);

  absl::StatusOr<llvm::Value*> InvokeRecvCallback(llvm::IRBuilder<>* builder,
                                                  JitChannelQueue* queue,
                                                  Node* node);

  absl::Status InvokeSendCallback(llvm::IRBuilder<>* builder,
                                  JitChannelQueue* queue, Node* node,
                                  absl::Span<Node* const> operands);

 public:
  JitChannelQueueManager* queue_mgr_;
  RecvFnT recv_fn_;
  SendFnT send_fn_;
};

}  // namespace xls

#endif  // XLS_JIT_PROC_BUILDER_VISITOR_H_
