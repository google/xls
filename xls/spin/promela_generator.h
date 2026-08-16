// Copyright 2026 The XLS Authors
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

#ifndef XLS_SPIN_PROMELA_GENERATOR_H_
#define XLS_SPIN_PROMELA_GENERATOR_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/package.h"

namespace xls::spin {

struct PromelaGeneratorOptions {
  // Annotate each Promela statement with a /* filename:line:col */ comment.
  bool emit_source_locations = false;

  // Annotate each Promela statement with a /* ir: <node-expr> */ comment.
  bool emit_source_hints = false;

  // Buffer depth N in `chan x = [N] of {T}`. Default 8.
  int64_t channel_depth = 8;

  // Block init on the terminator channel (mirrors #[test_proc]).
  bool emit_termination_hook = false;

  // Prefix every send with `assert(len(ch) < DEPTH)` to turn a full-channel
  // block into an explicit SPIN assertion violation.
  bool assert_send_on_full_channel = false;

  // Maps sanitised proc name to N: proc does real work at most once every N
  // loop iterations; other iterations are idle stalls. Procs not listed use 1.
  absl::flat_hash_map<std::string, int64_t> worst_case_throughput;

  // Prefix each channel send/receive with a SPIN progress label for livelock
  // detection via `spin -search -DNP`.
  bool emit_progress_labels = false;
};

// Translates an XLS IR package to Promela source text via DFS post-order
// traversal. Use the static Generate() factory; do not instantiate directly.
class PromelaGenerator : public DfsVisitorWithDefault {
 public:
  // Translates all functions and procs in `package` into Promela source text.
  // Requires proc-scoped channels; returns an error for old-style packages.
  static absl::StatusOr<std::string> Generate(
      Package* package,
      const PromelaGeneratorOptions& options = PromelaGeneratorOptions{});

  // DfsVisitorWithDefault overrides.
  absl::Status DefaultHandler(Node* node) override;
  absl::Status HandleAfterAll(AfterAll* after_all) override;
  absl::Status HandleMinDelay(MinDelay* min_delay) override;
  absl::Status HandleTrace(Trace* trace_op) override;
  absl::Status HandleCover(Cover* cover) override;
  absl::Status HandleNewChannel(NewChannel* new_channel) override;
  absl::Status HandleRecvChannelEnd(RecvChannelEnd* rce) override;
  absl::Status HandleSendChannelEnd(SendChannelEnd* sce) override;
  absl::Status HandleParam(Param* param) override;
  absl::Status HandleStateRead(StateRead* state_read) override;
  absl::Status HandleNext(Next* next) override;
  absl::Status HandleLiteral(Literal* literal) override;
  absl::Status HandleAdd(BinOp* add) override;
  absl::Status HandleSub(BinOp* sub) override;
  absl::Status HandleUMul(ArithOp* mul) override;
  absl::Status HandleSMul(ArithOp* mul) override;
  absl::Status HandleUDiv(BinOp* div) override;
  absl::Status HandleSDiv(BinOp* div) override;
  absl::Status HandleUMod(BinOp* mod) override;
  absl::Status HandleSMod(BinOp* mod) override;
  absl::Status HandleNeg(UnOp* neg) override;
  absl::Status HandleIdentity(UnOp* identity) override;
  absl::Status HandleNot(UnOp* not_op) override;
  absl::Status HandleNaryAnd(NaryOp* and_op) override;
  absl::Status HandleNaryOr(NaryOp* or_op) override;
  absl::Status HandleNaryXor(NaryOp* xor_op) override;
  absl::Status HandleNaryNand(NaryOp* nand_op) override;
  absl::Status HandleNaryNor(NaryOp* nor_op) override;
  absl::Status HandleAndReduce(BitwiseReductionOp* and_reduce) override;
  absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce) override;
  absl::Status HandleShll(BinOp* shll) override;
  absl::Status HandleShrl(BinOp* shrl) override;
  absl::Status HandleShra(BinOp* shra) override;
  absl::Status HandleEq(CompareOp* eq) override;
  absl::Status HandleNe(CompareOp* ne) override;
  absl::Status HandleULt(CompareOp* lt) override;
  absl::Status HandleULe(CompareOp* le) override;
  absl::Status HandleUGt(CompareOp* gt) override;
  absl::Status HandleUGe(CompareOp* ge) override;
  absl::Status HandleSLt(CompareOp* lt) override;
  absl::Status HandleSLe(CompareOp* le) override;
  absl::Status HandleSGt(CompareOp* gt) override;
  absl::Status HandleSGe(CompareOp* ge) override;
  absl::Status HandleBitSlice(BitSlice* bit_slice) override;
  absl::Status HandleDynamicBitSlice(
      DynamicBitSlice* dynamic_bit_slice) override;
  absl::Status HandleSignExtend(ExtendOp* sign_ext) override;
  absl::Status HandleZeroExtend(ExtendOp* zero_ext) override;
  absl::Status HandleConcat(Concat* concat) override;
  absl::Status HandleSel(Select* sel) override;
  absl::Status HandleOneHotSel(OneHotSelect* sel) override;
  absl::Status HandlePrioritySel(PrioritySelect* sel) override;
  absl::Status HandleGate(Gate* gate) override;
  absl::Status HandleTuple(Tuple* tuple) override;
  absl::Status HandleTupleIndex(TupleIndex* index) override;
  absl::Status HandleReceive(Receive* receive) override;
  absl::Status HandleSend(Send* send) override;
  absl::Status HandleAssert(Assert* assert_op) override;
  absl::Status HandleInvoke(Invoke* invoke) override;

  // Unhandled ops fall through to DefaultHandler (UnimplementedError):
  // InputPort, OutputPort, XorReduce, Reverse, OneHot, Array*, Map,
  // CountedFor, BitSliceUpdate, Encode, Decode, SMulp, UMulp.

 private:
  // Indented-line emitter. Each indent level adds two spaces.
  class Emitter {
   public:
    explicit Emitter(std::string& out, int64_t level = 0)
        : out_(out), level_(level) {}
    void Reset() { level_ = 0; }
    void Indent() { ++level_; }
    void Dedent() {
      if (level_ > 0) {
        --level_;
      }
    }
    void Blank() { out_ += '\n'; }
    void Line(std::string_view text) {
      WriteIndent();
      absl::StrAppend(&out_, text, "\n");
    }
    template <typename... Args>
    void Line(std::string_view fmt, const Args&... args) {
      WriteIndent();
      absl::SubstituteAndAppend(&out_, fmt, args...);
      out_ += '\n';
    }

   private:
    void WriteIndent() {
      for (int64_t i = 0; i < level_; ++i) {
        absl::StrAppend(&out_, "  ");
      }
    }
    std::string& out_;
    int64_t level_;
  };

  explicit PromelaGenerator(Package* package,
                            const PromelaGeneratorOptions& options);

  // Package-level orchestration.
  absl::Status ValidateTypes();
  std::vector<Proc*> FindRootProcs() const;
  absl::Status EmitFunction(Function* fn);
  absl::Status EmitProc(Proc* proc);
  void EmitInit(const std::vector<Proc*>& root_procs);

  // Node name tracking.
  absl::Status SetName(Node* node, std::string name);
  std::string_view Ref(Node* node) const;
  std::string Var(Node* node) const;

  // Code emission helpers.
  absl::Status Assign(Node* node, std::string_view expr);
  absl::Status EmitCompare(CompareOp* node, std::string_view cmp_op);
  absl::Status EmitNaryBitwise(NaryOp* node, std::string_view op_sym,
                               bool invert = false);
  absl::Status EmitBitwiseReduce(BitwiseReductionOp* node,
                                 std::string_view cmp_op,
                                 std::string_view sentinel);
  absl::Status EmitMul(ArithOp* mul);
  std::string ProgressLabel(std::string_view dir, std::string_view chan) const;
  void MaybeEmitLocComment(Node* node);
  void MaybeEmitIrHintComment(Node* node);

  void Emit(std::string_view text) { emit_.Line(text); }
  template <typename... Args>
  void Emit(std::string_view fmt, const Args&... args) {
    emit_.Line(fmt, args...);
  }

  Package* package_;
  PromelaGeneratorOptions options_;
  std::string out_;
  Emitter emit_;

  // Per-FunctionBase visit state; set up in EmitFunction / EmitProc.
  absl::flat_hash_map<Node*, std::string> names_;
  absl::flat_hash_map<Node*, std::vector<std::string>> tuple_components_;
  // Collects Next nodes during proc visits for deferred state-update emit.
  std::vector<Node*> next_nodes_;
};

}  // namespace xls::spin

#endif  // XLS_SPIN_PROMELA_GENERATOR_H_
