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

#ifndef XLS_SOLVERS_BITWUZLA_IR_TRANSLATOR_H_
#define XLS_SOLVERS_BITWUZLA_IR_TRANSLATOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "bitwuzla/cpp/bitwuzla.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/solvers/solver.h"

namespace xls::solvers::bitwuzla {

class IrTranslator : public DfsVisitorWithDefault {
 public:
  static absl::StatusOr<std::unique_ptr<IrTranslator>> CreateAndTranslate(
      FunctionBase* source, bool allow_unsupported = false);

  static absl::StatusOr<std::unique_ptr<IrTranslator>> CreateAndTranslate(
      ::bitwuzla::TermManager& tm, FunctionBase* source,
      absl::Span<const ::bitwuzla::Term> imported_params,
      bool allow_unsupported = false);

  static absl::StatusOr<std::unique_ptr<IrTranslator>> CreateAndTranslate(
      ::bitwuzla::TermManager& tm, Node* source,
      bool allow_unsupported = false);

  ~IrTranslator() override;

  void SetTimeout(std::optional<absl::Duration> timeout);
  void SetDeterministicLimit(std::optional<int64_t> limit);

  ::bitwuzla::Term GetTranslation(const Node* source);
  ::bitwuzla::Term GetReturnNode();

  absl::Status Retranslate(
      const absl::flat_hash_map<const Node*, ::bitwuzla::Term>& replacements);

  absl::StatusOr<ProverResult> TryProveCombination(
      absl::Span<const PredicateOfNode> terms, PredicateCombination combination,
      absl::Span<const PredicateOfNode> assumptions = {});

  ::bitwuzla::TermManager& tm() { return tm_; }
  FunctionBase* xls_function() { return xls_function_; }

  std::optional<absl::Duration> timeout() const { return timeout_; }
  std::optional<int64_t> deterministic_limit() const { return limit_; }

  // DfsVisitorWithDefault overrides
  absl::Status DefaultHandler(Node* node) override;
  absl::Status HandleAdd(BinOp* add) override;
  absl::Status HandleSub(BinOp* sub) override;
  absl::Status HandleSMul(ArithOp* mul) override;
  absl::Status HandleUMul(ArithOp* mul) override;
  absl::Status HandleSMulp(PartialProductOp* mul) override;
  absl::Status HandleUMulp(PartialProductOp* mul) override;
  absl::Status HandleSDiv(BinOp* div) override;
  absl::Status HandleUDiv(BinOp* div) override;
  absl::Status HandleSMod(BinOp* mod) override;
  absl::Status HandleUMod(BinOp* mod) override;

  absl::Status HandleEq(CompareOp* eq) override;
  absl::Status HandleNe(CompareOp* ne) override;
  absl::Status HandleSGe(CompareOp* sge) override;
  absl::Status HandleSGt(CompareOp* gt) override;
  absl::Status HandleSLe(CompareOp* le) override;
  absl::Status HandleSLt(CompareOp* lt) override;
  absl::Status HandleUGe(CompareOp* ge) override;
  absl::Status HandleUGt(CompareOp* gt) override;
  absl::Status HandleULe(CompareOp* le) override;
  absl::Status HandleULt(CompareOp* lt) override;

  absl::Status HandleNaryAnd(NaryOp* and_op) override;
  absl::Status HandleNaryOr(NaryOp* or_op) override;
  absl::Status HandleNaryXor(NaryOp* xor_op) override;
  absl::Status HandleNaryNand(NaryOp* nand_op) override;
  absl::Status HandleNaryNor(NaryOp* nor_op) override;

  absl::Status HandleNot(UnOp* not_op) override;
  absl::Status HandleNeg(UnOp* neg) override;
  absl::Status HandleIdentity(UnOp* identity) override;
  absl::Status HandleReverse(UnOp* reverse) override;

  absl::Status HandleAndReduce(BitwiseReductionOp* and_reduce) override;
  absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce) override;
  absl::Status HandleXorReduce(BitwiseReductionOp* xor_reduce) override;

  absl::Status HandleShll(BinOp* shll) override;
  absl::Status HandleShrl(BinOp* shrl) override;
  absl::Status HandleShra(BinOp* shra) override;

  absl::Status HandleSignExtend(ExtendOp* sign_ext) override;
  absl::Status HandleZeroExtend(ExtendOp* zero_ext) override;

  absl::Status HandleConcat(Concat* concat) override;
  absl::Status HandleBitSlice(BitSlice* bit_slice) override;
  absl::Status HandleDynamicBitSlice(DynamicBitSlice* bit_slice) override;
  absl::Status HandleBitSliceUpdate(BitSliceUpdate* update) override;

  absl::Status HandleSel(Select* sel) override;
  absl::Status HandleOneHot(OneHot* one_hot) override;
  absl::Status HandleOneHotSel(OneHotSelect* one_hot_sel) override;
  absl::Status HandlePrioritySel(PrioritySelect* priority_sel) override;

  absl::Status HandleTuple(Tuple* tuple) override;
  absl::Status HandleTupleIndex(TupleIndex* tuple_index) override;

  absl::Status HandleArray(Array* array) override;
  absl::Status HandleArrayIndex(ArrayIndex* array_index) override;
  absl::Status HandleArrayUpdate(ArrayUpdate* update) override;
  absl::Status HandleArrayConcat(ArrayConcat* array_concat) override;
  absl::Status HandleArraySlice(ArraySlice* slice) override;

  absl::Status HandleLiteral(Literal* literal) override;
  absl::Status HandleParam(Param* param) override;

  absl::Status HandleAfterAll(AfterAll* after_all) override;
  absl::Status HandleMinDelay(MinDelay* min_delay) override;
  absl::Status HandleDecode(Decode* decode) override;
  absl::Status HandleEncode(Encode* encode) override;
  absl::Status HandleGate(Gate* gate) override;
  absl::Status HandleNext(Next* next) override;
  absl::Status HandleInvoke(Invoke* invoke) override;

 private:
  IrTranslator(std::unique_ptr<::bitwuzla::TermManager> owned_tm,
               FunctionBase* source, bool allow_unsupported);
  IrTranslator(::bitwuzla::TermManager& tm, FunctionBase* source,
               bool allow_unsupported);

  ::bitwuzla::Sort TypeToSort(const Type* type);
  ::bitwuzla::Sort GetArrayIndexSort(const ArrayType* type);
  ::bitwuzla::Term GetAsFormattedArrayIndex(int64_t index,
                                            const ArrayType* type);
  ::bitwuzla::Term GetAsFormattedArrayIndex(::bitwuzla::Term index,
                                            const ArrayType* type,
                                            ::bitwuzla::Term* oob = nullptr);

  ::bitwuzla::Term TranslateLiteralBits(const Bits& bits);
  absl::StatusOr<::bitwuzla::Term> TranslateLiteralValue(const Type* type,
                                                         const Value& val);

  ::bitwuzla::Term ZeroOfSort(const ::bitwuzla::Sort& sort);
  ::bitwuzla::Term OnesOfSort(const ::bitwuzla::Sort& sort);
  ::bitwuzla::Term MinSignedOfSort(const ::bitwuzla::Sort& sort);
  ::bitwuzla::Term MaxSignedOfSort(const ::bitwuzla::Sort& sort);

  ::bitwuzla::Term ConcatN(const std::vector<::bitwuzla::Term>& args);
  ::bitwuzla::Term ConcatN(absl::Span<const ::bitwuzla::Term> args);
  ::bitwuzla::Term FlattenToBv(::bitwuzla::Term term, const Type* type);
  ::bitwuzla::Term UnflattenFromArrayBv(::bitwuzla::Term flat_bv,
                                        const Type* type);

  ::bitwuzla::Term GetArrayElement(const ArrayType* type,
                                   ::bitwuzla::Term array,
                                   ::bitwuzla::Term index);
  ::bitwuzla::Term UpdateArrayElement(
      const Type* type, ::bitwuzla::Term array, ::bitwuzla::Term value,
      ::bitwuzla::Term cond, absl::Span<const ::bitwuzla::Term> indices);

  ::bitwuzla::Term ComputeNe(::bitwuzla::Term lhs, ::bitwuzla::Term rhs,
                             const Type* type);
  ::bitwuzla::Term CoerceShiftAmount(::bitwuzla::Term val, Node* shamt_node);
  ::bitwuzla::Term ExtendOrTruncate(Node* operand, int64_t width,
                                    bool is_signed);

  absl::StatusOr<::bitwuzla::Term> GetLttElement(
      const Type* type, ::bitwuzla::Term value,
      absl::Span<const int64_t> index);
  absl::StatusOr<LeafTypeTree<::bitwuzla::Term>> ToLeafTypeTree(
      const Node* node);
  absl::StatusOr<LeafTypeTree<::bitwuzla::Term>> ToLeafTypeTree(
      Type* type, ::bitwuzla::Term term);

  absl::StatusOr<::bitwuzla::Term> TernaryToConstraint(
      ::bitwuzla::Term target_val, TernarySpan ternary);
  absl::StatusOr<::bitwuzla::Term> IntervalSetToConstraint(
      ::bitwuzla::Term target_val, const IntervalSet& intervals);

  absl::StatusOr<Value> ExtractValue(::bitwuzla::Bitwuzla& bitwuzla,
                                     ::bitwuzla::Term term, const Type* type);

  absl::StatusOr<::bitwuzla::Term> PredicateToAssertion(const Predicate& p,
                                                        Node* subject,
                                                        ::bitwuzla::Term val);
  absl::StatusOr<::bitwuzla::Term> PredicateToNegatedObjective(
      const Predicate& p, Node* subject, ::bitwuzla::Term val);

  void NoteTranslation(Node* node, ::bitwuzla::Term term);

  std::unique_ptr<::bitwuzla::TermManager> owned_tm_;
  ::bitwuzla::TermManager& tm_;
  FunctionBase* xls_function_;
  bool allow_unsupported_;
  std::optional<absl::Duration> timeout_;
  std::optional<int64_t> limit_;
  absl::flat_hash_map<const Node*, ::bitwuzla::Term> translations_;
  uint64_t symbol_count_ = 0;
  std::string GetNewSymbol(std::string_view prefix);
};

class BitwuzlaSolverInstance : public xls::solvers::SolverInstance {
 public:
  explicit BitwuzlaSolverInstance(std::unique_ptr<IrTranslator> translator)
      : translator_(std::move(translator)) {}

  void SetLimit(const SolverLimit& limit) override;

  absl::StatusOr<ProverResult> TryProve(
      Node* subject, const Predicate& p,
      absl::Span<const PredicateOfNode> assumptions = {}) override;

  absl::StatusOr<ProverResult> TryProveCombination(
      absl::Span<const PredicateOfNode> terms, PredicateCombination combination,
      absl::Span<const PredicateOfNode> assumptions = {}) override;

 private:
  std::unique_ptr<IrTranslator> translator_;
};

class BitwuzlaSolver : public xls::solvers::Solver {
 public:
  SolverKind kind() const override { return SolverKind::kBitwuzla; }

  absl::StatusOr<std::unique_ptr<xls::solvers::SolverInstance>>
  CreateSolverInstance(FunctionBase* f,
                       bool allow_unsupported = false) override;

  absl::StatusOr<ProverResult> TryProve(
      FunctionBase* f, Node* subject, const Predicate& p,
      const SolverLimit& limit, bool allow_unsupported = false,
      absl::Span<const PredicateOfNode> assumptions = {}) override;

  absl::StatusOr<ProverResult> TryProveCombination(
      FunctionBase* f, absl::Span<const PredicateOfNode> terms,
      PredicateCombination combination, const SolverLimit& limit,
      bool allow_unsupported = false,
      absl::Span<const PredicateOfNode> assumptions = {}) override;
};

}  // namespace xls::solvers::bitwuzla

#endif  // XLS_SOLVERS_BITWUZLA_IR_TRANSLATOR_H_
