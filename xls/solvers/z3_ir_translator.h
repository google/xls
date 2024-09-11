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

// API for turning XLS IR computations into Z3 solver form (so we can
// compute/query formal properties of nodes in the XLS IR).

#ifndef XLS_SOLVERS_Z3_IR_TRANSLATOR_H_
#define XLS_SOLVERS_Z3_IR_TRANSLATOR_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "external/z3/src/api/z3.h"  // IWYU pragma: keep
#include "external/z3/src/api/z3_api.h"

namespace xls {
namespace solvers {
namespace z3 {

// Kinds of predicates we can compute about a subject node.
enum class PredicateKind : uint8_t {
  kEqualToZero,
  kNotEqualToZero,
  kEqualToNode,
  kUnsignedGreaterOrEqual,  // vs some given (constant) value
  kUnsignedLessOrEqual,     // vs some given (constant) value
};

// Translates a function into its Z3 equivalent bit-vector circuit for use in
// theorem proving.
class IrTranslator : public DfsVisitorWithDefault {
 public:
  // Creates a translator and uses it to translate the given function into a Z3
  // AST. The `allow_unsupported` option will cause unsupported ops to be
  // translated into fresh variables of the appropriate type.
  static absl::StatusOr<std::unique_ptr<IrTranslator>> CreateAndTranslate(
      FunctionBase* source, bool allow_unsupported = false);

  // Translates the given function into a Z3 AST using a preexisting context
  // (i.e., that used by another Z3Translator). This binds the given function
  // to use the specified (already translated) parameters. This is to enable two
  // versions of the same function to be compared against the same inputs,
  // usually for equivalence checking.
  static absl::StatusOr<std::unique_ptr<IrTranslator>> CreateAndTranslate(
      Z3_context ctx, FunctionBase* function_base,
      absl::Span<const Z3_ast> imported_params, bool allow_unsupported = false);

  // Translates the given node into a Z3 AST using a preexisting context
  // (i.e., that used by another Z3Translator).
  static absl::StatusOr<std::unique_ptr<IrTranslator>> CreateAndTranslate(
      Z3_context ctx, Node* source, bool allow_unsupported = false);

  ~IrTranslator() override;

  // Sets the amount of time to allow Z3 to execute before aborting.
  void SetTimeout(absl::Duration timeout);

  // Sets the amount of "solver resources" Z3 can use before aborting.
  //
  // Useful for reproducible termination, since timeout is not reproducible.
  void SetRlimit(int64_t rlimit);

  // Returns the Z3 value (or set of values) corresponding to the given Node.
  // Translates if the translation is not yet stored.
  Z3_ast GetTranslation(const Node* source);

  // Re-translates the function from scratch, using fixed mappings for the
  // values in "replacements", i.e., when any node in "replacements" is
  // encountered, the fixed Z3_ast is used instead of using a translation from
  // the original IR.
  absl::Status Retranslate(
      const absl::flat_hash_map<const Node*, Z3_ast>& replacements);

  // Convenience version for the above for the function return Node.
  Z3_ast GetReturnNode();

  // Returns the kind (bit vector, tuple, function decl, etc.) of a Z3 sort.
  Z3_sort_kind GetValueKind(Z3_ast value);

  // Converts the "bits" value into a corresponding Z3 bitvector literal-valued
  // node.
  absl::StatusOr<Z3_ast> TranslateLiteralBits(const Bits& bits);

  // "Flattens" the given value to individual bits and returns the associated
  // array. For example, flattening a tuple of two 5-bit values will return a
  // 10-entry span.
  // If little-endian is true, then for each leaf "Bits"-type element, the
  // output will have its least-significant element in the lowest index.
  std::vector<Z3_ast> FlattenValue(Type* type, Z3_ast value,
                                   bool little_endian = false);

  // Does the opposite of flattening a value - takes a span of flat bits, and
  // reconstructs them into a single value of the specified type.
  // If little-endian is true, then for each leaf "Bits"-type element, the input
  // will be assumed to have the least-significant element in the lowest index.
  Z3_ast UnflattenZ3Ast(Type* type, absl::Span<const Z3_ast> flat,
                        bool little_endian = false);

  // Floating-point routines.
  // Returns a zero-valued [positive] floating-point value of the specified
  // sort.
  Z3_ast FloatZero(Z3_sort sort);

  // Flushes the given floating-point value to 0 if it's a subnormal value.
  absl::StatusOr<Z3_ast> FloatFlushSubnormal(Z3_ast value);

  // Converts three AST nodes (u1:sign, u8:bexp, u23:fraction) to a Z3-internal
  // floating-point number.
  absl::StatusOr<Z3_ast> ToFloat32(absl::Span<const Z3_ast> nodes);

  // Same as above, but for a tuple-typed value.
  absl::StatusOr<Z3_ast> ToFloat32(Z3_ast tuple);

  Z3_context ctx() { return ctx_; }

  // DfsVisitorWithDefault override decls.
  absl::Status DefaultHandler(Node* node) override;
  absl::Status HandleAdd(BinOp* add) override;
  absl::Status HandleAfterAll(AfterAll* after_all) override;
  absl::Status HandleMinDelay(MinDelay* min_delay) override;
  absl::Status HandleArray(Array* array) override;
  absl::Status HandleArrayIndex(ArrayIndex* array_index) override;
  absl::Status HandleArrayUpdate(ArrayUpdate* array_update) override;
  absl::Status HandleArrayConcat(ArrayConcat* array_concat) override;
  absl::Status HandleArraySlice(ArraySlice* array_slice) override;
  absl::Status HandleBitSlice(BitSlice* bit_slice) override;
  absl::Status HandleBitSliceUpdate(BitSliceUpdate* update) override;
  absl::Status HandleConcat(Concat* concat) override;
  absl::Status HandleDecode(Decode* decode) override;
  absl::Status HandleDynamicBitSlice(
      DynamicBitSlice* dynamic_bit_slice) override;
  absl::Status HandleEncode(Encode* encode) override;
  absl::Status HandleEq(CompareOp* eq) override;
  absl::Status HandleIdentity(UnOp* identity) override;
  absl::Status HandleLiteral(Literal* literal) override;
  absl::Status HandleNaryAnd(NaryOp* and_op) override;
  absl::Status HandleNaryNand(NaryOp* nand_op) override;
  absl::Status HandleNaryNor(NaryOp* nor_op) override;
  absl::Status HandleNaryOr(NaryOp* or_op) override;
  absl::Status HandleNaryXor(NaryOp* xor_op) override;
  absl::Status HandleNe(CompareOp* ne) override;
  absl::Status HandleNeg(UnOp* neg) override;
  absl::Status HandleNext(Next* next) override;
  absl::Status HandleNot(UnOp* not_op) override;
  absl::Status HandleParam(Param* param) override;
  absl::Status HandleOneHot(OneHot* one_hot) override;
  absl::Status HandleOneHotSel(OneHotSelect* one_hot) override;
  absl::Status HandlePrioritySel(PrioritySelect* sel) override;
  absl::Status HandleAndReduce(BitwiseReductionOp* and_reduce) override;
  absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce) override;
  absl::Status HandleXorReduce(BitwiseReductionOp* xor_reduce) override;
  absl::Status HandleReverse(UnOp* reverse) override;
  absl::Status HandleSDiv(BinOp* div) override;
  absl::Status HandleSMod(BinOp* mod) override;
  absl::Status HandleSel(Select* sel) override;
  absl::Status HandleSignExtend(ExtendOp* sign_ext) override;
  absl::Status HandleSGe(CompareOp* sge) override;
  absl::Status HandleShll(BinOp* shll) override;
  absl::Status HandleShra(BinOp* shra) override;
  absl::Status HandleShrl(BinOp* shrl) override;
  absl::Status HandleSGt(CompareOp* gt) override;
  absl::Status HandleSLe(CompareOp* le) override;
  absl::Status HandleSLt(CompareOp* lt) override;
  absl::Status HandleSMul(ArithOp* mul) override;
  absl::Status HandleSMulp(PartialProductOp* mul) override;
  absl::Status HandleSub(BinOp* sub) override;
  absl::Status HandleTuple(Tuple* tuple) override;
  absl::Status HandleTupleIndex(TupleIndex* tuple_index) override;
  absl::Status HandleUDiv(BinOp* div) override;
  absl::Status HandleUMod(BinOp* mod) override;
  absl::Status HandleUGe(CompareOp* ge) override;
  absl::Status HandleUGt(CompareOp* gt) override;
  absl::Status HandleULe(CompareOp* le) override;
  absl::Status HandleULt(CompareOp* lt) override;
  absl::Status HandleUMul(ArithOp* mul) override;
  absl::Status HandleUMulp(PartialProductOp* mul) override;
  absl::Status HandleZeroExtend(ExtendOp* zero_ext) override;
  absl::Status HandleInvoke(Invoke* invoke) override;

  FunctionBase* xls_function() { return xls_function_; }

 private:
  IrTranslator(Z3_config config, FunctionBase* source);

  IrTranslator(Z3_context ctx, FunctionBase* source,
               std::optional<absl::Span<const Z3_ast>> imported_params);

  // Gets the bit count associated with the bit-vector-sort Z3 node "arg".
  // (Arg must be known to be of bit-vector sort.)
  int64_t GetBvBitCount(Z3_ast arg);

  // Gets the translation of the given node. Crashes if "node" has not already
  // been translated. The following functions are short-cuts that assume the
  // node is of the indicated type.
  Z3_ast GetValue(const Node* node);
  Z3_ast GetBitVec(Node* node);
  Z3_ast GetArray(Node* node);
  Z3_ast GetTuple(Node* node);

  // Does the _actual_ work of processing binary, nary, etc. operations.
  template <typename OpT, typename FnT>
  absl::Status HandleBinary(OpT* op, FnT f);
  template <typename OpT, typename FnT>
  absl::Status HandleNary(OpT* op, FnT f, bool invert_result,
                          bool skip_empty_operands = false);
  template <typename FnT>
  absl::Status HandleShift(BinOp* shift, FnT f);
  template <typename FnT>
  absl::Status HandleUnary(Node* op, FnT f);

  // Recursive call to translate XLS literals into Z3 form.
  // The `has_nonconcat_uses` parameter is used for checking whether the literal
  // we're trying to translate contains a zero-width bitvector that has
  // nontrivial uses in the IR graph.
  absl::StatusOr<Z3_ast> TranslateLiteralValue(bool has_nonconcat_uses,
                                               Type* type, const Value& value);

  // Common multiply handling.
  void HandleMul(ArithOp* mul, bool is_signed);
  void HandleMulp(PartialProductOp* mul, bool is_signed);

  // Translates a OneHotSelect or Sel node whose (non-selector) operands are
  // Tuple typed. Accepts a function to actually call into the AbstractEvaluator
  // for that node. "FlatValue" is a helper to represent a value as a
  // vector of individual Z3 bits.
  using FlatValue = std::vector<Z3_ast>;
  template <typename NodeT>
  absl::Status HandleSelect(
      NodeT* node, std::function<FlatValue(const FlatValue& selector,
                                           const std::vector<FlatValue>& cases)>
                       evaluator);

  // Turn a single possibly compound typed z3 value into a leaf-type-tree of
  // bits values.
  absl::StatusOr<LeafTypeTree<Z3_ast>> ToLeafTypeTree(Node* node) {
    return ToLeafTypeTree(node->GetType(), GetValue(node));
  }
  // Turn a single possibly compound typed z3 value into a leaf-type-tree of
  // bits values.
  absl::StatusOr<LeafTypeTree<Z3_ast>> ToLeafTypeTree(Type* type, Z3_ast ast);
  // Turn a shattered z3 leaf-type-tree back into a single z3 value.
  absl::StatusOr<Z3_ast> FromLeafTypeTree(LeafTypeTreeView<Z3_ast> ast);

  // Get a single element from the LeafTypeTree representation of the given z3
  // value without actually creating the entire tree.
  absl::StatusOr<Z3_ast> GetLttElement(Type* type, Z3_ast value,
                                       absl::Span<int64_t const> index);

  // Handles the translation of the given unary op using the AbstractEvaluator.
  absl::Status HandleUnaryViaAbstractEval(Node* op);

  // Converts a XLS param decl into a Z3 param type.
  absl::StatusOr<Z3_ast> CreateZ3Param(Type* type, std::string_view param_name);

  // Records the mapping of the specified XLS IR node to Z3 value.
  void NoteTranslation(Node* node, Z3_ast translated);

  // Creates a Z3 tuple from the given XLS type or Z3 sort and Z3 elements.
  Z3_ast CreateTuple(Type* tuple_type, absl::Span<const Z3_ast> elements);
  Z3_ast CreateTuple(Z3_sort tuple_sort, absl::Span<const Z3_ast> elements);

  // Creates a Z3 array from the given XLS type and Z3 elements.
  Z3_ast CreateArray(ArrayType* type, absl::Span<const Z3_ast> elements);

  // Extracts a value from an array type.
  Z3_ast GetArrayElement(ArrayType* type, Z3_ast array, Z3_ast index);

  // Conditionally (based on 'cond') replaces the element in 'array' indexed by
  // 'indices' with 'value' and returns the result. 'array' may be a
  // multi-dimensional array in which case 'indices' may have more than one
  // element. In the sequence of indices 'indices' the first element is the
  // outermost index as in ArrayIndex and ArrayUpdate. 'type' is the
  // XLS type corresponding to 'array'.
  Z3_ast UpdateArrayElement(Type* type, Z3_ast array, Z3_ast value, Z3_ast cond,
                            absl::Span<const Z3_ast> indices);

  // Z3 version of xls::ZeroOfType() - creates a zero-valued element.
  Z3_ast ZeroOfSort(Z3_sort sort);

  // Get a new Z3_symbol.
  Z3_symbol GetNewSymbol();

  Z3_config config_;
  Z3_context ctx_;

  // Do we allow unsupported ops by translating them into fresh variables?
  bool allow_unsupported_;
  // True if this is translating a function called from another, in which case
  // we shouldn't delete our context, etc.!
  bool borrowed_context_;
  absl::flat_hash_map<const Node*, Z3_ast> translations_;
  // Params specified in the context-borrowing CreateAndTranslate() builder.
  // Parameters already translated in a separate function traversal that should
  // be used as this translation's parameter set.
  std::optional<absl::Span<const Z3_ast>> imported_params_;
  FunctionBase* xls_function_;
  int current_symbol_;
};

// Describes a predicate to compute about a subject node in an XLS IR function.
//
// Note: predicates currently implicitly refer to an (unreferenced) subject,
// like a return value, so you can make fairly context-free constructs like
// `Predicate::EqualToZero()`. (See `PredicateOfNode` for ways to explicitly
// provide a subject node for the predicate to act upon.)
class Predicate {
 public:
  static Predicate IsEqualTo(Node* other);
  static Predicate EqualToZero();
  static Predicate NotEqualToZero();
  static Predicate UnsignedGreaterOrEqual(Bits lower_bound);
  static Predicate UnsignedLessOrEqual(Bits upper_bound);

  PredicateKind kind() const { return kind_; }

  // For predicates that refer to another node; e.g.
  // `Predicate::IsEqualTo(other)`, returns the node the predicate is comparing
  // to (`other` in this example).
  Node* node() const {
    CHECK(node_.has_value());
    return node_.value();
  }

  std::string ToString() const;

  // For predicates that have a bits value as part of the predicate payload,
  // returns the bits value; e.g. for
  // `Predicate::UnsignedGreaterOrEqual(my_bits)` returns the value of
  // `my_bits`.
  const Bits& value() const {
    CHECK(value_.has_value());
    return value_.value();
  }

 private:
  explicit Predicate(PredicateKind kind);
  Predicate(PredicateKind kind, Node* node);
  Predicate(PredicateKind kind, Node* node, Bits value);

  PredicateKind kind_;
  std::optional<Node*> node_;
  std::optional<Bits> value_;
};

// Predicates generally don't encode a subject, they say things like "should be
// greater than zero" but the subject is implicit, e.g. a return value.
//
// This struct wraps a predicate with an explicit subject (that must be present
// inside of the associated function).
struct PredicateOfNode {
  Node* subject;
  Predicate p;
};

using ProvenTrue = std::true_type;
struct ProvenFalse {
  // If available, a set of Values for the function's Params that implement the
  // counterexample; otherwise, an absl::Status documenting the failure to
  // translate the counterexample.
  absl::StatusOr<absl::flat_hash_map<const Param*, Value>> counterexample =
      absl::UnimplementedError("no counterexample analysis attempted");

  // Typically contains the encoded Z3 solver result (which usually includes the
  // counterexample).
  std::string message;
};
using ProverResult = std::variant<ProvenTrue, ProvenFalse>;

template <typename Sink>
void AbslStringify(Sink& sink, const ProverResult& p) {
  if (std::holds_alternative<ProvenTrue>(p)) {
    absl::Format(&sink, "[ProvenTrue]");
    return;
  }
  absl::Format(&sink, "[ProvenFalse: %s]", std::get<ProvenFalse>(p).message);
}

// Attempts to prove the conjunction of "terms". "terms" refers to predicates on
// nodes within function "f". Returns true iff "terms" can be proven true in
// conjunction (over all possible inputs) within the given "timeout" or
// "rlimit".
absl::StatusOr<ProverResult> TryProveConjunction(
    FunctionBase* f, absl::Span<const PredicateOfNode> terms,
    absl::Duration timeout, bool allow_unsupported = false);
absl::StatusOr<ProverResult> TryProveConjunction(
    FunctionBase* f, absl::Span<const PredicateOfNode> terms, int64_t rlimit,
    bool allow_unsupported = false);

// Attempts to prove the disjunction of "terms". "terms" refers to predicates on
// nodes within function "f". Returns true iff "terms" can be proven true in
// disjunction (over all possible inputs) within the given "timeout" or
// "rlimit".
absl::StatusOr<ProverResult> TryProveDisjunction(
    FunctionBase* f, absl::Span<const PredicateOfNode> terms,
    absl::Duration timeout, bool allow_unsupported = false);
absl::StatusOr<ProverResult> TryProveDisjunction(
    FunctionBase* f, absl::Span<const PredicateOfNode> terms, int64_t rlimit,
    bool allow_unsupported = false);

// Attempts to prove node "subject" in function "f" satisfies the given
// predicate (over all possible inputs) within the duration "timeout" or the
// "rlimit".
//
// This offers a simpler subset of the functionality of TryProveConjunction
// above.
absl::StatusOr<ProverResult> TryProve(FunctionBase* f, Node* subject,
                                      Predicate p, absl::Duration timeout,
                                      bool allow_unsupported = false);
absl::StatusOr<ProverResult> TryProve(FunctionBase* f, Node* subject,
                                      Predicate p, int64_t rlimit,
                                      bool allow_unsupported = false);

}  // namespace z3
}  // namespace solvers
}  // namespace xls

#endif  // XLS_SOLVERS_Z3_IR_TRANSLATOR_H_
