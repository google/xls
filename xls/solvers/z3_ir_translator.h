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

#ifndef XLS_TOOLS_Z3_IR_TRANSLATOR_H_
#define XLS_TOOLS_Z3_IR_TRANSLATOR_H_

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xls/common/logging/logging.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/function.h"
#include "xls/ir/nodes.h"
#include "xls/solvers/z3_utils.h"
#include "../z3/src/api/z3.h"

namespace xls {
namespace solvers {
namespace z3 {

// Kinds of predicates we can compute about a subject node.
enum class PredicateKind {
  kEqualToZero,
  kNotEqualToZero,
  kEqualToNode,
};

// Translates a function into its Z3 equivalent bit-vector circuit for use in
// theorem proving.
class IrTranslator : public DfsVisitorWithDefault {
 public:
  // Creates a translator and uses it to translate the given function into a Z3
  // AST.
  static absl::StatusOr<std::unique_ptr<IrTranslator>> CreateAndTranslate(
      Function* function);

  // Translates the given function into a Z3 AST using a preexisting context
  // (i.e., that used by another Z3Translator). This binds the given function
  // to use the specified (already translated) parameters. This is to enable two
  // versions of the same function to be compared against the same inputs,
  // usually for equivalence checking.
  static absl::StatusOr<std::unique_ptr<IrTranslator>> CreateAndTranslate(
      Z3_context ctx, Function* function,
      absl::Span<const Z3_ast> imported_params);
  ~IrTranslator() override;

  // Sets the amount of time to allow Z3 to execute before aborting.
  void SetTimeout(absl::Duration timeout);

  // Returns the Z3 value (or set of values) corresponding to the given Node.
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

  // Converts three AST nodes (u1:sign, u8:bexp, u23:sfd) to a Z3-internal
  // floating-point number.
  absl::StatusOr<Z3_ast> ToFloat32(absl::Span<const Z3_ast> nodes);

  // Same as above, but for a tuple-typed value.
  absl::StatusOr<Z3_ast> ToFloat32(Z3_ast tuple);

  Z3_context ctx() { return ctx_; }

  // DfsVisitorWithDefault override decls.
  absl::Status DefaultHandler(Node* node) override;
  absl::Status HandleAdd(BinOp* add) override;
  absl::Status HandleAfterAll(AfterAll* after_all) override;
  absl::Status HandleArray(Array* array) override;
  absl::Status HandleArrayIndex(ArrayIndex* array_index) override;
  absl::Status HandleArrayUpdate(ArrayUpdate* array_update) override;
  absl::Status HandleArrayConcat(ArrayConcat* array_concat) override;
  absl::Status HandleArraySlice(ArraySlice* array_slice) override;
  absl::Status HandleBitSlice(BitSlice* bit_slice) override;
  absl::Status HandleBitSliceUpdate(BitSliceUpdate* update) override;
  absl::Status HandleConcat(Concat* concat) override;
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
  absl::Status HandleNot(UnOp* not_op) override;
  absl::Status HandleParam(Param* param) override;
  absl::Status HandleOneHot(OneHot* one_hot) override;
  absl::Status HandleOneHotSel(OneHotSelect* one_hot) override;
  absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce) override;
  absl::Status HandleReverse(UnOp* reverse) override;
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
  absl::Status HandleSub(BinOp* sub) override;
  absl::Status HandleTuple(Tuple* tuple) override;
  absl::Status HandleTupleIndex(TupleIndex* tuple_index) override;
  absl::Status HandleUGe(CompareOp* ge) override;
  absl::Status HandleUGt(CompareOp* gt) override;
  absl::Status HandleULe(CompareOp* le) override;
  absl::Status HandleULt(CompareOp* lt) override;
  absl::Status HandleUMul(ArithOp* mul) override;
  absl::Status HandleZeroExtend(ExtendOp* zero_ext) override;

  Function* xls_function() { return xls_function_; }

 private:
  IrTranslator(Z3_config config, Function* xls_function);

  IrTranslator(Z3_context ctx, Function* xls_function,
               absl::Span<const Z3_ast> imported_params);

  // Returns the index with the proper bitwidth for the given array_type.
  Z3_ast GetAsFormattedArrayIndex(Z3_ast index, ArrayType* array_type);

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
  absl::Status HandleNary(OpT* op, FnT f, bool invert_result);
  template <typename FnT>
  absl::Status HandleShift(BinOp* shift, FnT f);
  template <typename FnT>
  absl::Status HandleUnary(UnOp* op, FnT f);

  // Recursive call to translate XLS literals into Z3 form.
  absl::StatusOr<Z3_ast> TranslateLiteralValue(Type* type, const Value& value);

  // Common multiply handling.
  void HandleMul(ArithOp* mul, bool is_signed);

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

  // Handles the translation of the given unary op using the AbstractEvaluator.
  absl::Status HandleUnaryViaAbstractEval(Node* op);

  // Converts a XLS param decl into a Z3 param type.
  absl::StatusOr<Z3_ast> CreateZ3Param(Type* type,
                                       absl::string_view param_name);

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

  Z3_config config_;
  Z3_context ctx_;

  // True if this is translating a function called from another, in which case
  // we shouldn't delete our context, etc.!
  bool borrowed_context_;
  absl::flat_hash_map<const Node*, Z3_ast> translations_;
  // Params specified in the context-borrowing CreateAndTranslate() builder.
  // Parameters already translated in a separate function traversal that should
  // be used as this translation's parameter set.
  absl::optional<absl::Span<const Z3_ast>> imported_params_;
  Function* xls_function_;
};

// Describes a predicate to compute about a subject node in an XLS IR function.
class Predicate {
 public:
  static Predicate EqualTo(Node* node) {
    return Predicate(PredicateKind::kEqualToNode, node);
  }
  static Predicate EqualToZero() {
    return Predicate(PredicateKind::kEqualToZero);
  }
  static Predicate NotEqualToZero() {
    return Predicate(PredicateKind::kNotEqualToZero);
  }

  PredicateKind kind() const { return kind_; }
  Node* node() const {
    XLS_CHECK(node_.has_value());
    return node_.value();
  }

  std::string ToString() const;

 private:
  explicit Predicate(PredicateKind kind) : kind_(kind) {}
  Predicate(PredicateKind kind, Node* node) : kind_(kind), node_(node) {}

  PredicateKind kind_;
  absl::optional<Node*> node_;
};

// Attempts to prove node "subject" in function "f" satisfies the given
// predicate (over all possible inputs) within the duration "timeout".
absl::StatusOr<bool> TryProve(Function* f, Node* subject, Predicate p,
                              absl::Duration timeout);

}  // namespace z3
}  // namespace solvers
}  // namespace xls

#endif  // XLS_TOOLS_Z3_IR_TRANSLATOR_H_
