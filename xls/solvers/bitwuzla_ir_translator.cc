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

#include "xls/solvers/bitwuzla_ir_translator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "bitwuzla/cpp/bitwuzla.h"
#include "bitwuzla/cpp/sat_solver.h"
#include "bitwuzla/cpp/terminator.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/stopwatch.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/number_parser.h"
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/solvers/solver.h"

namespace xls::solvers::bitwuzla {
namespace {

using ::bitwuzla::Kind;
using ::bitwuzla::Sort;
using ::bitwuzla::Term;
using ::bitwuzla::TermManager;

class XlsTerminator : public ::bitwuzla::Terminator {
 public:
  XlsTerminator(std::optional<absl::Duration> timeout,
                std::optional<int64_t> call_limit)
      : timeout_(timeout), call_limit_(call_limit) {
    if (timeout_.has_value()) {
      stopwatch_.emplace();
    }
    if (call_limit_.has_value()) {
      calls_ = 0;
    }
  }

  bool terminate() override {
    if (timeout_.has_value() && stopwatch_->GetElapsedTime() >= *timeout_) {
      return true;
    }
    if (call_limit_.has_value() && (++*calls_) >= *call_limit_) {
      return true;
    }
    return false;
  }

  std::optional<int64_t> call_limit() const { return call_limit_; }
  std::optional<absl::Duration> timeout() const { return timeout_; }

  std::optional<int64_t> calls() const { return calls_; }

  void reset(std::optional<int64_t> call_limit) {
    if (call_limit.has_value()) {
      calls_ = 0;
      call_limit_ = call_limit;
    } else {
      calls_.reset();
      call_limit_ = std::nullopt;
    }
  }

 private:
  std::optional<absl::Duration> timeout_;
  std::optional<int64_t> call_limit_;

  std::optional<Stopwatch> stopwatch_;
  std::optional<int64_t> calls_ = 0;
};

Term MakeConjunction(TermManager& tm, absl::Span<const Term> terms) {
  if (terms.empty()) {
    return tm.mk_true();
  }
  if (terms.size() == 1) {
    return terms[0];
  }
  return tm.mk_term(Kind::AND, std::vector<Term>(terms.begin(), terms.end()));
}

absl::Status RequireBitVectorSort(const Term& t) {
  if (t.sort().is_array()) {
    return absl::InvalidArgumentError(
        "Cannot evaluate predicate directly on array sort");
  }
  return absl::OkStatus();
}

}  // namespace

IrTranslator::IrTranslator(std::unique_ptr<TermManager> owned_tm,
                           FunctionBase* source, bool allow_unsupported)
    : owned_tm_(std::move(owned_tm)),
      tm_(*owned_tm_),
      xls_function_(source),
      allow_unsupported_(allow_unsupported) {}

IrTranslator::IrTranslator(TermManager& tm, FunctionBase* source,
                           bool allow_unsupported)
    : tm_(tm), xls_function_(source), allow_unsupported_(allow_unsupported) {}

IrTranslator::~IrTranslator() = default;

absl::StatusOr<std::unique_ptr<IrTranslator>> IrTranslator::CreateAndTranslate(
    FunctionBase* source, bool allow_unsupported) {
  auto owned_tm = std::make_unique<TermManager>();
  auto translator = std::unique_ptr<IrTranslator>(
      new IrTranslator(std::move(owned_tm), source, allow_unsupported));
  XLS_RETURN_IF_ERROR(source->Accept(translator.get()));
  return translator;
}

absl::StatusOr<std::unique_ptr<IrTranslator>> IrTranslator::CreateAndTranslate(
    TermManager& tm, FunctionBase* source,
    absl::Span<const Term> imported_params, bool allow_unsupported) {
  auto translator = std::unique_ptr<IrTranslator>(
      new IrTranslator(tm, source, allow_unsupported));
  auto params = source->params();
  XLS_RET_CHECK_EQ(params.size(), imported_params.size());
  for (int64_t i = 0; i < params.size(); ++i) {
    translator->NoteTranslation(params[i], imported_params[i]);
  }
  XLS_RETURN_IF_ERROR(source->Accept(translator.get()));
  return translator;
}

absl::StatusOr<std::unique_ptr<IrTranslator>> IrTranslator::CreateAndTranslate(
    TermManager& tm, Node* source, bool allow_unsupported) {
  auto translator = std::unique_ptr<IrTranslator>(
      new IrTranslator(tm, source->function_base(), allow_unsupported));
  XLS_RETURN_IF_ERROR(source->Accept(translator.get()));
  return translator;
}

void IrTranslator::SetTimeout(std::optional<absl::Duration> timeout) {
  if (timeout.has_value() && (timeout.value() <= absl::ZeroDuration() ||
                              timeout.value() >= absl::InfiniteDuration())) {
    timeout_ = std::nullopt;
  } else {
    timeout_ = timeout;
  }
}

void IrTranslator::SetDeterministicLimit(std::optional<int64_t> limit) {
  if (limit.has_value() && limit.value() <= 0) {
    limit_ = std::nullopt;
  } else {
    limit_ = limit;
  }
}

Term IrTranslator::GetTranslation(const Node* source) {
  auto it = translations_.find(source);
  CHECK(it != translations_.end())
      << "Node not translated: " << source->ToString();
  return it->second;
}

Term IrTranslator::GetReturnNode() {
  CHECK_NE(xls_function_, nullptr);
  CHECK(xls_function_->IsFunction());
  return GetTranslation(xls_function_->AsFunctionOrDie()->return_value());
}

absl::Status IrTranslator::Retranslate(
    const absl::flat_hash_map<const Node*, Term>& replacements) {
  ResetVisitedState();
  translations_ = replacements;
  return xls_function_->Accept(this);
}

std::string IrTranslator::GetNewSymbol(std::string_view prefix) {
  return absl::StrCat(prefix, symbol_count_++);
}

Sort IrTranslator::TypeToSort(const Type* type) {
  switch (type->kind()) {
    case TypeKind::kToken:
      return tm_.mk_bv_sort(1);
    case TypeKind::kBits: {
      int64_t bc = type->AsBitsOrDie()->bit_count();
      return tm_.mk_bv_sort(bc == 0 ? 1 : bc);
    }
    case TypeKind::kTuple: {
      int64_t fbc = type->GetFlatBitCount();
      return tm_.mk_bv_sort(fbc == 0 ? 1 : fbc);
    }
    case TypeKind::kArray: {
      const ArrayType* at = type->AsArrayOrDie();
      if (at->GetFlatBitCount() == 0) {
        return tm_.mk_bv_sort(1);
      }
      // A 1-element array has 0 index bits (`MinBitCountUnsigned(0) == 0`);
      // wrapping it in a 1-bit `ArraySort` ({0, 1}) would invent unreachable
      // OOB sub-array slots across nested dimensions.
      if (at->size() == 1) {
        return TypeToSort(at->element_type());
      }
      return tm_.mk_array_sort(GetArrayIndexSort(at),
                               TypeToSort(at->element_type()));
    }
  }
  LOG(FATAL) << "Unknown type kind: " << type->ToString();
}

Sort IrTranslator::GetArrayIndexSort(const ArrayType* type) {
  int64_t bc = Bits::MinBitCountUnsigned(type->empty() ? 0 : type->size() - 1);
  return tm_.mk_bv_sort(bc == 0 ? 1 : bc);
}

Term IrTranslator::GetAsFormattedArrayIndex(int64_t index,
                                            const ArrayType* type) {
  Sort sort = GetArrayIndexSort(type);
  return tm_.mk_bv_value_uint64(sort, index);
}

Term IrTranslator::GetAsFormattedArrayIndex(Term index, const ArrayType* type,
                                            Term* oob) {
  Sort sort = GetArrayIndexSort(type);
  uint64_t target_width = sort.bv_size();
  uint64_t actual_width = index.sort().bv_size();
  const uint64_t max_idx_val =
      type->empty() ? 0 : static_cast<uint64_t>(type->size() - 1);
  Term max_idx = GetAsFormattedArrayIndex(max_idx_val, type);
  Term is_oob;
  Term formatted = index;
  if (actual_width < target_width) {
    formatted = tm_.mk_term(Kind::BV_ZERO_EXTEND, {index},
                            {target_width - actual_width});
    is_oob = tm_.mk_term(Kind::BV_UGT, {formatted, max_idx});
  } else {
    // Check OOB at full width before extracting low `target_width` bits.
    Term wide_max = tm_.mk_bv_value_uint64(index.sort(), max_idx_val);
    is_oob = tm_.mk_term(Kind::BV_UGT, {index, wide_max});
    if (actual_width > target_width) {
      formatted = tm_.mk_term(Kind::BV_EXTRACT, {index}, {target_width - 1, 0});
    }
  }
  if (oob != nullptr) {
    *oob = is_oob;
  }
  return tm_.mk_term(Kind::ITE, {is_oob, max_idx, formatted});
}

Term IrTranslator::TranslateLiteralBits(const Bits& bits) {
  int64_t bc = bits.bit_count();
  if (bc == 0) {
    return tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0);
  }
  if (bits.IsZero()) {
    return tm_.mk_bv_zero(tm_.mk_bv_sort(bc));
  }
  if (bits.IsAllOnes()) {
    return tm_.mk_bv_ones(tm_.mk_bv_sort(bc));
  }
  if (bits.FitsInUint64()) {
    return tm_.mk_bv_value_uint64(tm_.mk_bv_sort(bc), bits.ToUint64().value());
  }
  int64_t num_words = bits.bitmap().word_count();
  std::vector<Term> words;
  words.reserve(num_words);
  int64_t top_width = bc - (num_words - 1) * 64;
  words.push_back(tm_.mk_bv_value_uint64(tm_.mk_bv_sort(top_width),
                                         bits.bitmap().GetWord(num_words - 1)));
  for (int64_t i = num_words - 2; i >= 0; --i) {
    words.push_back(
        tm_.mk_bv_value_uint64(tm_.mk_bv_sort(64), bits.bitmap().GetWord(i)));
  }
  return ConcatN(words);
}

absl::StatusOr<Term> IrTranslator::TranslateLiteralValue(const Type* type,
                                                         const Value& val) {
  switch (type->kind()) {
    case TypeKind::kToken:
      return tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0);
    case TypeKind::kBits:
      return TranslateLiteralBits(val.bits());
    case TypeKind::kTuple: {
      const TupleType* tt = type->AsTupleOrDie();
      if (type->GetFlatBitCount() == 0) {
        return tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0);
      }
      std::vector<Term> elems;
      for (int64_t i = 0; i < tt->size(); ++i) {
        if (tt->element_type(i)->GetFlatBitCount() == 0) {
          continue;
        }
        XLS_ASSIGN_OR_RETURN(Term e, TranslateLiteralValue(tt->element_type(i),
                                                           val.elements()[i]));
        elems.push_back(FlattenToBv(e, tt->element_type(i)));
      }
      return ConcatN(elems);
    }
    case TypeKind::kArray: {
      const ArrayType* at = type->AsArrayOrDie();
      if (at->GetFlatBitCount() == 0) {
        return MakeZeroTerm(at);
      }
      std::vector<Term> elems;
      elems.reserve(at->size());
      for (int64_t i = 0; i < at->size(); ++i) {
        XLS_ASSIGN_OR_RETURN(
            Term elem,
            TranslateLiteralValue(at->element_type(), val.elements()[i]));
        elems.push_back(elem);
      }
      return BuildArrayFromElements(at, elems, "lit_arr");
    }
  }
  return absl::InvalidArgumentError("Unknown literal type");
}

Term IrTranslator::ZeroOfSort(const Sort& sort) {
  CHECK(sort.is_bv()) << "ZeroOfSort only supports bit-vector sorts";
  return tm_.mk_bv_value_uint64(sort, 0);
}

Term IrTranslator::MakeZeroTerm(const Type* type) {
  if (type->GetFlatBitCount() == 0) {
    return ZeroOfSort(tm_.mk_bv_sort(1));
  }
  if (!type->IsArray()) {
    return ZeroOfSort(TypeToSort(type));
  }
  const ArrayType* at = type->AsArrayOrDie();
  if (at->size() == 1) {
    return MakeZeroTerm(at->element_type());
  }
  std::vector<Term> elems(at->size(), MakeZeroTerm(at->element_type()));
  return BuildArrayFromElements(at, elems, "zero_arr");
}

Term IrTranslator::BuildArrayFromElements(const ArrayType* type,
                                          absl::Span<const Term> elements,
                                          std::string_view symbol_prefix) {
  if (type->GetFlatBitCount() == 0) {
    return MakeZeroTerm(type);
  }
  if (type->size() == 1) {
    CHECK_EQ(elements.size(), 1);
    return elements[0];
  }
  Sort asort = TypeToSort(type);
  // Avoid Bitwuzla CONST_ARRAY, which is unsupported under array equalities
  // and array ITEs; all valid indices [0, size) are stored explicitly.
  Term arr = tm_.mk_const(asort, GetNewSymbol(symbol_prefix));
  for (int64_t i = 0; i < elements.size(); ++i) {
    arr = tm_.mk_term(Kind::ARRAY_STORE,
                      {arr, GetAsFormattedArrayIndex(i, type), elements[i]});
  }
  return arr;
}

Term IrTranslator::OnesOfSort(const Sort& sort) {
  if (sort.is_bv()) {
    return tm_.mk_term(Kind::BV_NOT, {ZeroOfSort(sort)});
  }
  LOG(FATAL) << "Unsupported sort for ones";
}

Term IrTranslator::MinSignedOfSort(const Sort& sort) {
  CHECK(sort.is_bv());
  uint64_t w = sort.bv_size();
  Term sign = tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 1);
  if (w == 1) {
    return sign;
  }
  return tm_.mk_term(Kind::BV_CONCAT,
                     {sign, ZeroOfSort(tm_.mk_bv_sort(w - 1))});
}

Term IrTranslator::MaxSignedOfSort(const Sort& sort) {
  CHECK(sort.is_bv());
  uint64_t w = sort.bv_size();
  Term sign = tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0);
  if (w == 1) {
    return sign;
  }
  return tm_.mk_term(Kind::BV_CONCAT,
                     {sign, OnesOfSort(tm_.mk_bv_sort(w - 1))});
}

Term IrTranslator::ConcatN(const std::vector<Term>& args) {
  if (args.empty()) {
    return tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0);
  }
  if (args.size() == 1) {
    return args[0];
  }
  return tm_.mk_term(Kind::BV_CONCAT,
                     std::vector<Term>(args.begin(), args.end()));
}

Term IrTranslator::ConcatN(absl::Span<const Term> args) {
  if (args.empty()) {
    return tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0);
  }
  if (args.size() == 1) {
    return args[0];
  }
  return tm_.mk_term(Kind::BV_CONCAT,
                     std::vector<Term>(args.begin(), args.end()));
}

Term IrTranslator::FlattenToBv(Term term, const Type* type) {
  if (type->GetFlatBitCount() == 0) {
    return tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0);
  }
  if (!type->IsArray()) {
    return term;
  }
  const ArrayType* at = type->AsArrayOrDie();
  if (at->size() == 1) {
    return FlattenToBv(term, at->element_type());
  }
  std::vector<Term> elems;
  elems.reserve(at->size());
  for (int64_t i = 0; i < at->size(); ++i) {
    Term idx = GetAsFormattedArrayIndex(i, at);
    Term elem = tm_.mk_term(Kind::ARRAY_SELECT, {term, idx});
    elems.push_back(FlattenToBv(elem, at->element_type()));
  }
  return ConcatN(elems);
}

Term IrTranslator::UnflattenFromArrayBv(Term flat_bv, const Type* type) {
  if (type->GetFlatBitCount() == 0) {
    return MakeZeroTerm(type);
  }
  if (!type->IsArray()) {
    return flat_bv;
  }
  const ArrayType* at = type->AsArrayOrDie();
  if (at->size() == 1) {
    return UnflattenFromArrayBv(flat_bv, at->element_type());
  }
  int64_t elem_bits = at->element_type()->GetFlatBitCount();
  std::vector<Term> elems;
  elems.reserve(at->size());
  for (int64_t i = 0; i < at->size(); ++i) {
    int64_t low = (at->size() - 1 - i) * elem_bits;
    int64_t high = low + elem_bits - 1;
    Term slice =
        tm_.mk_term(Kind::BV_EXTRACT, {flat_bv},
                    {static_cast<uint64_t>(high), static_cast<uint64_t>(low)});
    elems.push_back(UnflattenFromArrayBv(slice, at->element_type()));
  }
  return BuildArrayFromElements(at, elems, "unflatten_arr");
}

Term IrTranslator::GetArrayElement(const ArrayType* type, Term array,
                                   Term index) {
  if (type->empty() || type->GetFlatBitCount() == 0) {
    return MakeZeroTerm(type->element_type());
  }
  if (type->size() == 1) {
    return array;
  }
  Term idx_formatted = GetAsFormattedArrayIndex(index, type);
  return tm_.mk_term(Kind::ARRAY_SELECT, {array, idx_formatted});
}

Term IrTranslator::UpdateArrayElement(const Type* type, Term array, Term value,
                                      Term cond,
                                      absl::Span<const Term> indices) {
  if (type->GetFlatBitCount() == 0) {
    return array;
  }
  if (indices.empty()) {
    return tm_.mk_term(Kind::ITE, {cond, value, array});
  }
  const ArrayType* at = type->AsArrayOrDie();
  Term oob;
  Term idx = GetAsFormattedArrayIndex(indices[0], at, &oob);
  Term in_bounds = tm_.mk_term(Kind::NOT, {oob});
  Term new_cond = tm_.mk_term(Kind::AND, {cond, in_bounds});
  if (at->size() == 1) {
    return UpdateArrayElement(at->element_type(), array, value, new_cond,
                              indices.subspan(1));
  }
  Term old_elem = tm_.mk_term(Kind::ARRAY_SELECT, {array, idx});
  Term updated_elem = UpdateArrayElement(at->element_type(), old_elem, value,
                                         new_cond, indices.subspan(1));
  Term new_arr = tm_.mk_term(Kind::ARRAY_STORE, {array, idx, updated_elem});
  return tm_.mk_term(Kind::ITE, {new_cond, new_arr, array});
}

Term IrTranslator::ComputeNe(Term lhs, Term rhs, const Type* type) {
  if (type->GetFlatBitCount() == 0) {
    return tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0);
  }
  Term flat_l = FlattenToBv(lhs, type);
  Term flat_r = FlattenToBv(rhs, type);
  uint64_t w = flat_l.sort().bv_size();
  if (w == 0) {
    return tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0);
  }
  Term eq = tm_.mk_term(Kind::EQUAL, {flat_l, flat_r});
  return tm_.mk_term(Kind::ITE,
                     {eq, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0),
                      tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 1)});
}

::bitwuzla::Term IrTranslator::CoerceShiftAmount(::bitwuzla::Term val,
                                                 Node* shamt_node) {
  uint64_t val_width = val.sort().bv_size();
  if (shamt_node->GetType()->GetFlatBitCount() == 0) {
    return tm_.mk_bv_value_uint64(val.sort(), 0);
  }
  Term shamt = GetTranslation(shamt_node);
  uint64_t shamt_width = shamt.sort().bv_size();
  if (val_width == shamt_width) {
    return shamt;
  }
  if (shamt_width < val_width) {
    return tm_.mk_term(Kind::BV_ZERO_EXTEND, {shamt},
                       {val_width - shamt_width});
  }
  Term max_val = tm_.mk_bv_value_uint64(shamt.sort(), val_width);
  Term oob = tm_.mk_term(Kind::BV_UGE, {shamt, max_val});
  Term trunc = tm_.mk_term(Kind::BV_EXTRACT, {shamt}, {val_width - 1, 0});
  return tm_.mk_term(
      Kind::ITE, {oob, tm_.mk_bv_value_uint64(val.sort(), val_width), trunc});
}

absl::StatusOr<Term> IrTranslator::GetLttElement(
    const Type* type, Term value, absl::Span<const int64_t> index) {
  if (index.empty()) {
    return value;
  }
  if (type->IsArray()) {
    const ArrayType* at = type->AsArrayOrDie();
    Term idx_term = GetAsFormattedArrayIndex(index[0], at);
    Term elem = GetArrayElement(at, value, idx_term);
    return GetLttElement(at->element_type(), elem, index.subspan(1));
  }
  CHECK(type->IsTuple());
  const TupleType* tt = type->AsTupleOrDie();
  int64_t k = index[0];
  int64_t low = 0;
  for (int64_t j = tt->size() - 1; j > k; --j) {
    low += tt->element_type(j)->GetFlatBitCount();
  }
  int64_t fb = tt->element_type(k)->GetFlatBitCount();
  if (fb == 0) {
    return MakeZeroTerm(tt->element_type(k));
  }
  Term slice = tm_.mk_term(
      Kind::BV_EXTRACT, {value},
      {static_cast<uint64_t>(low + fb - 1), static_cast<uint64_t>(low)});
  return GetLttElement(tt->element_type(k),
                       UnflattenFromArrayBv(slice, tt->element_type(k)),
                       index.subspan(1));
}

absl::StatusOr<LeafTypeTree<Term>> IrTranslator::ToLeafTypeTree(
    const Node* node) {
  return ToLeafTypeTree(node->GetType(), GetTranslation(node));
}

absl::StatusOr<LeafTypeTree<Term>> IrTranslator::ToLeafTypeTree(Type* type,
                                                                Term term) {
  return LeafTypeTree<Term>::CreateFromFunction(
      type,
      [&](Type* lt, absl::Span<const int64_t> idx) -> absl::StatusOr<Term> {
        return GetLttElement(type, term, idx);
      });
}

absl::StatusOr<Term> IrTranslator::TernaryToConstraint(Term target_val,
                                                       TernarySpan ternary) {
  int64_t bc = ternary.size();
  if (bc == 0) {
    return tm_.mk_true();
  }
  std::vector<Term> conds;
  for (int64_t i = 0; i < bc; ++i) {
    if (ternary_ops::IsUnknown(ternary[i])) {
      continue;
    }
    Term ext =
        tm_.mk_term(Kind::BV_EXTRACT, {target_val},
                    {static_cast<uint64_t>(i), static_cast<uint64_t>(i)});
    Term exp = tm_.mk_bv_value_uint64(
        tm_.mk_bv_sort(1), ternary[i] == TernaryValue::kKnownOne ? 1 : 0);
    conds.push_back(tm_.mk_term(Kind::EQUAL, {ext, exp}));
  }
  if (conds.empty()) {
    return tm_.mk_true();
  }
  if (conds.size() == 1) {
    return conds[0];
  }
  return tm_.mk_term(Kind::AND, conds);
}

absl::StatusOr<Term> IrTranslator::IntervalSetToConstraint(
    Term target_val, const IntervalSet& intervals) {
  if (intervals.IsMaximal()) {
    return tm_.mk_true();
  }
  if (intervals.IsEmpty()) {
    return tm_.mk_false();
  }
  std::vector<Term> int_conds;
  for (const Interval& interval : intervals.Intervals()) {
    std::vector<Term> bounds;
    if (!interval.LowerBound().IsZero()) {
      bounds.push_back(tm_.mk_term(
          Kind::BV_UGE,
          {target_val, TranslateLiteralBits(interval.LowerBound())}));
    }
    if (!interval.UpperBound().IsAllOnes()) {
      bounds.push_back(tm_.mk_term(
          Kind::BV_ULE,
          {target_val, TranslateLiteralBits(interval.UpperBound())}));
    }
    if (bounds.empty()) {
      int_conds.push_back(tm_.mk_true());
    } else if (bounds.size() == 1) {
      int_conds.push_back(bounds[0]);
    } else {
      int_conds.push_back(tm_.mk_term(Kind::AND, bounds));
    }
  }
  if (int_conds.empty()) {
    return tm_.mk_true();
  }
  if (int_conds.size() == 1) {
    return int_conds[0];
  }
  return tm_.mk_term(Kind::OR, int_conds);
}

absl::StatusOr<Value> IrTranslator::ExtractValue(::bitwuzla::Bitwuzla& bitwuzla,
                                                 Term term, const Type* type) {
  switch (type->kind()) {
    case TypeKind::kToken:
      return Value::Token();
    case TypeKind::kBits: {
      int64_t bc = type->AsBitsOrDie()->bit_count();
      if (bc == 0) {
        return Value(Bits());
      }
      Term eval = bitwuzla.get_value(term);
      XLS_ASSIGN_OR_RETURN(Bits bits, ParseUnsignedNumberWithoutPrefix(
                                          eval.value<std::string>(16),
                                          FormatPreference::kHex, bc));
      return Value(std::move(bits));
    }
    case TypeKind::kTuple: {
      const TupleType* tt = type->AsTupleOrDie();
      std::vector<Value> elems;
      for (int64_t i = 0; i < tt->size(); ++i) {
        XLS_ASSIGN_OR_RETURN(Term elem_t, GetLttElement(type, term, {i}));
        XLS_ASSIGN_OR_RETURN(
            Value ev, ExtractValue(bitwuzla, elem_t, tt->element_type(i)));
        elems.push_back(ev);
      }
      return Value::TupleOwned(std::move(elems));
    }
    case TypeKind::kArray: {
      const ArrayType* at = type->AsArrayOrDie();
      std::vector<Value> elems;
      for (int64_t i = 0; i < at->size(); ++i) {
        Term idx = GetAsFormattedArrayIndex(i, at);
        Term elem_t = GetArrayElement(at, term, idx);
        XLS_ASSIGN_OR_RETURN(
            Value ev, ExtractValue(bitwuzla, elem_t, at->element_type()));
        elems.push_back(ev);
      }
      return Value::Array(elems);
    }
  }
  return absl::InternalError("Unknown type kind in ExtractValue");
}

absl::StatusOr<Term> IrTranslator::TernaryTreePredicateToAssertion(
    LeafTypeTreeView<TernaryVector> ternaries, LeafTypeTreeView<Term> leaves) {
  XLS_RET_CHECK_EQ(leaves.size(), ternaries.size());
  std::vector<Term> constraints;
  constraints.reserve(ternaries.size());
  for (int64_t i = 0; i < ternaries.size(); ++i) {
    TernarySpan t = ternaries.elements()[i];
    if (!ternary_ops::AllUnknown(t)) {
      XLS_ASSIGN_OR_RETURN(Term c,
                           TernaryToConstraint(leaves.elements()[i], t));
      constraints.push_back(c);
    }
  }
  return MakeConjunction(tm_, constraints);
}

absl::StatusOr<Term> IrTranslator::IntervalTreePredicateToAssertion(
    LeafTypeTreeView<IntervalSet> intervals, LeafTypeTreeView<Term> leaves) {
  XLS_RET_CHECK_EQ(leaves.size(), intervals.size());
  std::vector<Term> constraints;
  constraints.reserve(intervals.size());
  for (int64_t i = 0; i < intervals.size(); ++i) {
    const IntervalSet& range = intervals.elements()[i];
    if (!range.IsMaximal()) {
      XLS_ASSIGN_OR_RETURN(
          Term c, IntervalSetToConstraint(leaves.elements()[i], range));
      constraints.push_back(c);
    }
  }
  return MakeConjunction(tm_, constraints);
}

absl::StatusOr<Term> IrTranslator::LeafTreePredicateToAssertion(
    const Predicate& p, Node* subject) {
  XLS_ASSIGN_OR_RETURN(LeafTypeTree<Term> leaves, ToLeafTypeTree(subject));
  if (p.kind() == PredicateKind::kCompatibleWithTernary) {
    return TernaryTreePredicateToAssertion(p.ternaries(), leaves.AsView());
  }
  return IntervalTreePredicateToAssertion(p.intervals(), leaves.AsView());
}

absl::StatusOr<Term> IrTranslator::PredicateToAssertion(const Predicate& p,
                                                        Node* subject,
                                                        Term val) {
  Term zero_1b = tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0);
  switch (p.kind()) {
    case PredicateKind::kEqualToZero:
      XLS_RETURN_IF_ERROR(RequireBitVectorSort(val));
      return tm_.mk_term(Kind::EQUAL, {val, ZeroOfSort(val.sort())});
    case PredicateKind::kNotEqualToZero:
      XLS_RETURN_IF_ERROR(RequireBitVectorSort(val));
      return tm_.mk_term(Kind::DISTINCT, {val, ZeroOfSort(val.sort())});
    case PredicateKind::kEqualToNode: {
      if (p.node()->GetType()->IsToken()) {
        return tm_.mk_true();
      }
      Term b = GetTranslation(p.node());
      Term ne = ComputeNe(val, b, subject->GetType());
      return tm_.mk_term(Kind::EQUAL, {ne, zero_1b});
    }
    case PredicateKind::kExclusiveWithNode: {
      Term b = GetTranslation(p.node());
      XLS_RETURN_IF_ERROR(RequireBitVectorSort(val));
      XLS_RETURN_IF_ERROR(RequireBitVectorSort(b));
      Term nz_a = tm_.mk_term(Kind::DISTINCT, {val, ZeroOfSort(val.sort())});
      Term nz_b = tm_.mk_term(Kind::DISTINCT, {b, ZeroOfSort(b.sort())});
      return tm_.mk_term(Kind::NOT, {tm_.mk_term(Kind::AND, {nz_a, nz_b})});
    }
    case PredicateKind::kUnsignedGreaterOrEqual:
      XLS_RETURN_IF_ERROR(RequireBitVectorSort(val));
      return tm_.mk_term(Kind::BV_UGE, {val, TranslateLiteralBits(p.value())});
    case PredicateKind::kUnsignedLessOrEqual:
      XLS_RETURN_IF_ERROR(RequireBitVectorSort(val));
      return tm_.mk_term(Kind::BV_ULE, {val, TranslateLiteralBits(p.value())});
    case PredicateKind::kCompatibleWithTernary:
    case PredicateKind::kCompatibleWithIntervalSet:
      return LeafTreePredicateToAssertion(p, subject);
  }
  return absl::UnimplementedError("Unhandled predicate kind");
}

absl::StatusOr<Term> IrTranslator::PredicateToNegatedObjective(
    const Predicate& p, Node* subject, Term val) {
  XLS_ASSIGN_OR_RETURN(Term ast, PredicateToAssertion(p, subject, val));
  return tm_.mk_term(Kind::NOT, {ast});
}

void IrTranslator::NoteTranslation(Node* node, Term term) {
  if (translations_.contains(node)) {
    return;
  }
  if (node->GetType()->GetFlatBitCount() == 0) {
    translations_[node] = MakeZeroTerm(node->GetType());
    return;
  }
  translations_[node] = term;
}

absl::Status IrTranslator::DefaultHandler(Node* node) {
  if (allow_unsupported_) {
    Term fresh = tm_.mk_const(
        TypeToSort(node->GetType()),
        GetNewSymbol(absl::StrCat(node->GetName(), "_unsupported")));
    NoteTranslation(node, fresh);
    return absl::OkStatus();
  }
  return absl::UnimplementedError(
      absl::StrCat("Unhandled XLS IR node for Bitwuzla: ", node->ToString()));
}

absl::Status IrTranslator::HandleLiteral(Literal* literal) {
  XLS_ASSIGN_OR_RETURN(
      Term val, TranslateLiteralValue(literal->GetType(), literal->value()));
  NoteTranslation(literal, val);
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleParam(Param* param) {
  if (translations_.contains(param)) {
    return absl::OkStatus();
  }
  if (param->GetType()->GetFlatBitCount() == 0) {
    NoteTranslation(param, MakeZeroTerm(param->GetType()));
    return absl::OkStatus();
  }
  Term c = tm_.mk_const(TypeToSort(param->GetType()), param->GetName());
  NoteTranslation(param, c);
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleAdd(BinOp* add) {
  Term res = tm_.mk_term(Kind::BV_ADD, {GetTranslation(add->operand(0)),
                                        GetTranslation(add->operand(1))});
  NoteTranslation(add, res);
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleSub(BinOp* sub) {
  Term res = tm_.mk_term(Kind::BV_SUB, {GetTranslation(sub->operand(0)),
                                        GetTranslation(sub->operand(1))});
  NoteTranslation(sub, res);
  return absl::OkStatus();
}

::bitwuzla::Term IrTranslator::ExtendOrTruncate(Node* operand, int64_t width,
                                                bool is_signed) {
  if (width == 0) {
    width = 1;
  }
  if (operand->GetType()->GetFlatBitCount() == 0) {
    return tm_.mk_bv_value_uint64(tm_.mk_bv_sort(width), 0);
  }
  Term val = GetTranslation(operand);
  int64_t current_width = val.sort().bv_size();
  if (current_width == width) {
    return val;
  }
  if (current_width < width) {
    if (is_signed) {
      return tm_.mk_term(Kind::BV_SIGN_EXTEND, {val},
                         {static_cast<uint64_t>(width - current_width)});
    } else {
      return tm_.mk_term(Kind::BV_ZERO_EXTEND, {val},
                         {static_cast<uint64_t>(width - current_width)});
    }
  }
  // current_width > width: truncate.
  return tm_.mk_term(Kind::BV_EXTRACT, {val},
                     {static_cast<uint64_t>(width - 1), 0});
}

absl::Status IrTranslator::HandleSMul(ArithOp* mul) {
  int64_t result_width = mul->BitCountOrDie();
  Term lhs = ExtendOrTruncate(mul->operand(0), result_width,
                              /*is_signed=*/true);
  Term rhs = ExtendOrTruncate(mul->operand(1), result_width,
                              /*is_signed=*/true);
  Term res = tm_.mk_term(Kind::BV_MUL, {lhs, rhs});
  NoteTranslation(mul, res);
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleUMul(ArithOp* mul) {
  int64_t result_width = mul->BitCountOrDie();
  Term lhs = ExtendOrTruncate(mul->operand(0), result_width,
                              /*is_signed=*/false);
  Term rhs = ExtendOrTruncate(mul->operand(1), result_width,
                              /*is_signed=*/false);
  Term res = tm_.mk_term(Kind::BV_MUL, {lhs, rhs});
  NoteTranslation(mul, res);
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleSMulp(PartialProductOp* mul) {
  uint64_t w = mul->width();
  Term lhs_ext = ExtendOrTruncate(mul->operand(0), w, /*is_signed=*/true);
  Term rhs_ext = ExtendOrTruncate(mul->operand(1), w, /*is_signed=*/true);
  Term prod = tm_.mk_term(Kind::BV_MUL, {lhs_ext, rhs_ext});
  Sort s = tm_.mk_bv_sort(w);
  Term offset = tm_.mk_const(s, GetNewSymbol("mulp_offset"));
  Term diff = tm_.mk_term(Kind::BV_SUB, {prod, offset});
  NoteTranslation(mul, ConcatN(absl::MakeConstSpan({offset, diff})));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleUMulp(PartialProductOp* mul) {
  uint64_t w = mul->width();
  Term lhs_ext = ExtendOrTruncate(mul->operand(0), w, /*is_signed=*/false);
  Term rhs_ext = ExtendOrTruncate(mul->operand(1), w, /*is_signed=*/false);
  Term prod = tm_.mk_term(Kind::BV_MUL, {lhs_ext, rhs_ext});
  Sort s = tm_.mk_bv_sort(w);
  Term offset = tm_.mk_const(s, GetNewSymbol("mulp_offset"));
  Term diff = tm_.mk_term(Kind::BV_SUB, {prod, offset});
  NoteTranslation(mul, ConcatN(absl::MakeConstSpan({offset, diff})));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleSDiv(BinOp* div) {
  Term lhs = GetTranslation(div->operand(0));
  if (div->BitCountOrDie() == 0) {
    NoteTranslation(div, ZeroOfSort(lhs.sort()));
    return absl::OkStatus();
  }
  Term rhs = GetTranslation(div->operand(1));
  uint64_t width = lhs.sort().bv_size();
  Term zero = tm_.mk_bv_value_uint64(rhs.sort(), 0);
  Term is_zero = tm_.mk_term(Kind::EQUAL, {rhs, zero});
  Term msb = tm_.mk_term(Kind::BV_EXTRACT, {lhs}, {width - 1, width - 1});
  Term is_neg =
      tm_.mk_term(Kind::EQUAL, {msb, tm_.mk_bv_value_uint64(msb.sort(), 1)});
  Term div_by_zero = tm_.mk_term(
      Kind::ITE,
      {is_neg, MinSignedOfSort(lhs.sort()), MaxSignedOfSort(lhs.sort())});
  Term res = tm_.mk_term(Kind::ITE, {is_zero, div_by_zero,
                                     tm_.mk_term(Kind::BV_SDIV, {lhs, rhs})});
  NoteTranslation(div, res);
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleUDiv(BinOp* div) {
  Term lhs = GetTranslation(div->operand(0));
  if (div->BitCountOrDie() == 0) {
    NoteTranslation(div, ZeroOfSort(lhs.sort()));
    return absl::OkStatus();
  }
  Term rhs = GetTranslation(div->operand(1));
  Term zero = tm_.mk_bv_value_uint64(rhs.sort(), 0);
  Term is_zero = tm_.mk_term(Kind::EQUAL, {rhs, zero});
  Term res = tm_.mk_term(Kind::ITE, {is_zero, OnesOfSort(lhs.sort()),
                                     tm_.mk_term(Kind::BV_UDIV, {lhs, rhs})});
  NoteTranslation(div, res);
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleSMod(BinOp* mod) {
  Term lhs = GetTranslation(mod->operand(0));
  Term rhs = GetTranslation(mod->operand(1));
  Term zero = tm_.mk_bv_value_uint64(rhs.sort(), 0);
  Term is_zero = tm_.mk_term(Kind::EQUAL, {rhs, zero});
  Term res = tm_.mk_term(Kind::ITE, {is_zero, ZeroOfSort(lhs.sort()),
                                     tm_.mk_term(Kind::BV_SREM, {lhs, rhs})});
  NoteTranslation(mod, res);
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleUMod(BinOp* mod) {
  Term lhs = GetTranslation(mod->operand(0));
  Term rhs = GetTranslation(mod->operand(1));
  Term zero = tm_.mk_bv_value_uint64(rhs.sort(), 0);
  Term is_zero = tm_.mk_term(Kind::EQUAL, {rhs, zero});
  Term res = tm_.mk_term(Kind::ITE, {is_zero, ZeroOfSort(lhs.sort()),
                                     tm_.mk_term(Kind::BV_UREM, {lhs, rhs})});
  NoteTranslation(mod, res);
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleEq(CompareOp* eq) {
  Term ne =
      ComputeNe(GetTranslation(eq->operand(0)), GetTranslation(eq->operand(1)),
                eq->operand(0)->GetType());
  Term eq_bv = tm_.mk_term(Kind::BV_NOT, {ne});
  NoteTranslation(eq, eq_bv);
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleNe(CompareOp* ne) {
  Term res =
      ComputeNe(GetTranslation(ne->operand(0)), GetTranslation(ne->operand(1)),
                ne->operand(0)->GetType());
  NoteTranslation(ne, res);
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleSGe(CompareOp* sge) {
  Term c = tm_.mk_term(Kind::BV_SGE, {GetTranslation(sge->operand(0)),
                                      GetTranslation(sge->operand(1))});
  NoteTranslation(
      sge,
      tm_.mk_term(Kind::ITE, {c, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 1),
                              tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0)}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleSGt(CompareOp* sgt) {
  Term c = tm_.mk_term(Kind::BV_SGT, {GetTranslation(sgt->operand(0)),
                                      GetTranslation(sgt->operand(1))});
  NoteTranslation(
      sgt,
      tm_.mk_term(Kind::ITE, {c, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 1),
                              tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0)}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleSLe(CompareOp* sle) {
  Term c = tm_.mk_term(Kind::BV_SLE, {GetTranslation(sle->operand(0)),
                                      GetTranslation(sle->operand(1))});
  NoteTranslation(
      sle,
      tm_.mk_term(Kind::ITE, {c, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 1),
                              tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0)}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleSLt(CompareOp* slt) {
  Term c = tm_.mk_term(Kind::BV_SLT, {GetTranslation(slt->operand(0)),
                                      GetTranslation(slt->operand(1))});
  NoteTranslation(
      slt,
      tm_.mk_term(Kind::ITE, {c, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 1),
                              tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0)}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleUGe(CompareOp* uge) {
  Term c = tm_.mk_term(Kind::BV_UGE, {GetTranslation(uge->operand(0)),
                                      GetTranslation(uge->operand(1))});
  NoteTranslation(
      uge,
      tm_.mk_term(Kind::ITE, {c, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 1),
                              tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0)}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleUGt(CompareOp* ugt) {
  Term c = tm_.mk_term(Kind::BV_UGT, {GetTranslation(ugt->operand(0)),
                                      GetTranslation(ugt->operand(1))});
  NoteTranslation(
      ugt,
      tm_.mk_term(Kind::ITE, {c, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 1),
                              tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0)}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleULe(CompareOp* ule) {
  Term c = tm_.mk_term(Kind::BV_ULE, {GetTranslation(ule->operand(0)),
                                      GetTranslation(ule->operand(1))});
  NoteTranslation(
      ule,
      tm_.mk_term(Kind::ITE, {c, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 1),
                              tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0)}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleULt(CompareOp* ult) {
  Term c = tm_.mk_term(Kind::BV_ULT, {GetTranslation(ult->operand(0)),
                                      GetTranslation(ult->operand(1))});
  NoteTranslation(
      ult,
      tm_.mk_term(Kind::ITE, {c, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 1),
                              tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0)}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleNary(NaryOp* op, Kind kind,
                                      bool invert_result) {
  if (op->operand_count() == 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(OpToString(op->op()), " has no operands"));
  }
  std::vector<Term> operands;
  operands.reserve(op->operand_count());
  for (Node* n : op->operands()) {
    operands.push_back(GetTranslation(n));
  }
  Term res = operands.size() == 1 ? operands[0] : tm_.mk_term(kind, operands);
  if (invert_result) {
    res = tm_.mk_term(Kind::BV_NOT, {res});
  }
  NoteTranslation(op, res);
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleNaryAnd(NaryOp* and_op) {
  return HandleNary(and_op, Kind::BV_AND, /*invert_result=*/false);
}

absl::Status IrTranslator::HandleNaryOr(NaryOp* or_op) {
  return HandleNary(or_op, Kind::BV_OR, /*invert_result=*/false);
}

absl::Status IrTranslator::HandleNaryXor(NaryOp* xor_op) {
  return HandleNary(xor_op, Kind::BV_XOR, /*invert_result=*/false);
}

absl::Status IrTranslator::HandleNaryNand(NaryOp* nand_op) {
  return HandleNary(nand_op, Kind::BV_AND, /*invert_result=*/true);
}

absl::Status IrTranslator::HandleNaryNor(NaryOp* nor_op) {
  return HandleNary(nor_op, Kind::BV_OR, /*invert_result=*/true);
}

absl::Status IrTranslator::HandleNot(UnOp* not_op) {
  NoteTranslation(
      not_op, tm_.mk_term(Kind::BV_NOT, {GetTranslation(not_op->operand(0))}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleNeg(UnOp* neg) {
  NoteTranslation(neg,
                  tm_.mk_term(Kind::BV_NEG, {GetTranslation(neg->operand(0))}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleIdentity(UnOp* identity) {
  NoteTranslation(identity, GetTranslation(identity->operand(0)));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleReverse(UnOp* reverse) {
  Term val = GetTranslation(reverse->operand(0));
  uint64_t w = val.sort().bv_size();
  if (w <= 1) {
    NoteTranslation(reverse, val);
    return absl::OkStatus();
  }
  std::vector<Term> bits;
  bits.reserve(w);
  for (uint64_t i = 0; i < w; ++i) {
    bits.push_back(tm_.mk_term(Kind::BV_EXTRACT, {val}, {i, i}));
  }
  NoteTranslation(reverse, ConcatN(bits));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleAndReduce(BitwiseReductionOp* and_reduce) {
  if (and_reduce->operand(0)->BitCountOrDie() == 0) {
    NoteTranslation(and_reduce, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 1));
    return absl::OkStatus();
  }
  NoteTranslation(
      and_reduce,
      tm_.mk_term(Kind::BV_REDAND, {GetTranslation(and_reduce->operand(0))}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleOrReduce(BitwiseReductionOp* or_reduce) {
  NoteTranslation(
      or_reduce,
      tm_.mk_term(Kind::BV_REDOR, {GetTranslation(or_reduce->operand(0))}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleXorReduce(BitwiseReductionOp* xor_reduce) {
  NoteTranslation(
      xor_reduce,
      tm_.mk_term(Kind::BV_REDXOR, {GetTranslation(xor_reduce->operand(0))}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleShll(BinOp* shll) {
  Term lhs = GetTranslation(shll->operand(0));
  Term rhs = CoerceShiftAmount(lhs, shll->operand(1));
  Term res = tm_.mk_term(Kind::BV_SHL, {lhs, rhs});
  uint64_t w = lhs.sort().bv_size();
  Term max_shamt = tm_.mk_bv_value_uint64(rhs.sort(), w);
  Term oob = tm_.mk_term(Kind::BV_UGE, {rhs, max_shamt});
  NoteTranslation(
      shll, tm_.mk_term(Kind::ITE,
                        {oob, tm_.mk_bv_value_uint64(lhs.sort(), 0), res}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleShrl(BinOp* shrl) {
  Term lhs = GetTranslation(shrl->operand(0));
  Term rhs = CoerceShiftAmount(lhs, shrl->operand(1));
  Term res = tm_.mk_term(Kind::BV_SHR, {lhs, rhs});
  uint64_t w = lhs.sort().bv_size();
  Term max_shamt = tm_.mk_bv_value_uint64(rhs.sort(), w);
  Term oob = tm_.mk_term(Kind::BV_UGE, {rhs, max_shamt});
  NoteTranslation(
      shrl, tm_.mk_term(Kind::ITE,
                        {oob, tm_.mk_bv_value_uint64(lhs.sort(), 0), res}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleShra(BinOp* shra) {
  Term lhs = GetTranslation(shra->operand(0));
  Term rhs = CoerceShiftAmount(lhs, shra->operand(1));
  Term res = tm_.mk_term(Kind::BV_ASHR, {lhs, rhs});
  uint64_t w = lhs.sort().bv_size();
  Term max_shamt = tm_.mk_bv_value_uint64(rhs.sort(), w);
  Term oob = tm_.mk_term(Kind::BV_UGE, {rhs, max_shamt});
  Term sign_bit = tm_.mk_term(Kind::BV_EXTRACT, {lhs}, {w - 1, w - 1});
  Term all_sign = tm_.mk_term(
      Kind::ITE, {tm_.mk_term(Kind::EQUAL, {sign_bit, tm_.mk_bv_value_uint64(
                                                          sign_bit.sort(), 1)}),
                  OnesOfSort(lhs.sort()), ZeroOfSort(lhs.sort())});
  NoteTranslation(shra, tm_.mk_term(Kind::ITE, {oob, all_sign, res}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleSignExtend(ExtendOp* sign_ext) {
  uint64_t w = sign_ext->new_bit_count();
  if (sign_ext->operand(0)->GetType()->GetFlatBitCount() == 0) {
    NoteTranslation(sign_ext, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(w), 0));
    return absl::OkStatus();
  }

  Term val = GetTranslation(sign_ext->operand(0));
  uint64_t old_w = val.sort().bv_size();
  if (w == old_w) {
    NoteTranslation(sign_ext, val);
  } else {
    NoteTranslation(sign_ext,
                    tm_.mk_term(Kind::BV_SIGN_EXTEND, {val}, {w - old_w}));
  }
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleZeroExtend(ExtendOp* zero_ext) {
  uint64_t w = zero_ext->new_bit_count();
  if (zero_ext->operand(0)->GetType()->GetFlatBitCount() == 0) {
    NoteTranslation(zero_ext, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(w), 0));
    return absl::OkStatus();
  }

  Term val = GetTranslation(zero_ext->operand(0));
  uint64_t old_w = val.sort().bv_size();
  if (w == old_w) {
    NoteTranslation(zero_ext, val);
  } else {
    NoteTranslation(zero_ext,
                    tm_.mk_term(Kind::BV_ZERO_EXTEND, {val}, {w - old_w}));
  }
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleConcat(Concat* concat) {
  if (concat->GetType()->GetFlatBitCount() == 0) {
    NoteTranslation(concat, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0));
    return absl::OkStatus();
  }
  std::vector<Term> args;
  for (Node* n : concat->operands()) {
    if (n->GetType()->GetFlatBitCount() == 0) {
      continue;
    }
    args.push_back(GetTranslation(n));
  }
  NoteTranslation(concat, ConcatN(args));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleBitSlice(BitSlice* bit_slice) {
  Term val = GetTranslation(bit_slice->operand(0));
  uint64_t start = bit_slice->start();
  uint64_t w = bit_slice->width();
  if (w == 0) {
    NoteTranslation(bit_slice, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0));
  } else {
    NoteTranslation(bit_slice, tm_.mk_term(Kind::BV_EXTRACT, {val},
                                           {start + w - 1, start}));
  }
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleDynamicBitSlice(DynamicBitSlice* bit_slice) {
  uint64_t w = bit_slice->width();
  if (w == 0) {
    NoteTranslation(bit_slice, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0));
    return absl::OkStatus();
  }
  Term val = GetTranslation(bit_slice->operand(0));
  Term start = GetTranslation(bit_slice->operand(1));
  uint64_t vw = val.sort().bv_size();
  uint64_t sw = start.sort().bv_size();
  uint64_t mw = std::max({vw, sw, w});
  Term val_ext =
      vw == mw ? val : tm_.mk_term(Kind::BV_ZERO_EXTEND, {val}, {mw - vw});
  Term start_ext =
      sw == mw ? start : tm_.mk_term(Kind::BV_ZERO_EXTEND, {start}, {mw - sw});
  Term max_start = tm_.mk_bv_value_uint64(start_ext.sort(), vw);
  Term oob = tm_.mk_term(Kind::BV_UGE, {start_ext, max_start});
  Term shifted = tm_.mk_term(Kind::BV_SHR, {val_ext, start_ext});
  Term trunc = tm_.mk_term(Kind::BV_EXTRACT, {shifted}, {w - 1, 0});
  NoteTranslation(
      bit_slice,
      tm_.mk_term(Kind::ITE,
                  {oob, tm_.mk_bv_value_uint64(trunc.sort(), 0), trunc}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleBitSliceUpdate(BitSliceUpdate* update) {
  Term to_update = GetTranslation(update->to_update());
  Term start = GetTranslation(update->start());
  Term update_val = GetTranslation(update->update_value());
  uint64_t uw = to_update.sort().bv_size();
  uint64_t vw = update_val.sort().bv_size();
  Term mask_w =
      tm_.mk_term(Kind::BV_NOT, {tm_.mk_bv_value_uint64(update_val.sort(), 0)});
  Term ext_mask =
      vw == uw
          ? mask_w
          : (vw < uw ? tm_.mk_term(Kind::BV_ZERO_EXTEND, {mask_w}, {uw - vw})
                     : tm_.mk_term(Kind::BV_EXTRACT, {mask_w}, {uw - 1, 0}));
  Term ext_val =
      vw == uw
          ? update_val
          : (vw < uw
                 ? tm_.mk_term(Kind::BV_ZERO_EXTEND, {update_val}, {uw - vw})
                 : tm_.mk_term(Kind::BV_EXTRACT, {update_val}, {uw - 1, 0}));
  Term coerced_start = CoerceShiftAmount(to_update, update->start());
  Term shifted_mask = tm_.mk_term(Kind::BV_SHL, {ext_mask, coerced_start});
  Term shifted_val = tm_.mk_term(Kind::BV_SHL, {ext_val, coerced_start});
  Term inv_mask = tm_.mk_term(Kind::BV_NOT, {shifted_mask});
  Term masked_old = tm_.mk_term(Kind::BV_AND, {to_update, inv_mask});
  NoteTranslation(update, tm_.mk_term(Kind::BV_OR, {masked_old, shifted_val}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleSel(Select* sel) {
  Term sel_val = GetTranslation(sel->selector());
  Term res;
  if (sel->default_value().has_value()) {
    res = GetTranslation(sel->default_value().value());
  } else {
    res = GetTranslation(sel->cases().back());
  }
  size_t num_cases = sel->default_value().has_value() ? sel->cases().size()
                                                      : sel->cases().size() - 1;
  for (size_t i = 0; i < num_cases; ++i) {
    Term c_val = GetTranslation(sel->cases()[i]);
    Term idx = tm_.mk_bv_value_uint64(sel_val.sort(), i);
    Term eq = tm_.mk_term(Kind::EQUAL, {sel_val, idx});
    res = tm_.mk_term(Kind::ITE, {eq, c_val, res});
  }
  NoteTranslation(sel, res);
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleOneHot(OneHot* one_hot) {
  Term x = GetTranslation(one_hot->operand(0));
  uint64_t w = one_hot->operand(0)->BitCountOrDie();
  Term zero_1b = tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0);
  Term one_1b = tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 1);
  if (w == 0) {
    NoteTranslation(one_hot, one_1b);
    return absl::OkStatus();
  }
  Term is_zero =
      tm_.mk_term(Kind::EQUAL, {x, tm_.mk_bv_value_uint64(x.sort(), 0)});
  Term msb = tm_.mk_term(Kind::ITE, {is_zero, one_1b, zero_1b});
  if (w == 1) {
    NoteTranslation(one_hot, tm_.mk_term(Kind::BV_CONCAT, {msb, x}));
    return absl::OkStatus();
  }
  Term input = one_hot->priority() == LsbOrMsb::kLsb
                   ? x
                   : FlattenToBv(x, one_hot->operand(0)->GetType());
  if (one_hot->priority() == LsbOrMsb::kMsb) {
    std::vector<Term> rev_bits;
    rev_bits.reserve(w);
    for (uint64_t i = 0; i < w; ++i) {
      rev_bits.push_back(tm_.mk_term(Kind::BV_EXTRACT, {x}, {i, i}));
    }
    input = ConcatN(rev_bits);
  }
  Term isolated =
      tm_.mk_term(Kind::BV_AND, {input, tm_.mk_term(Kind::BV_NEG, {input})});
  Term hot_bits = isolated;
  if (one_hot->priority() == LsbOrMsb::kMsb) {
    std::vector<Term> rev_bits;
    rev_bits.reserve(w);
    for (uint64_t i = 0; i < w; ++i) {
      rev_bits.push_back(tm_.mk_term(Kind::BV_EXTRACT, {isolated}, {i, i}));
    }
    hot_bits = ConcatN(rev_bits);
  }
  NoteTranslation(one_hot, tm_.mk_term(Kind::BV_CONCAT, {msb, hot_bits}));
  return absl::OkStatus();
}

Term IrTranslator::BuildOneHotSel(const Type* type,
                                  absl::Span<const Term> sel_conds,
                                  absl::Span<const Term> cases) {
  if (cases.empty() || type->GetFlatBitCount() == 0) {
    return MakeZeroTerm(type);
  }
  if (type->IsArray()) {
    const ArrayType* at = type->AsArrayOrDie();
    if (at->size() == 1) {
      return BuildOneHotSel(at->element_type(), sel_conds, cases);
    }
    std::vector<Term> elems;
    elems.reserve(at->size());
    std::vector<Term> elem_cases;
    elem_cases.reserve(cases.size());
    for (int64_t j = 0; j < at->size(); ++j) {
      Term idx = GetAsFormattedArrayIndex(j, at);
      elem_cases.clear();
      for (const Term& c : cases) {
        elem_cases.push_back(tm_.mk_term(Kind::ARRAY_SELECT, {c, idx}));
      }
      elems.push_back(
          BuildOneHotSel(at->element_type(), sel_conds, elem_cases));
    }
    return BuildArrayFromElements(at, elems, "ohs_arr");
  }

  Term zero = ZeroOfSort(TypeToSort(type));
  std::vector<Term> masked_terms;
  masked_terms.reserve(cases.size());
  for (size_t i = 0; i < cases.size(); ++i) {
    masked_terms.push_back(
        tm_.mk_term(Kind::ITE, {sel_conds[i], cases[i], zero}));
  }
  if (masked_terms.size() == 1) {
    return masked_terms[0];
  }
  return tm_.mk_term(Kind::BV_OR, masked_terms);
}

absl::Status IrTranslator::HandleOneHotSel(OneHotSelect* one_hot_sel) {
  Term sel = GetTranslation(one_hot_sel->selector());
  uint64_t w = sel.sort().bv_size();
  uint64_t num_cases =
      std::min(w, static_cast<uint64_t>(one_hot_sel->cases().size()));
  std::vector<Term> sel_conds;
  std::vector<Term> cases;
  sel_conds.reserve(num_cases);
  cases.reserve(num_cases);
  Term one = tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 1);
  for (uint64_t i = 0; i < num_cases; ++i) {
    Term bit = tm_.mk_term(Kind::BV_EXTRACT, {sel}, {i, i});
    sel_conds.push_back(tm_.mk_term(Kind::EQUAL, {bit, one}));
    cases.push_back(GetTranslation(one_hot_sel->cases()[i]));
  }
  NoteTranslation(one_hot_sel,
                  BuildOneHotSel(one_hot_sel->GetType(), sel_conds, cases));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandlePrioritySel(PrioritySelect* priority_sel) {
  Term sel = GetTranslation(priority_sel->selector());
  uint64_t w = sel.sort().bv_size();
  Term res = GetTranslation(priority_sel->default_value());
  for (int64_t i = w - 1; i >= 0; --i) {
    Term bit =
        tm_.mk_term(Kind::BV_EXTRACT, {sel},
                    {static_cast<uint64_t>(i), static_cast<uint64_t>(i)});
    Term c_val = GetTranslation(priority_sel->cases()[i]);
    Term cond =
        tm_.mk_term(Kind::EQUAL, {bit, tm_.mk_bv_value_uint64(bit.sort(), 1)});
    res = tm_.mk_term(Kind::ITE, {cond, c_val, res});
  }
  NoteTranslation(priority_sel, res);
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleTuple(Tuple* tuple) {
  if (tuple->GetType()->GetFlatBitCount() == 0) {
    NoteTranslation(tuple, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0));
    return absl::OkStatus();
  }
  std::vector<Term> elems;
  for (Node* n : tuple->operands()) {
    if (n->GetType()->GetFlatBitCount() == 0) {
      continue;
    }
    elems.push_back(FlattenToBv(GetTranslation(n), n->GetType()));
  }
  NoteTranslation(tuple, ConcatN(elems));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleTupleIndex(TupleIndex* tuple_index) {
  const TupleType* tt = tuple_index->operand(0)->GetType()->AsTupleOrDie();
  Term val = GetTranslation(tuple_index->operand(0));
  int64_t idx = tuple_index->index();
  int64_t low = 0;
  for (int64_t j = tt->size() - 1; j > idx; --j) {
    low += tt->element_type(j)->GetFlatBitCount();
  }
  int64_t fb = tt->element_type(idx)->GetFlatBitCount();
  if (fb == 0) {
    NoteTranslation(tuple_index, MakeZeroTerm(tuple_index->GetType()));
    return absl::OkStatus();
  }
  Term slice = tm_.mk_term(
      Kind::BV_EXTRACT, {val},
      {static_cast<uint64_t>(low + fb - 1), static_cast<uint64_t>(low)});
  NoteTranslation(tuple_index,
                  UnflattenFromArrayBv(slice, tuple_index->GetType()));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleArray(Array* array) {
  const ArrayType* at = array->GetType()->AsArrayOrDie();
  std::vector<Term> elems;
  elems.reserve(array->operand_count());
  for (Node* operand : array->operands()) {
    elems.push_back(GetTranslation(operand));
  }
  NoteTranslation(array, BuildArrayFromElements(at, elems, "array_init"));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleArrayIndex(ArrayIndex* array_index) {
  const Type* cur_type = array_index->array()->GetType();
  Term cur_val = GetTranslation(array_index->array());
  for (Node* idx : array_index->indices()) {
    const ArrayType* at = cur_type->AsArrayOrDie();
    cur_val = GetArrayElement(at, cur_val, GetTranslation(idx));
    cur_type = at->element_type();
  }
  NoteTranslation(array_index, cur_val);
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleArrayUpdate(ArrayUpdate* update) {
  std::vector<Term> indices;
  for (Node* n : update->indices()) {
    indices.push_back(GetTranslation(n));
  }
  Term res = UpdateArrayElement(
      update->GetType(), GetTranslation(update->array_to_update()),
      GetTranslation(update->update_value()), tm_.mk_true(), indices);
  NoteTranslation(update, res);
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleArrayConcat(ArrayConcat* array_concat) {
  const ArrayType* res_type = array_concat->GetType()->AsArrayOrDie();
  std::vector<Term> elems;
  elems.reserve(res_type->size());
  for (Node* arr : array_concat->operands()) {
    const ArrayType* op_type = arr->GetType()->AsArrayOrDie();
    Term arr_val = GetTranslation(arr);
    for (int64_t i = 0; i < op_type->size(); ++i) {
      elems.push_back(GetArrayElement(op_type, arr_val,
                                      GetAsFormattedArrayIndex(i, op_type)));
    }
  }
  NoteTranslation(array_concat,
                  BuildArrayFromElements(res_type, elems, "array_concat_init"));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleArraySlice(ArraySlice* slice) {
  const ArrayType* res_type = slice->GetType()->AsArrayOrDie();
  const ArrayType* src_type = slice->array()->GetType()->AsArrayOrDie();
  Term src = GetTranslation(slice->array());
  Term start = GetTranslation(slice->start());
  int64_t min_offset_bits = Bits::MinBitCountUnsigned(res_type->size());
  int64_t start_width = static_cast<int64_t>(start.sort().bv_size());
  // Zero-extend `start` to avoid wraparound overflow on `start + i`.
  int64_t offset_width = 1 + std::max(min_offset_bits, start_width);
  Sort offset_sort = tm_.mk_bv_sort(offset_width);
  Term start_ext =
      tm_.mk_term(Kind::BV_ZERO_EXTEND, {start},
                  {static_cast<uint64_t>(offset_width - start_width)});
  std::vector<Term> elems;
  elems.reserve(res_type->size());
  for (int64_t i = 0; i < res_type->size(); ++i) {
    Term offset = tm_.mk_bv_value_uint64(offset_sort, i);
    Term src_idx = tm_.mk_term(Kind::BV_ADD, {start_ext, offset});
    elems.push_back(GetArrayElement(src_type, src, src_idx));
  }
  NoteTranslation(slice,
                  BuildArrayFromElements(res_type, elems, "array_slice_init"));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleAfterAll(AfterAll* after_all) {
  NoteTranslation(after_all, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleMinDelay(MinDelay* min_delay) {
  NoteTranslation(min_delay, GetTranslation(min_delay->operand(0)));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleDecode(Decode* decode) {
  Term x = GetTranslation(decode->operand(0));
  uint64_t iw = x.sort().bv_size();
  uint64_t ow = decode->width();
  uint64_t mw = std::max(iw, ow);
  Sort msort = tm_.mk_bv_sort(mw);
  Term one = tm_.mk_bv_value_uint64(msort, 1);
  Term x_ext = tm_.mk_term(Kind::BV_ZERO_EXTEND, {x}, {mw - iw});
  Term oob =
      tm_.mk_term(Kind::BV_UGE, {x_ext, tm_.mk_bv_value_uint64(msort, mw)});
  Term shifted =
      tm_.mk_term(Kind::ITE, {oob, tm_.mk_bv_value_uint64(msort, 0),
                              tm_.mk_term(Kind::BV_SHL, {one, x_ext})});
  Term trunc = tm_.mk_term(Kind::BV_EXTRACT, {shifted}, {ow - 1, 0});
  NoteTranslation(decode, trunc);
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleEncode(Encode* encode) {
  Term x = GetTranslation(encode->operand(0));
  uint64_t iw = x.sort().bv_size();
  uint64_t ow = encode->BitCountOrDie();
  std::vector<Term> idx_bits;
  for (int64_t j = ow - 1; j >= 0; --j) {
    InlineBitmap bm(iw);
    for (int64_t k = 0; k < iw; ++k) {
      if ((k >> j) & 1) {
        bm.Set(k, true);
      }
    }
    Term mask = TranslateLiteralBits(Bits::FromBitmap(std::move(bm)));
    Term hit = tm_.mk_term(
        Kind::DISTINCT,
        {tm_.mk_term(Kind::BV_AND, {x, mask}), ZeroOfSort(x.sort())});
    idx_bits.push_back(tm_.mk_term(
        Kind::ITE, {hit, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 1),
                    tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0)}));
  }
  NoteTranslation(encode, ConcatN(idx_bits));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleGate(Gate* gate) {
  Term cond = GetTranslation(gate->condition());
  Term data = GetTranslation(gate->data());
  Term zero_1b = tm_.mk_bv_value_uint64(cond.sort(), 0);
  Term is_active = tm_.mk_term(Kind::DISTINCT, {cond, zero_1b});
  NoteTranslation(
      gate,
      tm_.mk_term(Kind::ITE, {is_active, data, MakeZeroTerm(gate->GetType())}));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleNext(Next* next) {
  NoteTranslation(next, tm_.mk_bv_value_uint64(tm_.mk_bv_sort(1), 0));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleInvoke(Invoke* invoke) {
  std::vector<Term> params;
  params.reserve(invoke->operands().size());
  for (Node* n : invoke->operands()) {
    params.push_back(GetTranslation(n));
  }
  XLS_ASSIGN_OR_RETURN(
      auto sub,
      CreateAndTranslate(tm_, invoke->to_apply(), params, allow_unsupported_));
  NoteTranslation(invoke, sub->GetReturnNode());
  return absl::OkStatus();
}

absl::StatusOr<ProverResult> IrTranslator::TryProveCombination(
    absl::Span<const PredicateOfNode> terms, PredicateCombination combination,
    absl::Span<const PredicateOfNode> assumptions) {
  CHECK(!terms.empty());
  std::vector<Term> neg_objs;
  neg_objs.reserve(terms.size());
  for (const auto& term : terms) {
    Term val = GetTranslation(term.subject);
    XLS_ASSIGN_OR_RETURN(
        Term neg_obj, PredicateToNegatedObjective(term.p, term.subject, val));
    neg_objs.push_back(neg_obj);
  }
  Term objective;
  if (neg_objs.size() == 1) {
    objective = neg_objs[0];
  } else if (combination == PredicateCombination::kConjunction) {
    objective = tm_.mk_term(Kind::OR, neg_objs);
  } else {
    objective = tm_.mk_term(Kind::AND, neg_objs);
  }

  ::bitwuzla::Options options;
  options.set(::bitwuzla::Option::PRODUCE_MODELS, true);
  ::bitwuzla::Bitwuzla bitwuzla(tm_, options);
  std::unique_ptr<::bitwuzla::Terminator> term_cb;
  if (timeout_.has_value() || limit_.has_value()) {
    term_cb = std::make_unique<XlsTerminator>(timeout_, limit_);
    bitwuzla.configure_terminator(term_cb.get());
  }

  for (const auto& ass : assumptions) {
    Term val = GetTranslation(ass.subject);
    XLS_ASSIGN_OR_RETURN(Term ass_t,
                         PredicateToAssertion(ass.p, ass.subject, val));
    bitwuzla.assert_formula(ass_t);
  }
  bitwuzla.assert_formula(objective);

  ::bitwuzla::Result res = bitwuzla.check_sat();
  if (res == ::bitwuzla::Result::UNSAT) {
    return ProvenTrue();
  }
  if (res == ::bitwuzla::Result::SAT) {
    absl::flat_hash_map<Node*, Value> counterexample;
    std::string message = "Bitwuzla returned SAT with counterexample:\n";
    for (Node* node : xls_function_->nodes()) {
      if (!node->OpIn({Op::kParam, Op::kRegisterRead, Op::kStateRead,
                       Op::kReceive, Op::kInputPort,
                       Op::kInstantiationInput})) {
        continue;
      }
      if (!translations_.contains(node)) {
        counterexample.emplace(node, ZeroOfType(node->GetType()));
        continue;
      }
      XLS_ASSIGN_OR_RETURN(
          Value val,
          ExtractValue(bitwuzla, GetTranslation(node), node->GetType()));
      absl::StrAppend(&message, "- ", node->ToString(), " = ", val.ToString(),
                      "\n");
      counterexample.emplace(node, std::move(val));
    }
    return ProvenFalse{.counterexample = std::move(counterexample),
                       .message = std::move(message)};
  }
  return absl::DeadlineExceededError("Bitwuzla solver timed out or unknown");
}

void BitwuzlaSolverInstance::SetLimit(const SolverLimit& limit) {
  translator_->SetTimeout(limit.timeout);
  translator_->SetDeterministicLimit(limit.deterministic_limit);
}

absl::StatusOr<ProverResult> BitwuzlaSolverInstance::TryProve(
    Node* subject, const Predicate& p,
    absl::Span<const PredicateOfNode> assumptions) {
  PredicateOfNode pon{.subject = subject, .p = p};
  return translator_->TryProveCombination(
      absl::MakeSpan(&pon, 1), PredicateCombination::kConjunction, assumptions);
}

absl::StatusOr<ProverResult> BitwuzlaSolverInstance::TryProveCombination(
    absl::Span<const PredicateOfNode> terms, PredicateCombination combination,
    absl::Span<const PredicateOfNode> assumptions) {
  return translator_->TryProveCombination(terms, combination, assumptions);
}

absl::StatusOr<std::unique_ptr<xls::solvers::SolverInstance>>
BitwuzlaSolver::CreateSolverInstance(FunctionBase* f, bool allow_unsupported) {
  XLS_ASSIGN_OR_RETURN(auto translator,
                       IrTranslator::CreateAndTranslate(f, allow_unsupported));
  return std::make_unique<BitwuzlaSolverInstance>(std::move(translator));
}

absl::StatusOr<ProverResult> BitwuzlaSolver::TryProve(
    FunctionBase* f, Node* subject, const Predicate& p,
    const SolverLimit& limit, bool allow_unsupported,
    absl::Span<const PredicateOfNode> assumptions) {
  XLS_ASSIGN_OR_RETURN(auto inst, CreateSolverInstance(f, allow_unsupported));
  inst->SetLimit(limit);
  return inst->TryProve(subject, p, assumptions);
}

absl::StatusOr<ProverResult> BitwuzlaSolver::TryProveCombination(
    FunctionBase* f, absl::Span<const PredicateOfNode> terms,
    PredicateCombination combination, const SolverLimit& limit,
    bool allow_unsupported, absl::Span<const PredicateOfNode> assumptions) {
  XLS_ASSIGN_OR_RETURN(auto inst, CreateSolverInstance(f, allow_unsupported));
  inst->SetLimit(limit);
  return inst->TryProveCombination(terms, combination, assumptions);
}

absl::StatusOr<ProverResult> TryProve(
    FunctionBase* f, Node* subject, Predicate p, absl::Duration timeout,
    bool allow_unsupported, absl::Span<const PredicateOfNode> assumptions) {
  BitwuzlaSolver solver;
  SolverLimit limit;
  limit.timeout = timeout;
  return solver.TryProve(f, subject, p, limit, allow_unsupported, assumptions);
}

absl::StatusOr<ProverResult> TryProve(
    FunctionBase* f, Node* subject, Predicate p, int64_t rlimit,
    bool allow_unsupported, absl::Span<const PredicateOfNode> assumptions) {
  BitwuzlaSolver solver;
  SolverLimit limit;
  limit.deterministic_limit = rlimit;
  return solver.TryProve(f, subject, p, limit, allow_unsupported, assumptions);
}

namespace {
bool Register() {
  SolverFactoryRegistry::Get().Register(SolverKind::kBitwuzla, []() {
    return std::make_unique<BitwuzlaSolver>();
  });
  return true;
}

static bool reg = Register();
}  // namespace

}  // namespace xls::solvers::bitwuzla
