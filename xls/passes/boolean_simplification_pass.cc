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

#include "xls/passes/boolean_simplification_pass.h"

#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node_util.h"
#include "xls/netlist/logical_effort.h"

namespace xls {
namespace {

// We store up to two "frontier nodes" on the frontier of the boolean
// computation -- if we have more than this many "frontier nodes", we switch
// over to "TooMany" via the TooManySentinel type below.
constexpr int64 kMaxFrontierNodes = 3;

}  // namespace

namespace internal {

TruthTable::TruthTable(const Bits& xyz_present, const Bits& xyz_negated,
                       absl::optional<Op> logical_op)
    : xyz_present_(xyz_present),
      xyz_negated_(xyz_negated),
      logical_op_(logical_op) {
  XLS_CHECK(!xyz_present.IsAllZeros());
  XLS_CHECK_EQ(3, xyz_present.bit_count());
  XLS_CHECK_EQ(3, xyz_negated.bit_count());
  if (!logical_op.has_value()) {
    XLS_CHECK_EQ(1, xyz_present.PopCount());
    XLS_CHECK_LE(xyz_negated.PopCount(), 1);
  }
  // Check for not-present bits that are negated.
  XLS_CHECK(
      bits_ops::And(bits_ops::Not(xyz_present), xyz_negated).IsAllZeros());
}

/* static */ Bits TruthTable::GetInitialVector(int64 i) {
  XLS_CHECK_LT(i, kMaxFrontierNodes);
  switch (i) {
    case 0:
      return UBits(0b00001111, /*bit_count=*/8);
    case 1:
      return UBits(0b00110011, /*bit_count=*/8);
    case 2:
      return UBits(0b01010101, /*bit_count=*/8);
  }
  XLS_LOG(FATAL) << "Unreachable.";
}

/* static */ Bits TruthTable::RunNaryOp(Op op,
                                        absl::Span<const Bits> operands) {
  switch (op) {
    case OP_AND:
      return bits_ops::NaryAnd(operands);
    case OP_OR:
      return bits_ops::NaryOr(operands);
    case OP_NAND:
      return bits_ops::NaryNand(operands);
    case OP_NOR:
      return bits_ops::NaryNor(operands);
    case OP_XOR:
      return bits_ops::NaryXor(operands);
    default:
      XLS_LOG(FATAL) << "Unhandled nary logical operation: " << op;
  }
}

Bits TruthTable::ComputeTruthTable() const {
  std::vector<Bits> operands;
  for (int64 i = 0; i < kMaxFrontierNodes; ++i) {
    if (xyz_present_.GetFromMsb(i)) {
      Bits bit_vector = GetInitialVector(i);
      if (xyz_negated_.GetFromMsb(i)) {
        bit_vector = bits_ops::Not(bit_vector);
      }
      operands.push_back(bit_vector);
    }
  }
  if (logical_op_.has_value()) {
    return RunNaryOp(logical_op_.value(), operands);
  }
  XLS_CHECK_EQ(1, operands.size());
  return operands[0];
}

bool TruthTable::MatchesVector(const Bits& table) const {
  return ComputeTruthTable() == table;
}

bool TruthTable::MatchesSymmetrical(
    Node* original, absl::Span<const Node* const> operands) const {
  if (logical_op_.has_value()) {
    if (original->op() != logical_op_.value()) {
      return false;
    }
    for (int64 i = 0; i < kMaxFrontierNodes; ++i) {
      if (!xyz_present_.GetFromMsb(i)) {
        continue;  // Don't care about this operand.
      }
      if (i >= operands.size()) {
        // Not enough operands to match this truth table.
        return false;
      }
      if (xyz_negated_.GetFromMsb(i)) {
        if (!AnyOperandWhere(
                original, [&](Node* o) { return IsNotOf(o, operands[i]); })) {
          return false;
        }
      } else {  // Present, not negated.
        if (!AnyOperandWhere(original,
                             [&](Node* o) { return o == operands[i]; })) {
          return false;
        }
      }
    }
    return true;
  }
  // When there is no logical function, only one of the "present" bits may be
  // set.
  XLS_CHECK_EQ(1, xyz_present_.PopCount());
  for (int64 i = 0; i < kMaxFrontierNodes; ++i) {
    if (!xyz_present_.GetFromMsb(i)) {
      continue;  // Don't care about this operand.
    }
    if (xyz_negated_.GetFromMsb(i)) {
      return IsNotOf(original, operands[i]);
    } else {
      return original == operands[i];
    }
  }
  XLS_LOG(FATAL) << "Unreachable.";
}

xabsl::StatusOr<Node*> TruthTable::CreateReplacement(
    const absl::optional<SourceLocation>& original_loc,
    absl::Span<Node* const> operands, Function* f) const {
  XLS_CHECK_LE(operands.size(), kMaxFrontierNodes);
  std::vector<Node*> this_operands;
  for (int64 i = 0; i < kMaxFrontierNodes; ++i) {
    if (!xyz_present_.GetFromMsb(i)) {
      continue;
    }
    Node* operand = operands[i];
    if (xyz_negated_.GetFromMsb(i)) {
      XLS_ASSIGN_OR_RETURN(operand,
                           f->MakeNode<UnOp>(original_loc, operand, OP_NOT));
    }
    this_operands.push_back(operand);
  }
  if (!logical_op_.has_value()) {
    XLS_CHECK_EQ(1, this_operands.size());
    return this_operands[0];
  }
  return f->MakeNode<NaryOp>(original_loc, this_operands, logical_op_.value());
}

}  // namespace internal

namespace {

using xls::internal::TruthTable;

// Indicates more than kMaxFrontierNodes are involved in a boolean expression.
struct TooManySentinel {};

// Type that notes the frontier nodes involved in a boolean expression.
using FrontierVector = absl::InlinedVector<Node*, kMaxFrontierNodes>;
using Frontier = absl::variant<FrontierVector, TooManySentinel>;

// Convenient converter of the Frontier type to string for debugging.
std::string ToString(const Frontier& x) {
  if (absl::holds_alternative<TooManySentinel>(x)) {
    return "Frontier(<too many>)";
  }
  const auto& nodes = absl::get<FrontierVector>(x);
  return absl::StrFormat("Frontier([%s])",
                         absl::StrJoin(nodes, ", ", NodeFormatter));
}

bool HasFrontierVector(const Frontier& frontier) {
  return absl::holds_alternative<FrontierVector>(frontier);
}
const FrontierVector& GetFrontierVector(const Frontier& frontier) {
  return absl::get<FrontierVector>(frontier);
}

// Adds a frontier "node" to the set -- note this may push it over to the
// "TooManySentinel" mode.
void AddNonBool(Node* node, Frontier* frontier) {
  if (absl::holds_alternative<TooManySentinel>(*frontier)) {
    return;
  }
  auto& nodes = absl::get<FrontierVector>(*frontier);
  if (std::find(nodes.begin(), nodes.end(), node) != nodes.end()) {
    return;  // Already present.
  }
  if (nodes.size() >= kMaxFrontierNodes) {
    *frontier = TooManySentinel{};
    return;
  }
  nodes.push_back(node);
}

// Unions the "arg" frontier set into the "out" frontier set. Note this may push
// it over to the "TooManySentinel" mode.
void Union(const Frontier& arg, Frontier* out) {
  if (absl::holds_alternative<TooManySentinel>(*out)) {
    return;
  }
  if (absl::holds_alternative<TooManySentinel>(arg)) {
    *out = TooManySentinel{};
    return;
  }
  for (Node* node : absl::get<FrontierVector>(arg)) {
    AddNonBool(node, out);
  }
}

// We arbitrarily call the node at index 0 in the frontier "X", and the one at
// index 1 "Y".
Node* GetX(const Frontier& frontier) {
  return absl::get<FrontierVector>(frontier)[0];
}
Node* GetY(const Frontier& frontier) {
  return absl::get<FrontierVector>(frontier)[1];
}
Node* GetZ(const Frontier& frontier) {
  const auto& v = absl::get<FrontierVector>(frontier);
  return v.size() > 2 ? v[2] : nullptr;
}

// DFS tracker for boolean value flow.
//
// When we arrive at a node we potentially replace it with a simpler boolean
// expression.
//
// * Performing the flow to find the composite logical operation is done by
//   FlowFromFrontierToNode().
// * Determining what sub-expression to replace the composite logical operation
//   with is done by ResolveTruthTable().
//
// Note that because we aggressively rewrite all intermediate logical operations
// to their simplest known form, we may increase the amount of gates required by
// the computation overall (we're breaking dependencies and simplifying
// composite expression instead of reusing all intermediaries in the way
// originally specified). Since we expect bitwise / boolean operations to be
// relatively cheap, this seems worthwhile.
class BooleanFlowTracker : public DfsVisitorWithDefault {
 public:
  absl::Status HandleNaryAnd(NaryOp* and_op) override {
    return HandleLogicOp(and_op);
  }
  absl::Status HandleNaryNand(NaryOp* nand_op) override {
    return HandleLogicOp(nand_op);
  }
  absl::Status HandleNaryNor(NaryOp* nor_op) override {
    return HandleLogicOp(nor_op);
  }
  absl::Status HandleNot(UnOp* not_op) override {
    return HandleLogicOp(not_op);
  }
  absl::Status HandleNaryOr(NaryOp* or_op) override {
    return HandleLogicOp(or_op);
  }

  static absl::flat_hash_map<uint64, TruthTable>* CreateTruthTables() {
    auto results = new absl::flat_hash_map<uint64, TruthTable>;
    auto add = [results](TruthTable table) {
      uint64 truth_table_bits = table.ComputeTruthTable().ToUint64().value();
      // Note: we don't update to the lowest-cost table, because other passes
      // (e.g. simplification passes) seem to do better with simpler operations
      // (e.g. or(x, y, z) instead of nand(~x, ~y, ~z) -- we also don't have a
      // very many overlapping truth tables that fail to follow the simple cost
      // ordering defined in the loops below.
      results->insert({truth_table_bits, table});
    };

    // Approximately in cheapest-to-more-expensive order.
    const auto kLogicOps = {OP_NAND, OP_NOR, OP_XOR, OP_AND, OP_OR};

    // First populate the truth tables for single values and their negations.
    for (int64 presence = 0b0001; presence < 0b1000; presence <<= 1) {
      for (bool negate : {false, true}) {
        add(TruthTable(UBits(presence, /*bit_count=*/3),
                       UBits(negate ? presence : 0LL, /*bit_count=*/3),
                       absl::nullopt));
      }
    }

    // Then populate the truth tables for the operations, approximately
    // sequenced from cheap to more expensive.
    for (Op op : kLogicOps) {
      for (int64 presence : {0b011, 0b101, 0b110, 0b111}) {
        const uint64 negation = 0;
        add(TruthTable(UBits(presence, /*bit_count=*/3),
                       UBits(negation, /*bit_count=*/3), op));
      }
    }
    // Now add operations that negate their operands.
    for (Op op : kLogicOps) {
      for (int64 presence : {0b011, 0b101, 0b110, 0b111}) {
        for (int64 negation = 1; negation < 0b1000; ++negation) {
          if ((~presence & negation)) {
            // Don't negate things that are not present.
            continue;
          }
          add(TruthTable(UBits(presence, /*bit_count=*/3),
                         UBits(negation, /*bit_count=*/3), op));
        }
      }
    }
    return results;
  }

  // Returns a memoized (constant) version of all the truth tables.
  static const absl::flat_hash_map<uint64, TruthTable>& GetTruthTables() {
    static absl::flat_hash_map<uint64, TruthTable>* tables =
        CreateTruthTables();
    return *tables;
  }

  // "operands" are the nodes on the frontier of the logical operations.
  xabsl::StatusOr<Node*> ResolveTruthTable(const Bits& bits,
                                           absl::Span<Node* const> operands,
                                           Node* original) {
    XLS_RET_CHECK(2 <= operands.size() && operands.size() <= 3);
    Function* f = original->function();
    if (bits.IsAllOnes()) {
      return f->MakeNode<Literal>(original->loc(),
                                  Value(SBits(-1, original->BitCountOrDie())));
    }
    if (bits.IsAllZeros()) {
      return f->MakeNode<Literal>(original->loc(),
                                  Value(UBits(0, original->BitCountOrDie())));
    }

    const auto& truth_tables = GetTruthTables();
    auto it = truth_tables.find(bits.ToUint64().value());
    if (it == truth_tables.end()) {
      return nullptr;  // No match.
    }
    const TruthTable& table = it->second;
    if (table.MatchesSymmetrical(original, operands)) {
      // Already in minimal form.
      return nullptr;
    }
    return table.CreateReplacement(original->loc(), operands, f);
  }

  absl::Status DefaultHandler(Node* node) override { return absl::OkStatus(); }

  absl::Span<const std::pair<Node*, Node*>> node_replacements() {
    return node_replacements_;
  }

 private:
  // Flows a truth table from the frontier to the output node "node".
  //
  // Starting at node, recursively invokes itself until it gets to the "frontier
  // nodes" at the frontier of the boolean computation. Those get initialized
  // with the full truth table of possibiliites for two variables:
  //
  //    X: 0 0 1 1 0 0 1 1
  //    Y: 0 1 0 1 0 1 0 1
  //    Z: 0 0 0 0 1 1 1 1
  //
  // Once we've pushed this vector of possiblities though all the intermediate
  // bitwise nodes what we're left with at "node" is the resulting logical
  // function. At that point we can just look up whether there's a simplified
  // expression of that logical function is and replace it accordingly, as we do
  // in ResolveTruthTable().
  xabsl::StatusOr<Bits> FlowFromFrontierToNode(const Frontier& frontier,
                                               Node* node) {
    if (node == GetX(frontier)) {
      return TruthTable::GetInitialVector(0);
    }
    if (node == GetY(frontier)) {
      return TruthTable::GetInitialVector(1);
    }
    if (node == GetZ(frontier)) {
      return TruthTable::GetInitialVector(2);
    }
    std::vector<Bits> operands;
    for (Node* operand : node->operands()) {
      XLS_ASSIGN_OR_RETURN(Bits result,
                           FlowFromFrontierToNode(frontier, operand));
      operands.push_back(result);
    }
    switch (node->op()) {
      case OP_AND:
        return bits_ops::NaryAnd(operands);
      case OP_OR:
        return bits_ops::NaryOr(operands);
      case OP_XOR:
        return bits_ops::NaryXor(operands);
      case OP_NAND:
        return bits_ops::NaryNand(operands);
      case OP_NOR:
        return bits_ops::NaryNor(operands);
      case OP_NOT:
        XLS_RET_CHECK(operands.size() == 1);
        return bits_ops::Not(operands[0]);
      default:
        XLS_LOG(FATAL) << "Expected node to be logical bitwise: " << node;
    }
  }

  absl::Status HandleLogicOp(Node* node) {
    // If there are >= 2 frontier nodes we flow the vectors from those to this
    // node to figure out its aggregate truth table.
    const Frontier& frontier = UnionOperandFrontier(node);
    if (HasFrontierVector(frontier) &&
        GetFrontierVector(frontier).size() >= 2) {
      const auto& operands = GetFrontierVector(frontier);
      XLS_ASSIGN_OR_RETURN(Bits result, FlowFromFrontierToNode(frontier, node));
      XLS_VLOG(3) << "Flow result for " << node << ": "
                  << result.ToString(FormatPreference::kBinary, true);
      XLS_ASSIGN_OR_RETURN(Node * replacement,
                           ResolveTruthTable(result, operands, node));
      if (replacement == nullptr) {
        return absl::OkStatus();
      }
      node_replacements_.push_back({node, replacement});
    }
    return absl::OkStatus();
  }

  // If a node's operand is not found in the node_to_frontier_ mapping, we
  // assume it is itself a frontier node.
  const Frontier& UnionOperandFrontier(Node* node) {
    Frontier accum;
    for (Node* operand : node->operands()) {
      auto it = node_to_frontier_.find(operand);
      if (it == node_to_frontier_.end()) {
        AddNonBool(operand, &accum);
      } else {
        Union(it->second, &accum);
      }
    }
    node_to_frontier_[node] = accum;
    return node_to_frontier_[node];
  }

  // Either a node has some limited number of "frontier nodes" on its frontier,
  // or we exceed the number and stop trying to track them.
  absl::flat_hash_map<Node*, Frontier> node_to_frontier_;

  std::vector<std::pair<Node*, Node*>> node_replacements_;
};

}  // namespace

xabsl::StatusOr<bool> BooleanSimplificationPass::RunOnFunction(
    Function* f, const PassOptions& options, PassResults* results) const {
  BooleanFlowTracker visitor;
  XLS_RETURN_IF_ERROR(f->Accept(&visitor));
  for (auto& pair : visitor.node_replacements()) {
    Node* node = pair.first;
    Node* replacement = pair.second;
    XLS_VLOG(3) << "Replacing " << node << " with " << replacement;
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(replacement).status());
  }
  return !visitor.node_replacements().empty();
}

}  // namespace xls
