// Copyright 2024 The XLS Authors
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

#include "xls/codegen/materialize_fifos_pass.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls::verilog {
namespace {
using testing::AllOf;
using testing::Contains;
using testing::Pair;

MATCHER_P2(UsableOutputsMatch, input_set, output_set,
           "All ready/valid signals to match and the data signal to match if "
           "ready & valid is asserted.") {
  const absl::flat_hash_map<std::string, Value>& inputs = input_set;
  const absl::flat_hash_map<std::string, Value>& outputs = output_set;
  if (outputs.at(FifoInstantiation::kPopValidPortName).bits().IsOne() &&
      inputs.at(FifoInstantiation::kPopReadyPortName).bits().IsOne()) {
    return testing::ExplainMatchResult(
        testing::UnorderedElementsAreArray(outputs), arg, result_listener);
  }
  // It would be nice to check the pop_data value is the same too but it
  // doesn't seem useful if the data can't be read.
  return testing::ExplainMatchResult(
      AllOf(Contains(Pair(FifoInstantiation::kPopValidPortName,
                          outputs.at(FifoInstantiation::kPopValidPortName))),
            Contains(Pair(FifoInstantiation::kPushReadyPortName,
                          outputs.at(FifoInstantiation::kPushReadyPortName)))),
      arg, result_listener);
}

class BaseOperation {
 public:
  virtual ~BaseOperation() = default;
  virtual absl::flat_hash_map<std::string, Value> InputSet() const = 0;
};
struct Push : public BaseOperation {
  explicit Push(int32_t v) : v(v) {}
  int32_t v;

  absl::flat_hash_map<std::string, Value> InputSet() const override {
    return {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushDataPortName),
         Value(UBits(v, 32))},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushValidPortName), Value::Bool(true)},
    };
  }
};
struct Pop : public BaseOperation {
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    return {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushDataPortName),
         Value(UBits(0xf0f0f0f0, 32))},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushValidPortName),
         Value::Bool(false)},
    };
  }
};
struct PushAndPop : public BaseOperation {
  explicit PushAndPop(int32_t v) : v(v) {}
  uint32_t v;
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    return {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushDataPortName),
         Value(UBits(v, 32))},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushValidPortName), Value::Bool(true)},
    };
  }
};
struct NotReady : public BaseOperation {
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    return {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushDataPortName),
         Value(UBits(987654321, 32))},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushValidPortName),
         Value::Bool(false)},
    };
  }
};
struct Reset : public BaseOperation {
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    return {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushDataPortName),
         Value(UBits(123456789, 32))},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushValidPortName),
         Value::Bool(false)},
    };
  }
};
struct ResetPop : public BaseOperation {
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    return {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushDataPortName),
         Value(UBits(123456789, 32))},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushValidPortName),
         Value::Bool(false)},
    };
  }
};
struct ResetPush : public BaseOperation {
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    return {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushDataPortName),
         Value(UBits(123456789, 32))},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(false)},
        {std::string(FifoInstantiation::kPushValidPortName), Value::Bool(true)},
    };
  }
};
struct ResetPushPop : public BaseOperation {
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    return {
        {std::string(FifoInstantiation::kResetPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushDataPortName),
         Value(UBits(123456789, 32))},
        {std::string(FifoInstantiation::kPopReadyPortName), Value::Bool(true)},
        {std::string(FifoInstantiation::kPushValidPortName), Value::Bool(true)},
    };
  }
};
class Operation : public BaseOperation,
                  public std::variant<Push, Pop, PushAndPop, NotReady, Reset,
                                      ResetPush, ResetPop, ResetPushPop> {
 public:
  using std::variant<Push, Pop, PushAndPop, NotReady, Reset, ResetPush,
                     ResetPop, ResetPushPop>::variant;
  absl::flat_hash_map<std::string, Value> InputSet() const override {
    return std::visit([&](auto v) { return v.InputSet(); }, *this);
  }
};

class MaterializeFifosPassTestHelper {
 public:
  virtual ~MaterializeFifosPassTestHelper() = default;
  virtual std::string TestName() const = 0;
  virtual std::unique_ptr<Package> CreatePackage() const = 0;
  // NB The block-jit internally uses this pass to implement fifos so it
  // cannot be used as an oracle.
  static constexpr const BlockEvaluator& kOracleEvaluator =
      kInterpreterBlockEvaluator;
  static constexpr const BlockEvaluator& kTestEvaluator =
      kInterpreterBlockEvaluator;
  absl::StatusOr<Block*> MakeOracleBlock(Package* p, FifoConfig config) {
    return MakeOracleBlock(p, config, p->GetBitsType(32));
  }
  absl::StatusOr<Block*> MakeOracleBlock(Package* p, FifoConfig config,
                                         Type* ty,
                                         std::string_view name = "oracle") {
    BlockBuilder bb(uniq_.GetSanitizedUniqueName(
                        absl::StrFormat("%s_%s", name, TestName())),
                    p);
    XLS_ASSIGN_OR_RETURN(auto inst,
                         bb.block()->AddFifoInstantiation("fifo", config, ty));
    XLS_ASSIGN_OR_RETURN(InstantiationType inst_ty, inst->type());
    for (const auto& [port_name, in_ty] : inst_ty.input_types()) {
      bb.InstantiationInput(inst, port_name, bb.InputPort(port_name, in_ty));
    }
    for (const auto& [port_name, out_ty] : inst_ty.output_types()) {
      bb.OutputPort(port_name, bb.InstantiationOutput(inst, port_name));
    }
    return bb.Build();
  }
  absl::StatusOr<Block*> MakeFifoBlock(Package* p, FifoConfig config) {
    return MakeFifoBlock(p, config, p->GetBitsType(32));
  }
  absl::StatusOr<Block*> MakeFifoBlock(Package* p, FifoConfig config,
                                       Type* ty) {
    XLS_ASSIGN_OR_RETURN(Block * wrapper,
                         MakeOracleBlock(p, config, ty, "test"));

    MaterializeFifosPass mfp;
    CodegenPassUnit pu(p, wrapper);
    CodegenPassOptions opt;
    opt.codegen_options.reset("rst", /*asynchronous=*/false,
                              /*active_low=*/false, /*reset_data_path=*/false);
    CodegenPassResults res;
    XLS_ASSIGN_OR_RETURN(auto changed, mfp.Run(&pu, opt, &res));
    XLS_RET_CHECK(changed);
    XLS_RET_CHECK_EQ(pu.top_block->GetInstantiations().size(), 1);
    XLS_ASSIGN_OR_RETURN(
        auto bi,
        pu.top_block->GetInstantiations().front()->AsBlockInstantiation());
    return bi->instantiated_block();
  }

  template <typename... Args>
  absl::StatusOr<std::unique_ptr<BlockContinuation>> MakeOracle(Args... args) {
    XLS_ASSIGN_OR_RETURN(Block * oracle, MakeOracleBlock(args...));
    // TODO(allight): Ideally we could use the actual verilog to act as an
    // oracle but this is complicated.
    return kOracleEvaluator.NewContinuation(oracle);
  }

  absl::StatusOr<std::unique_ptr<BlockContinuation>> MakeTestEval(Block* b) {
    return kTestEvaluator.NewContinuation(b);
  }

  // Compare against the interpreter as an oracle.
  void RunTestVector(FifoConfig config,
                     absl::Span<Operation const> operations) {
    auto p = CreatePackage();
    XLS_ASSERT_OK_AND_ASSIGN(Block * fifo_block,
                             MakeFifoBlock(p.get(), config));
    testing::Test::RecordProperty("test_block", p->DumpIr());
    XLS_ASSERT_OK_AND_ASSIGN(auto tester, MakeTestEval(fifo_block));
    // NB Oracle is in a different package to make sure nothing runs over it.
    auto oracle_package = CreatePackage();
    XLS_ASSERT_OK_AND_ASSIGN(auto oracle,
                             MakeOracle(oracle_package.get(), config));
    testing::Test::RecordProperty("oracle", oracle_package->DumpIr());
    int64_t i = 0;
    for (const BaseOperation& op : operations) {
      ScopedMaybeRecord op_cnt("op_cnt", i);
      ScopedMaybeRecord init_state("initial_state", tester->registers());
      auto test_res = tester->RunOneCycle(op.InputSet());
      auto oracle_res = oracle->RunOneCycle(op.InputSet());
      ASSERT_THAT(std::make_pair(test_res, oracle_res),
                  testing::Pair(status_testing::IsOk(), status_testing::IsOk()))
          << "@i=" << i;
      auto tester_outputs = tester->output_ports();
      ScopedMaybeRecord state("out_state", tester->registers());
      ScopedMaybeRecord input("input", op.InputSet());
      ASSERT_THAT(tester_outputs,
                  UsableOutputsMatch(op.InputSet(), oracle->output_ports()))
          << "@" << i << ". tester_state: {"
          << absl::StrJoin(tester->registers(), ", ", absl::PairFormatter("="))
          << "}";
      ++i;
    }
  }

  NameUniquer uniq_ = NameUniquer("___");
};
class MaterializeFifosPassTest : public MaterializeFifosPassTestHelper,
                                 public IrTestBase {
 public:
  std::string TestName() const override { return IrTestBase::TestName(); }
  std::unique_ptr<Package> CreatePackage() const override {
    return IrTestBase::CreatePackage();
  }
};

TEST_F(MaterializeFifosPassTest, BasicFifo) {
  FifoConfig cfg(3, false, false, false);
  RunTestVector(cfg, {
                         Push{1},
                         Push{2},
                         Push{3},
                         NotReady{},
                         NotReady{},
                         NotReady{},
                         Pop{},
                         Pop{},
                         Pop{},
                         // Out of elements.
                         Pop{},
                         Pop{},
                         PushAndPop{4},
                         PushAndPop{5},
                         PushAndPop{6},
                         PushAndPop{7},
                         Pop{},
                         // Out of elements
                         Pop{},
                         Pop{},
                         Push{8},
                         Push{9},
                         Push{10},
                         // Full
                         Push{11},
                         Push{12},
                         Push{13},
                         Pop{},
                         Pop{},
                         Pop{},
                         // Out of elements.
                         Pop{},
                         Pop{},
                     });
}

TEST_F(MaterializeFifosPassTest, BasicFifoFull) {
  FifoConfig cfg(3, false, false, false);
  RunTestVector(cfg, {
                         Push{1},
                         Push{2},
                         Push{3},
                         Push{4},
                         Push{5},
                         Push{6},
                         Push{7},
                         PushAndPop{8},
                         PushAndPop{9},
                         PushAndPop{10},
                         PushAndPop{11},
                         Pop{},
                         Pop{},
                         Pop{},
                         Pop{},
                         Pop{},
                         Pop{},
                     });
}

TEST_F(MaterializeFifosPassTest, FifoLengthIsSmall) {
  // If length is near the overflow point for the bit-width mistakes in mod
  // calculation are possible.
  FifoConfig cfg(2, false, false, false);
  RunTestVector(cfg,
                {PushAndPop{1}, PushAndPop{2}, PushAndPop{3}, PushAndPop{4}});
}

TEST_F(MaterializeFifosPassTest, ResetFullFifo) {
  FifoConfig cfg(3, false, false, false);
  RunTestVector(cfg, {
                         Push{1},
                         Push{2},
                         Push{3},
                         Push{4},
                         Push{5},
                         Push{6},
                         Push{7},
                         PushAndPop{8},
                         PushAndPop{9},
                         PushAndPop{10},
                         PushAndPop{11},
                         Reset{},
                         NotReady{},
                         NotReady{},
                         NotReady{},
                         NotReady{},
                         NotReady{},
                         NotReady{},
                     });
}

class MaterializeFifosPassFuzzTest : public MaterializeFifosPassTestHelper {
 public:
  std::unique_ptr<Package> CreatePackage() const override {
    return std::make_unique<VerifiedPackage>("FuzzTest");
  }
  std::string TestName() const override { return "FuzzTest"; }
};
auto OperationDomain() {
  return fuzztest::OneOf(
      fuzztest::ConstructorOf<Operation>(fuzztest::ConstructorOf<Pop>()),
      fuzztest::ConstructorOf<Operation>(fuzztest::ConstructorOf<NotReady>()),
      fuzztest::ConstructorOf<Operation>(
          fuzztest::ConstructorOf<Push>(fuzztest::InRange(1, 1000))),
      fuzztest::ConstructorOf<Operation>(
          fuzztest::ConstructorOf<PushAndPop>(fuzztest::InRange(1, 1000))),
      fuzztest::ConstructorOf<Operation>(fuzztest::ConstructorOf<Reset>()),
      fuzztest::ConstructorOf<Operation>(fuzztest::ConstructorOf<ResetPush>()),
      fuzztest::ConstructorOf<Operation>(fuzztest::ConstructorOf<ResetPop>()),
      fuzztest::ConstructorOf<Operation>(
          fuzztest::ConstructorOf<ResetPushPop>()));
}
void FuzzTestBasicFifo(uint32_t depth, const std::vector<Operation>& ops) {
  FifoConfig cfg(depth, false, false, false);
  MaterializeFifosPassFuzzTest fixture;
  fixture.RunTestVector(cfg, ops);
}
FUZZ_TEST(MaterializeFifosPassFuzzTest, FuzzTestBasicFifo)
    .WithDomains(fuzztest::InRange(1, 10),
                 fuzztest::VectorOf(OperationDomain()).WithMaxSize(1000));

}  // namespace
}  // namespace xls::verilog
