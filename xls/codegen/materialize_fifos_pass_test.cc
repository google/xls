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
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/fifo_model_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"

namespace xls::verilog {
namespace {

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
  absl::StatusOr<Block*> MakeOracleBlock(Package* p, FifoTestParam params,
                                         std::string_view name = "oracle") {
    BlockBuilder bb(uniq_.GetSanitizedUniqueName(
                        absl::StrFormat("%s_%s", name, TestName())),
                    p);
    XLS_ASSIGN_OR_RETURN(auto inst, bb.block()->AddFifoInstantiation(
                                        "fifo", params.config,
                                        p->GetBitsType(params.data_bit_count)));
    XLS_ASSIGN_OR_RETURN(InstantiationType inst_ty, inst->type());
    for (const auto& [port_name, in_ty] : inst_ty.input_types()) {
      bb.InstantiationInput(inst, port_name, bb.InputPort(port_name, in_ty));
    }
    for (const auto& [port_name, out_ty] : inst_ty.output_types()) {
      bb.OutputPort(port_name, bb.InstantiationOutput(inst, port_name));
    }
    return bb.Build();
  }
  absl::StatusOr<Block*> MakeFifoBlock(Package* p, FifoTestParam params,
                                       std::string_view reset_name) {
    XLS_ASSIGN_OR_RETURN(Block * wrapper, MakeOracleBlock(p, params, "test"));

    MaterializeFifosPass mfp;
    CodegenPassUnit pu(p, wrapper);
    CodegenPassOptions opt;
    opt.codegen_options.reset(reset_name, /*asynchronous=*/false,
                              /*active_low=*/false, /*reset_data_path=*/false);
    CodegenPassResults res;
    XLS_ASSIGN_OR_RETURN(auto changed, mfp.Run(&pu, opt, &res));
    XLS_RET_CHECK(changed);
    XLS_RET_CHECK_EQ(pu.top_block()->GetInstantiations().size(), 1);
    XLS_ASSIGN_OR_RETURN(
        auto bi,
        pu.top_block()->GetInstantiations().front()->AsBlockInstantiation());
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
  void RunTestVector(
      FifoTestParam config, absl::Span<Operation const> operations,
      std::string_view reset_name = FifoInstantiation::kResetPortName) {
    auto p = CreatePackage();
    XLS_ASSERT_OK_AND_ASSIGN(Block * fifo_block,
                             MakeFifoBlock(p.get(), config, reset_name));
    testing::Test::RecordProperty("test_block", p->DumpIr());
    XLS_ASSERT_OK_AND_ASSIGN(auto tester, MakeTestEval(fifo_block));
    // NB Oracle is in a different package to make sure nothing runs over it.
    auto oracle_package = CreatePackage();
    XLS_ASSERT_OK_AND_ASSIGN(auto oracle,
                             MakeOracle(oracle_package.get(), config));
    testing::Test::RecordProperty("oracle", oracle_package->DumpIr());
    int64_t i = 0;
    for (const BaseOperation& orig_op : operations) {
      // Ensure applied operation has the expected type:
      std::unique_ptr<BaseOperation> op =
          orig_op.WithBitWidth(config.data_bit_count);
      auto input_set_inst = op->InputSet();

      // Note that the name of the materialized reset port can differ from
      // the fifo instantiation reset port name, which is always
      // FifoInstantiation::kResetPortName.
      auto input_set_materialized = input_set_inst;
      auto node =
          input_set_materialized.extract(FifoInstantiation::kResetPortName);
      node.key() = std::string(reset_name);
      input_set_materialized.insert(std::move(node));

      ScopedMaybeRecord op_cnt("op_cnt", i);
      ScopedMaybeRecord init_state("initial_state", tester->registers());
      auto test_res = tester->RunOneCycle(input_set_materialized);
      auto oracle_res = oracle->RunOneCycle(input_set_inst);
      ASSERT_THAT(std::make_pair(test_res, oracle_res),
                  testing::Pair(absl_testing::IsOk(), absl_testing::IsOk()))
          << "@i=" << i;
      auto tester_outputs = tester->output_ports();
      ScopedMaybeRecord state("out_state", tester->registers());
      ScopedMaybeRecord input_inst("input", input_set_inst);
      ScopedMaybeRecord test_out("test_output", tester_outputs);
      ScopedMaybeRecord oracle_out("oracle_output", oracle->output_ports());
      ScopedMaybeRecord trace("trace", tester->events().trace_msgs);
      ASSERT_THAT(tester_outputs,
                  UsableOutputsMatch(input_set_inst, oracle->output_ports()))
          << "@" << i << ". tester_state: {"
          << absl::StrJoin(tester->registers(), ", ", absl::PairFormatter("="))
          << "}";
      ++i;
    }
  }

  NameUniquer uniq_ = NameUniquer("___");
};

class MaterializeFifosPassTest
    : public MaterializeFifosPassTestHelper,
      public IrTestBase,
      public testing::WithParamInterface<FifoTestParam> {
 public:
  std::string TestName() const override { return IrTestBase::TestName(); }
  std::unique_ptr<Package> CreatePackage() const override {
    return IrTestBase::CreatePackage();
  }
};

TEST_P(MaterializeFifosPassTest, BasicFifo) {
  RunTestVector(GetParam(), {
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

TEST_P(MaterializeFifosPassTest, BasicFifoFull) {
  RunTestVector(GetParam(), {
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

TEST_P(MaterializeFifosPassTest, FifoLengthIsSmall) {
  // If length is near the overflow point for the bit-width mistakes in mod
  // calculation are possible.
  RunTestVector(GetParam(),
                {PushAndPop{1}, PushAndPop{2}, PushAndPop{3}, PushAndPop{4}});
}

TEST_F(MaterializeFifosPassTest, ResetFullFifo) {
  FifoTestParam cfg{32, {3, false, false, false}};
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
                         ResetOp{},
                         NotReady{},
                         NotReady{},
                         NotReady{},
                         NotReady{},
                         NotReady{},
                         NotReady{},
                     });
}

TEST_F(MaterializeFifosPassTest, ResetFullFifoWithDifferentResetName) {
  FifoTestParam cfg{32, {3, false, false, false}};
  RunTestVector(cfg,
                {
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
                    ResetOp{},
                    NotReady{},
                    NotReady{},
                    NotReady{},
                    NotReady{},
                    NotReady{},
                    NotReady{},
                },
                /*reset_name=*/"foobar");
}

TEST_P(MaterializeFifosPassTest, BasicFifoBypass) {
  RunTestVector(GetParam(), {
                                PushAndPop{1},
                                PushAndPop{2},
                                PushAndPop{3},
                                PushAndPop{4},
                                PushAndPop{5},
                                PushAndPop{6},
                                PushAndPop{7},
                                PushAndPop{8},
                                Pop{},
                                Pop{},
                                Pop{},
                                Pop{},
                                Pop{},
                                Pop{},
                            });
}

inline std::vector<FifoTestParam> GenerateMaterializeFifoTestParams() {
  std::vector<FifoTestParam> params;
  for (int64_t data_bit_count : {0, 32}) {
    for (int64_t depth : {0, 3, 5}) {
      for (bool bypass : {true, false}) {
        for (bool register_push_outputs : {true, false}) {
          for (bool register_pop_outputs : {true, false}) {
            if (depth == 0 &&
                (!bypass || register_push_outputs || register_pop_outputs)) {
              // Unsupported configurations of depth=0 fifos.
              continue;
            }
            if (depth == 1 && register_pop_outputs) {
              // Unsupported configuration of depth=1 fifo with
              // register_pop_outputs.
              continue;
            }
            params.push_back(FifoTestParam{
                .data_bit_count = data_bit_count,
                .config = FifoConfig(depth, bypass, register_push_outputs,
                                     register_pop_outputs)});
          }
        }
      }
    }
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(MaterializeFifosPassTest, MaterializeFifosPassTest,
                         testing::ValuesIn(GenerateMaterializeFifoTestParams()),
                         testing::PrintToStringParamName());

class MaterializeFifosPassFuzzTest : public MaterializeFifosPassTestHelper {
 public:
  std::unique_ptr<Package> CreatePackage() const override {
    return std::make_unique<VerifiedPackage>("FuzzTest");
  }
  std::string TestName() const override { return "FuzzTest"; }
};
void FuzzTestFifo(FifoTestParam cfg, const std::vector<Operation>& ops) {
  MaterializeFifosPassFuzzTest fixture;
  fixture.RunTestVector(cfg, ops);
}

FUZZ_TEST(MaterializeFifosPassFuzzTest, FuzzTestFifo)
    .WithDomains(FifoTestParamDomain(),
                 fuzztest::VectorOf(OperationDomain()).WithMaxSize(1000));

TEST(FifoFuzzTest, FuzzTestJitRegression) {
  FuzzTestFifo(FifoTestParam{32, xls::FifoConfig(1, false, false, false)},
               {Operation(NotReady()), Operation(Pop())});
}

TEST(FifoFuzzTest, FuzzTestJitRegression2) {
  FuzzTestFifo(FifoTestParam{32, xls::FifoConfig(10, true, false, false)},
               {Operation(ResetOp()), Operation(ResetOp())});
}

}  // namespace
}  // namespace xls::verilog
