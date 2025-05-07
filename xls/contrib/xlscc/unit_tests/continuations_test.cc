// Copyright 2025 The XLS Authors
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

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "clang/include/clang/AST/Decl.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

class ContinuationsTest : public XlsccTestBase {
 public:
  absl::StatusOr<const xlscc::GeneratedFunction*> GenerateTopFunction(
      std::string_view content) {
    XLS_ASSIGN_OR_RETURN(xls::TempFile temp,
                         xls::TempFile::CreateWithContent(content, ".cc"));

    XLS_RETURN_IF_ERROR(
        ScanFile(temp, /* clang_argv= */ {}, /*io_test_mode=*/true,
                 /*error_on_init_interval=*/false,
                 /*error_on_uninitialized=*/false,
                 /*loc=*/xls::SourceLocation(),
                 /*fail_xlscc_check=*/false, /*max_unroll_iters=*/0));

    package_ = std::make_unique<xls::Package>("my_package");
    absl::flat_hash_map<const clang::NamedDecl*, xlscc::ChannelBundle>
        top_channel_injections = {};

    XLS_ASSIGN_OR_RETURN(xlscc::GeneratedFunction * func,
                         translator_->GenerateIR_Top_Function(
                             package_.get(), top_channel_injections));

    LOG(INFO) << "Package IR: ";
    LOG(INFO) << package_->DumpIr();

    LogContinuations(func);

    return func;
  }

  bool SliceOutputsDecl(const xlscc::GeneratedFunctionSlice& slice,
                        std::string_view name) {
    for (const xlscc::ContinuationValue& continuation_out :
         slice.continuations_out) {
      if (continuation_out.decl == nullptr) {
        continue;
      }
      if (continuation_out.decl->getNameAsString() == name) {
        return true;
      }
    }
    return false;
  };

  void LogContinuations(xlscc::GeneratedFunction* func) {
    for (const xlscc::GeneratedFunctionSlice& slice : func->slices) {
      LOG(INFO) << "Slice continuations:";
      for (const xlscc::ContinuationValue* continuation_in :
           slice.continuations_in) {
        CHECK_NE(continuation_in, nullptr);
        LOG(INFO) << "-- in: " << continuation_in->name << ": "
                  << continuation_in->output_node->ToString();
      }
      for (const xlscc::ContinuationValue& continuation_out :
           slice.continuations_out) {
        LOG(INFO) << "-- out " << continuation_out.name << ": "
                  << continuation_out.output_node->ToString();
      }
    }
  }
};

// TODO(seanhaskell): Check remove unused continuation
// TODO(seanhaskell): Check literals available
// TODO(seanhaskell): Pipelined loop phi
// TODO(seanhaskell): DISABLED_IOOutputNodesDoNotMakeContinuations
// TODO(seanhaskell): DISABLED_OneContinuationOutputPerNode
// TODO(seanhaskell): Check names, incl LValues
// TODO(seanhaskell): Check that single bit variable used as a condition gets
// its own continuation without a decl, so that its state element doesn't get
// overwritten
// TODO(seanhaskell): Check that loops preserve values declared before them and
// used after them (state element isn't shared with assignment in the loop)

// TODO(seanhaskell): Implement continuation optimization
TEST_F(ContinuationsTest, DISABLED_OneContinuationOutputPerNode) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int x = in.read();
      const int y = x;
      out.write(x + y);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  for (const xlscc::GeneratedFunctionSlice& slice : func->slices) {
    absl::flat_hash_set<const xls::Node*> nodes_seen;
    for (const xlscc::ContinuationValue& continuation_out :
         slice.continuations_out) {
      EXPECT_FALSE(nodes_seen.contains(continuation_out.output_node));
      nodes_seen.insert(continuation_out.output_node);
    }
  }
}

// TODO(seanhaskell): Implement continuation optimization
TEST_F(ContinuationsTest, DISABLED_IOOutputNodesDoNotMakeContinuations) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int x = in.read();
      out.write(x);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  absl::flat_hash_set<const xls::Node*> io_output_nodes;
  for (const xlscc::IOOp& op : func->io_ops) {
    ASSERT_TRUE(op.ret_value.valid());
    io_output_nodes.insert(op.ret_value.node());
  }

  for (const xlscc::GeneratedFunctionSlice& slice : func->slices) {
    for (const xlscc::ContinuationValue& continuation_out :
         slice.continuations_out) {
      EXPECT_FALSE(io_output_nodes.contains(continuation_out.output_node));
    }
  }
}

TEST_F(ContinuationsTest, ContinuationsFollowScope) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int y = 0;
      {
        const int x = in.read();
        out.write(x);
        y += x;
      }
      out.write(y);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_GE(func->slices.size(), 3);

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;

  EXPECT_FALSE(SliceOutputsDecl(first_slice, "x"));
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "x"));
  EXPECT_FALSE(SliceOutputsDecl(third_slice, "x"));
}

TEST_F(ContinuationsTest, Determinism) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int x = in.read();
      const int y = in.read() + 111;
      out.write(3*x + y);
      out.write(y);
    })";

  std::vector<const xls::Node*> nodes_first_order;
  std::vector<std::unique_ptr<xls::Package>> packages;

  for (int64_t trials = 0; trials < 10; ++trials) {
    XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                             GenerateTopFunction(content));
    // Keep packages alive for each trial so pointers remain valid
    packages.push_back(std::move(package_));

    std::vector<const xls::Node*> nodes_this_trial;
    for (const xlscc::GeneratedFunctionSlice& slice : func->slices) {
      for (const xlscc::ContinuationValue& continuation_out :
           slice.continuations_out) {
        ASSERT_NE(continuation_out.output_node, nullptr);
        nodes_this_trial.push_back(continuation_out.output_node);
      }
      for (const xlscc::ContinuationValue* continuation_in :
           slice.continuations_in) {
        ASSERT_NE(continuation_in->output_node, nullptr);
        nodes_this_trial.push_back(continuation_in->output_node);
      }
    }
    ASSERT_GT(nodes_this_trial.size(), 0);
    if (nodes_first_order.empty()) {
      nodes_first_order = nodes_this_trial;
      continue;
    }
    ASSERT_EQ(nodes_first_order.size(), nodes_this_trial.size());
    for (int64_t i = 0; i < nodes_first_order.size(); ++i) {
      const xls::Node* first_order_node = nodes_first_order.at(i);
      const xls::Node* this_trial_node = nodes_this_trial.at(i);

      EXPECT_EQ(first_order_node->GetName(), this_trial_node->GetName());
      EXPECT_TRUE(
          first_order_node->GetType()->IsEqualTo(this_trial_node->GetType()));
    }
  }
}

}  // namespace
}  // namespace xls
