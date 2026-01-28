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

#include <algorithm>
#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "clang/include/clang/AST/Decl.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/tracked_bvalue.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/translator_types.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xlscc {
namespace {

class ContinuationsTest : public XlsccTestBase {
 public:
  absl::StatusOr<const xlscc::GeneratedFunction*> GenerateTopFunction(
      std::string_view content) {
    generate_new_fsm_ = true;

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

    LogContinuations(*func);
    testing::Test::RecordProperty("GraphViz file",
                                  Debug_GenerateSliceGraph(*func));

    return func;
  }

  bool SliceOutputsDecl(const xlscc::GeneratedFunctionSlice& slice,
                        std::string_view name,
                        std::optional<bool> direct_in = std::nullopt) {
    for (const xlscc::ContinuationValue& continuation_out :
         slice.continuations_out) {
      for (const DeclLeaf& decl : continuation_out.decls) {
        if (decl.decl->getNameAsString() == name &&
            (!direct_in.has_value() ||
             direct_in.value() == continuation_out.direct_in)) {
          return true;
        }
      }
    }
    return false;
  };

  int64_t SliceOutputsDeclCount(const xlscc::GeneratedFunctionSlice& slice,
                                std::string_view name) {
    int64_t count = 0;
    for (const xlscc::ContinuationValue& continuation_out :
         slice.continuations_out) {
      for (const DeclLeaf& decl : continuation_out.decls) {
        if (decl.decl->getNameAsString() == name) {
          ++count;
        }
      }
    }
    return count;
  };

  static absl::flat_hash_map<const ContinuationValue*, int64_t>
  GetSliceIndicesByValue(const GeneratedFunction& func) {
    absl::flat_hash_map<const ContinuationValue*, int64_t> slice_index_by_value;
    {
      int64_t slice_index = 0;
      for (const GeneratedFunctionSlice& full_slice : func.slices) {
        for (const ContinuationValue& cont_out : full_slice.continuations_out) {
          slice_index_by_value[&cont_out] = slice_index;
        }
        ++slice_index;
      }
    }

    return slice_index_by_value;
  }

  // func must be specified when is_feedback is specified
  bool SliceInputsDecl(
      const xlscc::GeneratedFunctionSlice& slice, std::string_view name,
      std::optional<bool> direct_in = std::nullopt,
      std::optional<bool> is_feedback = std::nullopt,
      std::optional<const GeneratedFunction*> func = std::nullopt,
      std::optional<int64_t> decl_index = std::nullopt) {
    auto check_is_feedback =
        [&func,
         &slice](const xlscc::ContinuationInput& continuation_in) -> bool {
      CHECK(func.has_value());
      const xlscc::GeneratedFunction& full_func = *func.value();
      std::optional<int64_t> input_slice_index = std::nullopt;
      absl::flat_hash_map<const ContinuationValue*, int64_t>
          slice_index_by_value = GetSliceIndicesByValue(full_func);
      {
        int64_t slice_index = 0;
        for (const GeneratedFunctionSlice& full_slice : full_func.slices) {
          if (&slice == &full_slice) {
            input_slice_index = slice_index;
          }
          ++slice_index;
        }
      }
      CHECK(input_slice_index.has_value());
      CHECK(slice_index_by_value.contains(continuation_in.continuation_out));
      return input_slice_index <=
             slice_index_by_value.at(continuation_in.continuation_out);
    };

    for (const xlscc::ContinuationInput& continuation_in :
         slice.continuations_in) {
      absl::flat_hash_set<DeclLeaf> all_decls = continuation_in.decls;

      all_decls.insert(continuation_in.continuation_out->decls.begin(),
                       continuation_in.continuation_out->decls.end());

      for (const DeclLeaf& decl : all_decls) {
        if (decl.decl->getNameAsString() == name &&
            (!direct_in.has_value() ||
             direct_in.value() ==
                 continuation_in.continuation_out->direct_in) &&
            (!is_feedback.has_value() ||
             is_feedback.value() == check_is_feedback(continuation_in)) &&
            (!decl_index.has_value() ||
             decl_index.value() == decl.leaf_index)) {
          return true;
        }
      }
    }
    return false;
  }

  int64_t SliceInputsDeclCount(const xlscc::GeneratedFunctionSlice& slice,
                               std::string_view name) {
    int64_t count = 0;
    for (const xlscc::ContinuationInput& continuation_in :
         slice.continuations_in) {
      absl::flat_hash_set<DeclLeaf> all_decls = continuation_in.decls;

      all_decls.insert(continuation_in.continuation_out->decls.begin(),
                       continuation_in.continuation_out->decls.end());

      for (const DeclLeaf& decl : all_decls) {
        if (decl.decl->getNameAsString() == name) {
          ++count;
        }
      }
    }
    return count;
  }

  bool SliceInputDoesNotInputBothDecls(
      const xlscc::GeneratedFunctionSlice& slice, std::string_view name_a,
      std::string_view name_b) {
    absl::flat_hash_set<DeclLeaf> decls_found;
    bool found = false;
    for (const xlscc::ContinuationInput& continuation_in :
         slice.continuations_in) {
      absl::flat_hash_set<DeclLeaf> all_decls = continuation_in.decls;

      all_decls.insert(continuation_in.continuation_out->decls.begin(),
                       continuation_in.continuation_out->decls.end());

      for (const DeclLeaf& decl : all_decls) {
        if (decl.decl->getNameAsString() == name_a) {
          decls_found = continuation_in.decls;
          found = true;
          continue;
        }
      }
    }
    if (!found) {
      return true;
    }
    for (const DeclLeaf& decl : decls_found) {
      if (decl.decl->getNameAsString() == name_b) {
        return false;
      }
    }
    return true;
  }
};

template <typename MapT>
std::vector<typename MapT::mapped_type*> OrderedBValuesForMap(MapT& map) {
  absl::flat_hash_set<typename MapT::mapped_type*> bvals;
  for (auto& [_, bval] : map) {
    bvals.insert(&bval);
  }
  return TrackedBValue::OrderBValues(bvals);
}

template <typename MapT>
std::vector<const typename MapT::mapped_type*> OrderedCValuesForMap(MapT& map) {
  absl::flat_hash_set<const typename MapT::mapped_type*> bvals;
  for (auto& [_, bval] : map) {
    bvals.insert(&bval);
  }
  return xlscc::OrderCValuesFunc()(bvals);
}

// TODO(seanhaskell): Don't continue IO output
// TODO(seanhaskell): DISABLED_IOOutputNodesDoNotMakeContinuations
// TODO(seanhaskell): Check names, incl LValues
// TODO(seanhaskell): Check that single bit variable used as a condition gets
// its own continuation without a decl, so that its state element doesn't get
// overwritten
// TODO(seanhaskell): Check that loops preserve values declared before them and
// used after them (state element isn't shared with assignment in the loop)
// TODO(seanhaskell): DISABLED_ParameterNotInContinuations
// TODO(seanhaskell): Remove unnecessary conditions from loop body? Can jump
// past body with loop begin

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

  // TODO(seanhaskell): Can't use ret_value, need to look at return tuple
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

TEST_F(ContinuationsTest, PassthroughsRemoved) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int x = in.read();
      const int y = x * 3;
      out.write(x);
      out.write(x);
      out.write(y);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 5);

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fifth_slice = *slice_it;

  EXPECT_FALSE(SliceOutputsDecl(first_slice, "x"));
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "x"));
  EXPECT_FALSE(SliceOutputsDecl(third_slice, "x"));
  EXPECT_FALSE(SliceOutputsDecl(fourth_slice, "x"));
  EXPECT_FALSE(SliceOutputsDecl(fifth_slice, "x"));
}

TEST_F(ContinuationsTest, PassthroughsRemovedWithInvoke) {
  const std::string content = R"(
    void Run(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      const int x = in.read();
      const int y = x * 3;
      out.write(x);
      out.write(x);
      out.write(y);
    }

    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      Run(in, out);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 5);

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fifth_slice = *slice_it;

  EXPECT_FALSE(SliceOutputsDecl(first_slice, "x"));
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "x"));
  EXPECT_FALSE(SliceOutputsDecl(third_slice, "x"));
  EXPECT_FALSE(SliceOutputsDecl(fourth_slice, "x"));
  EXPECT_FALSE(SliceOutputsDecl(fifth_slice, "x"));
}

TEST_F(ContinuationsTest, PassthroughsRemovedTuple) {
  const std::string content = R"(
    struct Something {
      int x;
      int y;
    };

    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      Something x = {.x = in.read()};
      out.write(x.x);
      out.write(x.x);
      out.write(x.x);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 5);

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fifth_slice = *slice_it;

  EXPECT_FALSE(SliceOutputsDecl(first_slice, "x"));
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "x"));
  EXPECT_FALSE(SliceOutputsDecl(third_slice, "x"));
  EXPECT_FALSE(SliceOutputsDecl(fourth_slice, "x"));
  EXPECT_FALSE(SliceOutputsDecl(fifth_slice, "x"));
}

TEST_F(ContinuationsTest, PassthroughsRemovedScoped) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int x = in.read();
      out.write(x);
      if (x > 10) {
        ++x;
        out.write(x);
      }
      out.write(x * 3);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 5);

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fifth_slice = *slice_it;

  EXPECT_FALSE(SliceOutputsDecl(first_slice, "x"));
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "x"));
  EXPECT_TRUE(SliceInputsDecl(third_slice, "x"));
  EXPECT_FALSE(SliceOutputsDecl(fourth_slice, "x"));
  EXPECT_TRUE(SliceInputsDecl(fourth_slice, "x"));
  EXPECT_FALSE(SliceOutputsDecl(fifth_slice, "x"));
}

TEST_F(ContinuationsTest, SwizzleNotRemoved) {
  const std::string content = R"(
    struct Swizzlable {
      int x;
      int y;
    };

    #pragma hls_top
    void my_package(__xls_channel<Swizzlable>& in,
                    __xls_channel<Swizzlable>& out) {
      const Swizzlable x = in.read();
      out.write(x);
      const Swizzlable y = {.x = x.y, .y = x.x};
      out.write(x);
      out.write(y);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 5);

  auto slice_it = func->slices.begin();
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;
  ++slice_it;

  EXPECT_FALSE(SliceOutputsDecl(second_slice, "y"));
  EXPECT_TRUE(SliceInputsDecl(third_slice, "x"));
  EXPECT_TRUE(SliceInputsDecl(fourth_slice, "y"));
}

TEST_F(ContinuationsTest, UnusedContinuationOutputsRemoved) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int x = in.read();
      const int y = x * 3;
      out.write(y);
      out.write(x);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 4);

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;

  EXPECT_FALSE(SliceOutputsDecl(first_slice, "x"));
  EXPECT_FALSE(SliceOutputsDecl(first_slice, "y"));

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "x"));
  EXPECT_FALSE(SliceOutputsDecl(second_slice, "y"));

  EXPECT_FALSE(SliceOutputsDecl(third_slice, "x"));
  EXPECT_FALSE(SliceOutputsDecl(third_slice, "y"));
}

TEST_F(ContinuationsTest, UnusedContinuationInputsRemoved) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int x = in.read();
      const int y = in.read();
      out.write(y);
      out.write(x);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 5);

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fifth_slice = *slice_it;

  EXPECT_FALSE(SliceInputsDecl(first_slice, "y"));

  EXPECT_FALSE(SliceInputsDecl(second_slice, "y"));

  EXPECT_FALSE(SliceInputsDecl(third_slice, "y"));

  EXPECT_FALSE(SliceInputsDecl(fourth_slice, "y"));

  EXPECT_FALSE(SliceInputsDecl(fifth_slice, "y"));
}

TEST_F(ContinuationsTest, InputsFeedingUnnusedOutputsRemoved) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int x = in.read();
      const int y = in.read();
      out.write(3);
      int z = x * 3;
      int w = y * 5;
      out.write(3);
      (void)z;
      (void)w;
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 5);

  for (const xlscc::GeneratedFunctionSlice& slice : func->slices) {
    EXPECT_FALSE(SliceInputsDecl(slice, "x"));
    EXPECT_FALSE(SliceInputsDecl(slice, "y"));
    EXPECT_FALSE(SliceOutputsDecl(slice, "x"));
    EXPECT_FALSE(SliceOutputsDecl(slice, "y"));
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

  ASSERT_EQ(func->slices.size(), 4);

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
      for (const xlscc::ContinuationInput& continuation_in :
           slice.continuations_in) {
        ASSERT_NE(continuation_in.continuation_out->output_node, nullptr);
        nodes_this_trial.push_back(
            continuation_in.continuation_out->output_node);
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

// This test checks that TrackedBValue sequence numbers don't change when
// stored in a map [with a nondeterministic key].
TEST_F(ContinuationsTest, BValuesDontChangeSequenceNumberInMap) {
  package_ = std::make_unique<xls::Package>("my_package");
  TrackedFunctionBuilder fb("test_func", package_.get());

  const int64_t kNBVals = 100;
  TrackedBValue bval[kNBVals];

  absl::flat_hash_set<int64_t> sequence_numbers;

  for (int64_t i = 0; i < kNBVals; ++i) {
    bval[i] = TrackedBValue(fb.builder()->Literal(xls::SBits(i, 32)));
    ASSERT_FALSE(sequence_numbers.contains(bval[i].sequence_number()));
    sequence_numbers.insert(bval[i].sequence_number());
  }

  absl::flat_hash_map<int64_t, TrackedBValue> test_map;
  for (int64_t i = 0; i < 10; ++i) {
    ASSERT_FALSE(test_map.contains(i));
    test_map[i] = bval[i];

    TrackedBValue& bval_in_map = test_map.at(i);
    int64_t seq = bval_in_map.sequence_number();
    EXPECT_GT(seq, 0);
    ASSERT_FALSE(sequence_numbers.contains(seq));
    sequence_numbers.insert(seq);
  }
  for (int64_t i = 10; i < kNBVals; ++i) {
    const int64_t key = (kNBVals - (i - 10) - 1);
    ASSERT_FALSE(test_map.contains(key));
    test_map[key] = bval[i];

    TrackedBValue& bval_in_map = test_map.at(key);
    int64_t seq = bval_in_map.sequence_number();
    EXPECT_GT(seq, 0);
    ASSERT_FALSE(sequence_numbers.contains(seq));
    sequence_numbers.insert(seq);
  }

  for (const auto& [key, bval] : test_map) {
    EXPECT_TRUE(sequence_numbers.contains(bval.sequence_number()));
    sequence_numbers.erase(bval.sequence_number());
  }
}

TEST_F(ContinuationsTest, BValuesDontChangeSequenceNumberInMap2) {
  package_ = std::make_unique<xls::Package>("my_package");
  TrackedFunctionBuilder fb("test_func", package_.get());

  const int64_t kNBVals = 100;
  TrackedBValue bval[kNBVals];

  absl::flat_hash_set<int64_t> sequence_numbers;

  for (int64_t i = 0; i < kNBVals; ++i) {
    bval[i] = TrackedBValue(fb.builder()->Literal(xls::SBits(i, 32)));
    ASSERT_FALSE(sequence_numbers.contains(bval[i].sequence_number()));
    sequence_numbers.insert(bval[i].sequence_number());
  }

  TrackedBValueMap<int64_t> test_map;
  for (int64_t i = 0; i < 10; ++i) {
    ASSERT_FALSE(test_map.contains(i));
    test_map[i] = bval[i];

    TrackedBValue& bval_in_map = test_map.at(i);
    int64_t seq = bval_in_map.sequence_number();
    EXPECT_GT(seq, 0);
    ASSERT_FALSE(sequence_numbers.contains(seq));
    sequence_numbers.insert(seq);
  }
  for (int64_t i = 10; i < kNBVals; ++i) {
    const int64_t key = (kNBVals - (i - 10) - 1);
    ASSERT_FALSE(test_map.contains(key));
    test_map[key] = bval[i];

    TrackedBValue& bval_in_map = test_map.at(key);
    int64_t seq = bval_in_map.sequence_number();
    EXPECT_GT(seq, 0);
    ASSERT_FALSE(sequence_numbers.contains(seq));
    sequence_numbers.insert(seq);
  }

  for (const auto& [key, bval] : test_map) {
    EXPECT_TRUE(sequence_numbers.contains(bval.sequence_number()));
    sequence_numbers.erase(bval.sequence_number());
  }
}

TEST_F(ContinuationsTest, CopyBValueMapWithSeqNumberDeterminism) {
  package_ = std::make_unique<xls::Package>("my_package");
  TrackedFunctionBuilder fb("test_func", package_.get());

  const int64_t kNBVals = 100;
  const int64_t kNTrails = 10;
  TrackedBValue bval[kNBVals];
  TrackedBValue bvals_by_trial[kNBVals][kNTrails];

  absl::flat_hash_set<int64_t> sequence_numbers;
  TrackedBValueMap<int64_t> ref_map;

  for (int64_t i = 0; i < kNBVals; ++i) {
    bval[i] = TrackedBValue(fb.builder()->Literal(xls::SBits(i, 32)));
    ref_map[i] = bval[i];
  }

  std::vector<TrackedBValue*> ref_ordered = OrderedBValuesForMap(ref_map);

  for (int64_t trial = 0; trial < kNTrails; ++trial) {
    TrackedBValueMap<int64_t> copy_map = ref_map;
    std::vector<TrackedBValue*> copy_ordered = OrderedBValuesForMap(copy_map);
    ASSERT_EQ(copy_ordered.size(), ref_ordered.size());
    for (int64_t i = 0; i < copy_ordered.size(); ++i) {
      EXPECT_EQ(copy_ordered[i]->node()->id(), ref_ordered[i]->node()->id());
    }
  }
}

TEST_F(ContinuationsTest, CopyCValueMapWithSeqNumberDeterminism) {
  package_ = std::make_unique<xls::Package>("my_package");
  TrackedFunctionBuilder fb("test_func", package_.get());

  const int64_t kNBVals = 100;
  const int64_t kNTrails = 10;
  CValue bval[kNBVals];
  CValue bvals_by_trial[kNBVals][kNTrails];

  absl::flat_hash_set<int64_t> sequence_numbers;
  CValueMap<int64_t> ref_map;

  for (int64_t i = 0; i < kNBVals; ++i) {
    bval[i] = CValue(TrackedBValue(fb.builder()->Literal(xls::SBits(i, 32))),
                     std::make_shared<CIntType>(32, /*signed=*/true));
    ref_map[i] = bval[i];
  }

  std::vector<const CValue*> ref_ordered = OrderedCValuesForMap(ref_map);

  for (int64_t trial = 0; trial < kNTrails; ++trial) {
    CValueMap<int64_t> copy_map = ref_map;
    std::vector<const CValue*> copy_ordered = OrderedCValuesForMap(copy_map);
    ASSERT_EQ(copy_ordered.size(), ref_ordered.size());
    for (int64_t i = 0; i < copy_ordered.size(); ++i) {
      EXPECT_EQ(copy_ordered[i]->rvalue().node()->id(),
                ref_ordered[i]->rvalue().node()->id());
    }
  }
}

TEST_F(ContinuationsTest, CopyCValueMapWithSeqNumberDeterminism_WithLValues) {
  package_ = std::make_unique<xls::Package>("my_package");
  TrackedFunctionBuilder fb("test_func", package_.get());

  const int64_t kNBVals = 100;
  const int64_t kNTrails = 10;
  CValue bval[kNBVals];
  CValue bvals_by_trial[kNBVals][kNTrails];

  absl::flat_hash_set<int64_t> sequence_numbers;
  CValueMap<int64_t> ref_map;

  for (int64_t i = 0; i < kNBVals; ++i) {
    if (i % 10 == 0) {
      bval[i] = CValue(TrackedBValue(fb.builder()->Literal(xls::SBits(i, 32))),
                       std::make_shared<CIntType>(32, /*signed=*/true));
    } else {
      std::shared_ptr<LValue> lval;
      if (i % 2 == 0) {
        lval = std::make_shared<LValue>(
            TrackedBValue(fb.builder()->Literal(xls::UBits(i % 2, 1))),
            /*lvalue_true=*/std::make_shared<LValue>(),
            /*lvalue_false=*/std::make_shared<LValue>());
      } else {
        absl::flat_hash_map<int64_t, std::shared_ptr<LValue>> compound_by_index;
        compound_by_index[0] = std::make_shared<LValue>(
            TrackedBValue(fb.builder()->Literal(xls::UBits(i % 5 == 0, 1))),
            /*lvalue_true=*/std::make_shared<LValue>(),
            /*lvalue_false=*/std::make_shared<LValue>());
        compound_by_index[1] = std::make_shared<LValue>(
            TrackedBValue(fb.builder()->Literal(xls::UBits(i % 3 == 0, 1))),
            /*lvalue_true=*/std::make_shared<LValue>(),
            /*lvalue_false=*/std::make_shared<LValue>());
        lval = std::make_shared<LValue>(compound_by_index);
      }
      bval[i] = CValue(TrackedBValue(),
                       std::make_shared<CPointerType>(
                           std::make_shared<CIntType>(32, /*signed=*/true)),
                       /*disable_type_check=*/false, lval);
      ref_map[i] = bval[i];
    }
  }

  std::vector<const CValue*> ref_ordered = OrderedCValuesForMap(ref_map);

  for (int64_t trial = 0; trial < kNTrails; ++trial) {
    CValueMap<int64_t> copy_map = ref_map;
    std::vector<const CValue*> copy_ordered = OrderedCValuesForMap(copy_map);
    ASSERT_EQ(copy_ordered.size(), ref_ordered.size());
    for (int64_t i = 0; i < copy_ordered.size(); ++i) {
      EXPECT_EQ(copy_ordered[i]->lvalue().get(),
                ref_ordered[i]->lvalue().get());
    }
  }
}

TEST_F(ContinuationsTest, MergeContinuationValuesWithSameLifetimes) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int r = in.read();
      int y = r;
      out.write(1);
      out.write(3);
      out.write(r + y);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 5);

  auto slice_it = func->slices.begin();
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "r"));
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "y"));
  EXPECT_EQ(second_slice.continuations_out.size(), 1);

  EXPECT_FALSE(SliceOutputsDecl(third_slice, "r"));
  EXPECT_FALSE(SliceOutputsDecl(third_slice, "y"));

  EXPECT_TRUE(SliceInputsDecl(fourth_slice, "r"));
  EXPECT_TRUE(SliceInputsDecl(fourth_slice, "y"));
  EXPECT_EQ(fourth_slice.continuations_in.size(), 1);
}

TEST_F(ContinuationsTest, MergeContinuationValuesWithSameLifetimes2) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int ctrl = in.read();
      int ret = 5;
      #pragma hls_unroll yes
      for(int i=0;i<3;++i) {
        if(ctrl == 1) {
          ret += 2*in.read() + ctrl;
        }
      }
      out.write(ret);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 6);

  auto slice_it = func->slices.begin();
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fifth_slice = *slice_it;

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "ctrl"));
  // Also condition output
  EXPECT_EQ(second_slice.continuations_out.size(), 2);

  EXPECT_TRUE(SliceInputsDecl(third_slice, "ctrl"));
  // Also condition input
  EXPECT_EQ(third_slice.continuations_in.size(), 2);
  EXPECT_TRUE(SliceOutputsDecl(third_slice, "ret"));
  // Also condition output
  EXPECT_EQ(third_slice.continuations_out.size(), 2);

  EXPECT_TRUE(SliceInputsDecl(fourth_slice, "ctrl"));
  EXPECT_TRUE(SliceInputsDecl(fourth_slice, "ret"));
  // Also condition input
  EXPECT_EQ(fourth_slice.continuations_in.size(), 3);
  EXPECT_TRUE(SliceOutputsDecl(fourth_slice, "ret"));
  // Also condition output
  EXPECT_EQ(fourth_slice.continuations_out.size(), 2);

  EXPECT_TRUE(SliceInputsDecl(fifth_slice, "ctrl"));
  EXPECT_TRUE(SliceInputsDecl(fifth_slice, "ret"));
  // Also condition input
  EXPECT_EQ(fifth_slice.continuations_in.size(), 3);
}

TEST_F(ContinuationsTest, DISABLED_ParameterNotInContinuations) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(int&dir,
                    __xls_channel<int>& in,
                    __xls_channel<int>& out1,
                    __xls_channel<int>& out2) {
      const int x = in.read();
      if(dir) {
        out1.write(x);
      } else {
        out2.write(x);
      }
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 4);

  for (const xlscc::GeneratedFunctionSlice& slice : func->slices) {
    EXPECT_FALSE(SliceInputsDecl(slice, "dir"));
    EXPECT_FALSE(SliceOutputsDecl(slice, "dir"));
  }
}

TEST_F(ContinuationsTest, LiteralPropagation) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int y = 1;
      int r = in.read();
      out.write(y);
      int z = y + 3;
      out.write(z);
      int w = z * 5;
      out.write(w);
      out.write(r + y + z + w);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 6);

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fifth_slice = *slice_it;

  EXPECT_FALSE(SliceOutputsDecl(first_slice, "y"));

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "r"));

  EXPECT_FALSE(SliceOutputsDecl(third_slice, "z"));

  EXPECT_FALSE(SliceOutputsDecl(fourth_slice, "w"));

  EXPECT_TRUE(SliceInputsDecl(fifth_slice, "r"));
  EXPECT_FALSE(SliceOutputsDecl(fifth_slice, "y"));
  EXPECT_FALSE(SliceOutputsDecl(fifth_slice, "z"));
  EXPECT_FALSE(SliceOutputsDecl(fifth_slice, "w"));
}

TEST_F(ContinuationsTest, PipelinedLoopBackwardsPropagation) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int ctrl = in.read();
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for(int i=1;i <= ctrl;++i) {
        a += i;
      }
      out.write(a);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 5);

  auto slice_it = func->slices.begin();
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "ctrl"));
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "a"));
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "i"));

  EXPECT_EQ(SliceInputsDeclCount(third_slice, "i"), 2);
  EXPECT_EQ(SliceInputsDeclCount(third_slice, "a"), 2);
  EXPECT_TRUE(SliceInputsDecl(third_slice, "a", /*direct_in=*/false,
                              /*is_feedback=*/false, /*func=*/func));
  EXPECT_TRUE(SliceInputsDecl(third_slice, "i", /*direct_in=*/false,
                              /*is_feedback=*/false, /*func=*/func));
  EXPECT_TRUE(SliceInputsDecl(third_slice, "a", /*direct_in=*/false,
                              /*is_feedback=*/true, /*func=*/func));
  EXPECT_TRUE(SliceInputsDecl(third_slice, "i", /*direct_in=*/false,
                              /*is_feedback=*/true, /*func=*/func));

  EXPECT_TRUE(SliceInputsDecl(fourth_slice, "a"));
  EXPECT_FALSE(SliceInputsDecl(fourth_slice, "i"));
}

TEST_F(ContinuationsTest, PipelinedLoopBackwardsPropagation2) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int a = in.read();
      #pragma hls_pipeline_init_interval 1
      for(int i=1;i<=4;++i) {
        a += i * in.read();
      }
      out.write(a);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 6);

  auto slice_it = func->slices.begin();
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fifth_slice = *slice_it;

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "a"));
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "i"));

  EXPECT_FALSE(SliceInputsDecl(third_slice, "i"));
  EXPECT_FALSE(SliceInputsDecl(third_slice, "a"));
  EXPECT_FALSE(SliceOutputsDecl(third_slice, "i"));
  EXPECT_FALSE(SliceOutputsDecl(third_slice, "a"));

  EXPECT_EQ(SliceInputsDeclCount(fourth_slice, "a"), 2);
  EXPECT_EQ(SliceInputsDeclCount(fourth_slice, "i"), 2);

  EXPECT_TRUE(SliceInputsDecl(fifth_slice, "a"));
  EXPECT_FALSE(SliceInputsDecl(fifth_slice, "i"));
}

TEST_F(ContinuationsTest, PipelinedLoopBackwardsPropagation3) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int a = in.read();
      #pragma hls_pipeline_init_interval 1
      for(int i=1;i<=4;++i) {
        if(a < 30) {
          a += i * in.read();
        }
      }
      out.write(a);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 6);

  auto slice_it = func->slices.begin();
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fifth_slice = *slice_it;

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "a"));
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "i"));

  EXPECT_EQ(SliceInputsDeclCount(third_slice, "a"), 2);
  EXPECT_FALSE(SliceOutputsDecl(third_slice, "i"));
  EXPECT_FALSE(SliceOutputsDecl(third_slice, "a"));

  EXPECT_EQ(SliceInputsDeclCount(fourth_slice, "a"), 2);
  EXPECT_EQ(SliceInputsDeclCount(fourth_slice, "i"), 2);

  EXPECT_TRUE(SliceInputsDecl(fifth_slice, "a"));
  EXPECT_FALSE(SliceInputsDecl(fifth_slice, "i"));
}

TEST_F(ContinuationsTest, PipelinedLoopBackwardsPropagation4) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int ctrl = in.read();
      int a = ctrl;
      #pragma hls_pipeline_init_interval 1
      for(int i=1;i<=4;++i) {
        if(ctrl == 1) {
          a += i * in.read();
        }
      }
      out.write(a);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 6);

  auto slice_it = func->slices.begin();
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fifth_slice = *slice_it;

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "a"));
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "i"));
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "ctrl"));

  EXPECT_TRUE(SliceInputsDecl(third_slice, "ctrl"));
  EXPECT_FALSE(SliceOutputsDecl(third_slice, "i"));
  EXPECT_FALSE(SliceOutputsDecl(third_slice, "a"));

  EXPECT_EQ(SliceInputsDeclCount(fourth_slice, "a"), 2);
  EXPECT_EQ(SliceInputsDeclCount(fourth_slice, "i"), 2);

  EXPECT_TRUE(SliceInputsDecl(fifth_slice, "a"));
  EXPECT_FALSE(SliceInputsDecl(fifth_slice, "i"));
}

TEST_F(ContinuationsTest, PipelinedLoopConstantPropagation) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int c = 5;
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for(int i=1;i<=4;++i) {
        a += c * in.read();
      }
      out.write(a);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 5);

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;

  EXPECT_TRUE(SliceOutputsDecl(first_slice, "a"));
  EXPECT_TRUE(SliceOutputsDecl(first_slice, "i"));
  EXPECT_FALSE(SliceOutputsDecl(first_slice, "c"));

  EXPECT_FALSE(SliceInputsDecl(second_slice, "i"));
  EXPECT_FALSE(SliceInputsDecl(second_slice, "a"));

  EXPECT_EQ(SliceInputsDeclCount(third_slice, "i"), 2);
  EXPECT_EQ(SliceInputsDeclCount(third_slice, "a"), 2);

  EXPECT_TRUE(SliceInputsDecl(fourth_slice, "a"));
  EXPECT_FALSE(SliceInputsDecl(fourth_slice, "i"));
}

TEST_F(ContinuationsTest, PipelinedLoopSameNodeOneBypass) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int r = in.read();
      int a = r;
      #pragma hls_pipeline_init_interval 1
      for(int i=1;i<=4;++i) {
        a += r * in.read();
      }
      out.write(r + a);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 6);

  auto slice_it = func->slices.begin();
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fifth_slice = *slice_it;

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "i"));
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "a"));
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "r"));

  EXPECT_FALSE(SliceInputsDecl(third_slice, "a"));
  EXPECT_FALSE(SliceInputsDecl(third_slice, "i"));

  EXPECT_TRUE(SliceInputsDecl(fourth_slice, "r"));
  EXPECT_TRUE(SliceInputDoesNotInputBothDecls(fourth_slice, "a", "r"));
  EXPECT_EQ(SliceInputsDeclCount(fourth_slice, "i"), 2);
  EXPECT_EQ(SliceInputsDeclCount(fourth_slice, "a"), 3);
  EXPECT_TRUE(SliceOutputsDecl(fourth_slice, "a"));
  EXPECT_TRUE(SliceOutputsDecl(fourth_slice, "i"));

  EXPECT_TRUE(SliceInputsDecl(fifth_slice, "a"));
  EXPECT_TRUE(SliceInputsDecl(fifth_slice, "r"));
  EXPECT_TRUE(SliceInputDoesNotInputBothDecls(fifth_slice, "a", "r"));
}

TEST_F(ContinuationsTest, PipelinedLoopNothingOutside) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int c = 5;
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for(int i=1;i<=4;++i) {
        a += c * in.read();
        out.write(a);
      }
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 5);

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;

  EXPECT_TRUE(SliceOutputsDecl(first_slice, "a", /*direct_in*/ false));
  EXPECT_TRUE(SliceOutputsDecl(first_slice, "i", /*direct_in*/ false));
  EXPECT_TRUE(SliceOutputsDecl(first_slice, "c", /*direct_in*/ false));

  EXPECT_FALSE(SliceInputsDecl(second_slice, "i"));
  EXPECT_FALSE(SliceInputsDecl(second_slice, "a"));

  EXPECT_FALSE(SliceInputsDecl(third_slice, "i"));
  EXPECT_EQ(SliceInputsDeclCount(third_slice, "a"), 2);
  EXPECT_FALSE(SliceOutputsDecl(third_slice, "i"));
  EXPECT_TRUE(SliceOutputsDecl(third_slice, "a"));
}

TEST_F(ContinuationsTest, PipelinedLoopInIf) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int c = 5;
      int a = 0;
      if (in.read() == 3) {
        #pragma hls_pipeline_init_interval 1
        for(int i=1;i<=4;++i) {
          a += c * in.read();
          out.write(a);
        }
      }
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 6);

  auto slice_it = func->slices.begin();
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "a"));
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "i"));

  EXPECT_FALSE(SliceInputsDecl(third_slice, "a"));
  EXPECT_FALSE(SliceInputsDecl(third_slice, "i"));
  EXPECT_FALSE(SliceOutputsDecl(third_slice, "a"));
  EXPECT_FALSE(SliceOutputsDecl(third_slice, "i"));

  EXPECT_FALSE(SliceInputsDecl(fourth_slice, "i"));
  EXPECT_EQ(SliceInputsDeclCount(fourth_slice, "a"), 2);
  EXPECT_TRUE(SliceOutputsDecl(fourth_slice, "a"));
  EXPECT_FALSE(SliceOutputsDecl(fourth_slice, "i"));
}

TEST_F(ContinuationsTest, DuplicateName) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int acc = 0;
      {
        static int x = in.read();
        acc += x;
      }
      {
        static int x = in.read();
        acc += 3*x;
      }
      out.write(acc);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 4);

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;

  EXPECT_TRUE(SliceOutputsDecl(first_slice, "x"));

  EXPECT_TRUE(SliceInputsDecl(second_slice, "x"));
  EXPECT_EQ(SliceOutputsDeclCount(second_slice, "x"), 2);
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "acc"));

  EXPECT_TRUE(SliceOutputsDecl(third_slice, "x"));
  EXPECT_TRUE(SliceInputsDecl(third_slice, "x"));
  EXPECT_TRUE(SliceInputsDecl(third_slice, "acc"));
}

TEST_F(ContinuationsTest, SingleOutputLastSlice) {
  const std::string content = R"(
    int single_output(int x) {
      return x + 1;
    }

    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int x = in.read();
      int y = single_output(x);
      out.write(y);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 3);

  // Just checking that optimization doesn't crash
}

TEST_F(ContinuationsTest, PipelinedLoopNested) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int ctrl = in.read();
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for(int i=1;i <= 16;++i) {
        #pragma hls_pipeline_init_interval 1
        for(int j=1;j <= 16;++j) {
          a += ctrl;
          if (j == ctrl) {
            break;
          }
        }
        if (i == ctrl) {
          break;
        }
      }
      out.write(a);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 7);

  auto slice_it = func->slices.begin();
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fifth_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& sixth_slice = *slice_it;

  EXPECT_EQ(SliceOutputsDeclCount(second_slice, "ctrl"), 1);
  EXPECT_EQ(SliceOutputsDeclCount(second_slice, "a"), 1);
  EXPECT_EQ(SliceOutputsDeclCount(second_slice, "i"), 1);

  EXPECT_EQ(SliceOutputsDeclCount(third_slice, "j"), 1);
  EXPECT_FALSE(SliceOutputsDecl(third_slice, "i"));

  EXPECT_EQ(SliceInputsDeclCount(fourth_slice, "ctrl"), 1);
  EXPECT_TRUE(SliceInputsDecl(fourth_slice, "ctrl", /*direct_in=*/false,
                              /*is_feedback=*/false, /*func=*/func));
  EXPECT_EQ(SliceInputsDeclCount(fourth_slice, "a"), 3);
  EXPECT_EQ(SliceInputsDeclCount(fourth_slice, "j"), 2);
  EXPECT_EQ(SliceOutputsDeclCount(fourth_slice, "a"), 1);
  EXPECT_EQ(SliceOutputsDeclCount(fourth_slice, "j"), 1);

  EXPECT_EQ(SliceInputsDeclCount(fifth_slice, "ctrl"), 1);
  EXPECT_TRUE(SliceInputsDecl(fifth_slice, "ctrl", /*direct_in=*/false,
                              /*is_feedback=*/false, /*func=*/func));
  EXPECT_EQ(SliceInputsDeclCount(fifth_slice, "i"), 1);
  EXPECT_EQ(SliceOutputsDeclCount(fifth_slice, "a"), 1);
  EXPECT_EQ(SliceInputsDeclCount(fifth_slice, "a"), 1);
  EXPECT_EQ(SliceOutputsDeclCount(fifth_slice, "i"), 1);
  EXPECT_FALSE(SliceOutputsDecl(fifth_slice, "j"));

  EXPECT_EQ(SliceInputsDeclCount(sixth_slice, "a"), 1);
}

TEST_F(ContinuationsTest, EmptyTupleRemoved) {
  const std::string content = R"(
    struct Empty {
      __xls_channel<int>& out;
    };

    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      static Empty empty = {.out = out};

      const int x = in.read();
      const int y = x * 3;
      out.write(x);
      out.write(x);
      empty.out.write(y);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  for (const GeneratedFunctionSlice& slice : func->slices) {
    for (const ContinuationInput& continuation_in : slice.continuations_in) {
      EXPECT_GT(continuation_in.input_node->GetType()->GetFlatBitCount(), 0);
    }
    for (const ContinuationValue& continuation_out : slice.continuations_out) {
      EXPECT_GT(continuation_out.output_node->GetType()->GetFlatBitCount(), 0);
    }
  }
}

TEST_F(ContinuationsTest, EmptyTupleRemovedWithLoop) {
  const std::string content = R"(
    struct Empty {
      __xls_channel<int>& out;
    };

    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      static Empty empty = {.out = out};

      const int x = in.read();
      const int y = x * 3;
      [[hls_pipeline_init_interval(1)]]
      for (int i = 0; i < 3; ++i) {
        out.write(x);
      }
      out.write(x);
      empty.out.write(y);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  for (const GeneratedFunctionSlice& slice : func->slices) {
    for (const ContinuationInput& continuation_in : slice.continuations_in) {
      EXPECT_GT(continuation_in.input_node->GetType()->GetFlatBitCount(), 0);
    }
    for (const ContinuationValue& continuation_out : slice.continuations_out) {
      EXPECT_GT(continuation_out.output_node->GetType()->GetFlatBitCount(), 0);
    }
  }
}

TEST_F(ContinuationsTest, DirectInMarked) {
  const std::string content = R"(
    struct DirectIn {
      int x;
      int y;
    };

    #pragma hls_top
    void my_package(const DirectIn&direct_in,
                    __xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int x = in.read();
      const int dval = direct_in.x;
      out.write(x);
      out.write(x * dval);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 4);

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;

  EXPECT_TRUE(SliceOutputsDecl(first_slice, "direct_in", /*direct_in=*/true));

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "x", /*direct_in=*/false));

  EXPECT_TRUE(SliceInputsDecl(third_slice, "dval", /*direct_in=*/true));
}

TEST_F(ContinuationsTest, DirectInNotDecomposed) {
  const std::string content = R"(
    struct DirectIn {
      int x;
      int y;
    };

    #pragma hls_top
    void my_package(const DirectIn&direct_in,
                    __xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int x = in.read();
      out.write(x);
      out.write(x * direct_in.x);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 4);

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;

  EXPECT_TRUE(SliceOutputsDecl(first_slice, "direct_in", /*direct_in=*/true));

  EXPECT_TRUE(SliceInputsDecl(third_slice, "direct_in", /*direct_in=*/true));

  for (const ContinuationInput& continuation_in :
       third_slice.continuations_in) {
    if (!continuation_in.continuation_out->direct_in) {
      continue;
    }
    EXPECT_TRUE(continuation_in.input_node->GetType()->IsTuple());
  }
}

TEST_F(ContinuationsTest, DirectInShouldNotFeedback) {
  const std::string content = R"(
    struct DirectIn {
      int x;
      int y;
    };

    #pragma hls_top
    void my_package(DirectIn&direct_in,
                    __xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int v = 0;
      [[hls_pipeline_init_interval(1)]]
      for (int i=0;i<4;++i) {
        if (v & 1) {
          [[hls_pipeline_init_interval(1)]]
          for (int j=0;j<4;++j) {
            const int x = in.read();
            out.write(v + direct_in.x);
            v += x;
          }
        }
      }
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_GE(func->slices.size(), 1);

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;

  EXPECT_TRUE(SliceOutputsDecl(first_slice, "direct_in", /*direct_in=*/true));

  for (const GeneratedFunctionSlice& slice : func->slices) {
    if (&slice == &first_slice) {
      continue;
    }
    EXPECT_FALSE(SliceOutputsDecl(slice, "direct_in"));
  }
}

TEST_F(ContinuationsTest, UnusedDirectInShouldNotFeedback) {
  const std::string content = R"(
    struct DirectIn {
      int x;
      int y;
    };

    #pragma hls_top
    void my_package(DirectIn&direct_in,
                    __xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int v = 0;
      [[hls_pipeline_init_interval(1)]]
      for (int i=0;i<4;++i) {
        if (v & 1) {
          [[hls_pipeline_init_interval(1)]]
          for (int j=0;j<4;++j) {
            const int x = in.read();
            v += x;
          }
        }
      }
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_GE(func->slices.size(), 1);

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;

  EXPECT_TRUE(SliceOutputsDecl(first_slice, "direct_in", /*direct_in=*/true));

  for (const GeneratedFunctionSlice& slice : func->slices) {
    if (&slice == &first_slice) {
      continue;
    }
    EXPECT_FALSE(SliceOutputsDecl(slice, "direct_in"));
  }
}

TEST_F(ContinuationsTest, DirectInArithmeticWithLiteralMarked) {
  const std::string content = R"(
    struct DirectIn {
      int x;
      int y;
    };

    #pragma hls_top
    void my_package(const DirectIn&direct_in,
                    __xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int x = in.read();
      const int dval = direct_in.x + 10;
      out.write(x);
      out.write(x * dval);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;

  ASSERT_EQ(func->slices.size(), 4);

  EXPECT_TRUE(SliceOutputsDecl(first_slice, "direct_in", /*direct_in=*/true));

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "x", /*direct_in=*/false));
  EXPECT_TRUE(SliceInputsDecl(second_slice, "direct_in", /*direct_in=*/true));
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "dval", /*direct_in=*/true));

  EXPECT_TRUE(SliceInputsDecl(third_slice, "dval", /*direct_in=*/true));
}

TEST_F(ContinuationsTest, DirectInMarkedInSubroutine) {
  const std::string content = R"(
    struct DirectIn {
      int x;
      int y;
    };

    void Inner(int a,
               const DirectIn&direct_in,
                __xls_channel<int>& in,
                __xls_channel<int>& out) {
      const int x = in.read();
      const int dval = direct_in.x;
      out.write(x + a);
      out.write(x * dval);
    }

    #pragma hls_top
    void my_package(const DirectIn&direct_in,
                    __xls_channel<int>& in,
                    __xls_channel<int>& out) {
      Inner(3, direct_in, in, out);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;

  ASSERT_EQ(func->slices.size(), 4);

  EXPECT_TRUE(SliceOutputsDecl(first_slice, "direct_in", /*direct_in=*/true));

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "x", /*direct_in=*/false));

  EXPECT_TRUE(SliceInputsDecl(third_slice, "dval", /*direct_in=*/true));
}

TEST_F(ContinuationsTest, DirectInMarkedInSubroutineMultiCall) {
  const std::string content = R"(
    struct DirectIn {
      int x;
      int y;
    };

    void Inner(int a,
               const DirectIn&direct_in,
                __xls_channel<int>& in,
                __xls_channel<int>& out) {
      const int x = in.read();
      const int dval = direct_in.x;
      out.write(x + a);
      out.write(x * dval);
    }

    #pragma hls_top
    void my_package(const DirectIn&direct_in,
                    __xls_channel<int>& in,
                    __xls_channel<int>& out) {
      Inner(3, direct_in, in, out);
      DirectIn not_direct_in = {
        .x = in.read(),
        .y = in.read()
      };
      Inner(5, not_direct_in, in, out);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  auto slice_it = func->slices.begin();
  ASSERT_EQ(func->slices.size(), 9);

  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  ++slice_it;
  ++slice_it;
  ++slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& eighth_slice = *slice_it;

  EXPECT_TRUE(SliceOutputsDecl(first_slice, "direct_in", /*direct_in=*/true));

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "x", /*direct_in=*/false));

  EXPECT_TRUE(SliceInputsDecl(third_slice, "dval", /*direct_in=*/true));

  EXPECT_TRUE(SliceInputsDecl(eighth_slice, "dval", /*direct_in=*/false));
}

TEST_F(ContinuationsTest, DirectInMarkedInSubroutineNonConst) {
  const std::string content = R"(
    struct DirectIn {
      int x;
      int y;
    };

    void Inner(int a,
               DirectIn&direct_in,
                __xls_channel<int>& in,
                __xls_channel<int>& out) {
      const int x = in.read();
      const int dval = direct_in.x;
      out.write(x + a);
      out.write(x * dval);
    }

    #pragma hls_top
    void my_package(DirectIn&direct_in,
                    __xls_channel<int>& in,
                    __xls_channel<int>& out) {
      Inner(3, direct_in, in, out);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;

  ASSERT_EQ(func->slices.size(), 4);

  EXPECT_TRUE(SliceOutputsDecl(first_slice, "direct_in", /*direct_in=*/true));

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "x", /*direct_in=*/false));

  EXPECT_TRUE(SliceInputsDecl(third_slice, "dval", /*direct_in=*/true));
}

TEST_F(ContinuationsTest, DirectInMarkedInSubroutineNested) {
  const std::string content = R"(
    struct DirectIn {
      int x;
      int y;
    };

    void Inner2(const DirectIn&direct_in,
                __xls_channel<int>& in,
                __xls_channel<int>& out) {
      const int x = in.read();
      const int dval = direct_in.x;
      out.write(x);
      out.write(x * dval);
    }

    void Inner1(const DirectIn&direct_in,
                __xls_channel<int>& in,
                __xls_channel<int>& out) {
      Inner2(direct_in, in, out);
    }

    #pragma hls_top
    void my_package(const DirectIn&direct_in,
                    __xls_channel<int>& in,
                    __xls_channel<int>& out) {
      Inner1(direct_in, in, out);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;

  ASSERT_EQ(func->slices.size(), 4);

  EXPECT_TRUE(SliceOutputsDecl(first_slice, "direct_in", /*direct_in=*/true));

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "x", /*direct_in=*/false));

  EXPECT_TRUE(SliceInputsDecl(third_slice, "dval", /*direct_in=*/true));
}

TEST_F(ContinuationsTest, DirectInMarkedInSubroutineNested2) {
  const std::string content = R"(
    struct DirectIn {
      int x;
      int y;
    };

    void Inner3(const DirectIn&direct_in,
                __xls_channel<int>& in,
                __xls_channel<int>& out) {
      const int x = in.read();
      const int dval = direct_in.x;
      out.write(x);
      out.write(x * dval);
    }

    void Inner2(const DirectIn&direct_in,
                __xls_channel<int>& in,
                __xls_channel<int>& out) {
      Inner3(direct_in, in, out);
    }

    void Inner1(const DirectIn&direct_in,
                __xls_channel<int>& in,
                __xls_channel<int>& out) {
      Inner2(direct_in, in, out);
    }

    #pragma hls_top
    void my_package(const DirectIn&direct_in,
                    __xls_channel<int>& in,
                    __xls_channel<int>& out) {
      Inner1(direct_in, in, out);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;

  ASSERT_EQ(func->slices.size(), 4);

  EXPECT_TRUE(SliceOutputsDecl(first_slice, "direct_in", /*direct_in=*/true));

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "x", /*direct_in=*/false));

  EXPECT_TRUE(SliceInputsDecl(third_slice, "dval", /*direct_in=*/true));
}

TEST_F(ContinuationsTest, DirectInSelectNotMarked) {
  const std::string content = R"(
    struct DirectIn {
      int x;
      int y;
    };

    #pragma hls_top
    void my_package(const DirectIn&direct_in,
                    __xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int x = in.read();
      const int dval = x == 5 ? direct_in.x : direct_in.y;
      out.write(x);
      out.write(x * dval);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 4);

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;

  EXPECT_TRUE(SliceOutputsDecl(first_slice, "direct_in", /*direct_in=*/true));

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "x", /*direct_in=*/false));

  EXPECT_TRUE(SliceInputsDecl(third_slice, "dval", /*direct_in=*/false));
}

TEST_F(ContinuationsTest, SplitOnChannelOps) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int x = in.read();
      const int y = in.read();
      const int z = x + y;
      out.write(y);
      out.write(z);
    })";

  split_states_on_channel_ops_ = true;
  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 9);

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fifth_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& sixth_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& seventh_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& eighth_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& ninth_slice = *slice_it;

  EXPECT_EQ(first_slice.after_op, nullptr);
  EXPECT_EQ(second_slice.after_op, nullptr);
  EXPECT_NE(third_slice.after_op, nullptr);
  EXPECT_EQ(fourth_slice.after_op, nullptr);
  EXPECT_NE(fifth_slice.after_op, nullptr);
  EXPECT_EQ(sixth_slice.after_op, nullptr);
  EXPECT_NE(seventh_slice.after_op, nullptr);
  EXPECT_EQ(eighth_slice.after_op, nullptr);
  EXPECT_NE(ninth_slice.after_op, nullptr);
}

TEST_F(ContinuationsTest, PipelinedLoopBackwardsPropagationInSubroutine) {
  const std::string content = R"(
    int accum(int ctrl) {
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for(int i=1;i <= ctrl;++i) {
        a += i;
      }
      return a;
    }

    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int ctrl = in.read();
      out.write(accum(ctrl));
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 5);

  auto slice_it = func->slices.begin();
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "ctrl"));
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "a", /*direct_in=*/false));
  EXPECT_TRUE(SliceOutputsDecl(second_slice, "i", /*direct_in=*/false));

  EXPECT_EQ(SliceInputsDeclCount(third_slice, "i"), 2);
  EXPECT_EQ(SliceInputsDeclCount(third_slice, "a"), 2);
  EXPECT_TRUE(SliceInputsDecl(third_slice, "a"));
  EXPECT_TRUE(SliceInputsDecl(third_slice, "i"));
  EXPECT_TRUE(SliceOutputsDecl(third_slice, "a", /*direct_in=*/false));
  EXPECT_TRUE(SliceOutputsDecl(third_slice, "i", /*direct_in=*/false));

  EXPECT_TRUE(SliceInputsDecl(fourth_slice, "a", /*direct_in=*/false));
  EXPECT_FALSE(SliceInputsDecl(fourth_slice, "i", /*direct_in=*/false));
}

TEST_F(ContinuationsTest, PassthroughSliceNotRemoved) {
  const std::string content = R"(
       struct Big {
         int v[7];
       };
       struct Small {
         int v[3];
         bool flag;
       };

       class Block {
        public:
         __xls_channel<Big, __xls_channel_dir_In>& in;
         __xls_channel<Big, __xls_channel_dir_In>& in2;
         __xls_channel<int, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
           Big big = in.read();
           Small small;
           (void)in2.read();
           small.v[0] = big.v[0];
           small.v[1] = big.v[1];
           small.v[2] = big.v[2];
           [[hls_pipeline_init_interval(1)]]
           for (int i = 0; i < 24; ++i) {
            small.flag = false;
           }
        }
      };)";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));
  (void)func;
  // (Just check that it doesn't crash)
}

TEST_F(ContinuationsTest, StructSingleElementContinued) {
  const std::string content = R"(
    struct Test {
      int x = 0;
      int y = 0;
    };

    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      Test test;
      test.x = in.read();
      test.y = test.x * 3;
      out.write(test.y);
      out.write(test.x);
      out.write(test.x);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 5);

  auto slice_it = func->slices.begin();
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;

  EXPECT_EQ(SliceOutputsDeclCount(second_slice, "test"), 1);
}

TEST_F(ContinuationsTest, PipelinedLoopSimpleFeedback) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int, __xls_channel_dir_In>& in;
         __xls_channel<int, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          int sum = 0;

          #pragma hls_pipeline_init_interval 1
          while(true) {
            sum += in.read();
            out.write(sum);
          }
        }
      };)";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));
  ASSERT_EQ(func->slices.size(), 5);

  auto slice_it = func->slices.begin();
  const xlscc::GeneratedFunctionSlice& first_slice = *slice_it;
  ++slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;
  ++slice_it;

  EXPECT_TRUE(SliceOutputsDecl(first_slice, "sum"));
  EXPECT_EQ(SliceInputsDeclCount(third_slice, "sum"), 2);
  EXPECT_TRUE(SliceInputsDecl(third_slice, "sum", /*direct_in=*/false,
                              /*is_feedback=*/true, /*func=*/func));
  EXPECT_TRUE(SliceOutputsDecl(third_slice, "sum"));
  EXPECT_EQ(SliceInputsDeclCount(fourth_slice, "sum"), 1);
  EXPECT_TRUE(SliceOutputsDecl(fourth_slice, "sum"));
}

TEST_F(ContinuationsTest, PipelinedLoopSimpleNoFeedback) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int, __xls_channel_dir_In>& in;
         __xls_channel<int, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          const int v = in.read();

          #pragma hls_pipeline_init_interval 1
          while(true) {
            out.write(v);
          }
        }
      };)";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));
  ASSERT_EQ(func->slices.size(), 5);

  auto slice_it = func->slices.begin();
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;
  ++slice_it;

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "v"));
  EXPECT_TRUE(SliceInputsDecl(third_slice, "v", /*direct_in=*/false,
                              /*is_feedback=*/false, /*func=*/func));
}

TEST_F(ContinuationsTest, PassthroughFeedbackOrder) {
  const std::string content = R"(
       class Block {
        public:
         __xls_channel<int, __xls_channel_dir_In>& in;
         __xls_channel<int, __xls_channel_dir_Out>& out;

         #pragma hls_top
         void Run() {
          int value = in.read();
          int set_it = 10;

          #pragma hls_pipeline_init_interval 1
          for(int i=0;i<6;++i) {
            #pragma hls_pipeline_init_interval 1
            for(int j=0;j<6;++j) {
              out.write(value);
            }
            value = set_it;
          }
        }
      };)";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));
  ASSERT_GE(func->slices.size(), 4);

  auto slice_it = func->slices.begin();
  ++slice_it;
  ++slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& fourth_slice = *slice_it;

  ASSERT_EQ(fourth_slice.continuations_in.size(), 3);

  absl::flat_hash_map<const ContinuationValue*, int64_t>
      slice_indices_by_value = GetSliceIndicesByValue(*func);

  std::list<ContinuationInput> inputs_sorted_by_slice_index =
      fourth_slice.continuations_in;

  inputs_sorted_by_slice_index.sort(
      [&slice_indices_by_value](const ContinuationInput& a,
                                const ContinuationInput& b) -> bool {
        return slice_indices_by_value.at(a.continuation_out) <
               slice_indices_by_value.at(b.continuation_out);
      });

  const absl::flat_hash_set<DeclLeaf>& first_decls =
      inputs_sorted_by_slice_index.front().continuation_out->decls;
  EXPECT_FALSE(std::any_of(first_decls.begin(), first_decls.end(),
                           [](const DeclLeaf& decl) {
                             return decl.decl->getNameAsString() == "set_it";
                           }));

  const absl::flat_hash_set<DeclLeaf>& last_decls =
      inputs_sorted_by_slice_index.back().continuation_out->decls;
  EXPECT_TRUE(std::any_of(last_decls.begin(), last_decls.end(),
                          [](const DeclLeaf& decl) {
                            return decl.decl->getNameAsString() == "set_it";
                          }));
}

TEST_F(ContinuationsTest, ContinuationDecomposed) {
  const std::string content = R"(
    struct Thing {
      int x;
      long y;
    };

    #pragma hls_top
    void my_package(__xls_channel<Thing>& in,
                    __xls_channel<Thing>& out) {
      const Thing x = in.read();
      out.write(x);
      out.write(x);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 4);

  auto slice_it = func->slices.begin();
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& third_slice = *slice_it;
  ++slice_it;

  EXPECT_TRUE(SliceOutputsDecl(second_slice, "x"));
  EXPECT_TRUE(SliceInputsDecl(third_slice, "x", /*direct_in=*/std::nullopt,
                              /*is_feedback=*/std::nullopt,
                              /*func=*/std::nullopt, /*decl_index=*/0));
  EXPECT_TRUE(SliceInputsDecl(third_slice, "x", /*direct_in=*/std::nullopt,
                              /*is_feedback=*/std::nullopt,
                              /*func=*/std::nullopt, /*decl_index=*/1));
  EXPECT_FALSE(SliceInputsDecl(third_slice, "x", /*direct_in=*/std::nullopt,
                               /*is_feedback=*/std::nullopt,
                               /*func=*/std::nullopt, /*decl_index=*/-1));
  EXPECT_FALSE(SliceInputsDecl(third_slice, "x", /*direct_in=*/std::nullopt,
                               /*is_feedback=*/std::nullopt,
                               /*func=*/std::nullopt, /*decl_index=*/2));
}

TEST_F(ContinuationsTest, ContinuationLiteralDecomposed) {
  const std::string content = R"(
    struct Thing {
      int x;
      long y;
    };

    #pragma hls_top
    void my_package(__xls_channel<Thing>& in,
                    __xls_channel<Thing>& out) {
      Thing x = {.x = 5, .y = 10};
      [[hls_pipeline_init_interval(1)]]
      for (int i=0;i<2;++i) {
        out.write(x);
        x = in.read();
      }
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_GE(func->slices.size(), 2);

  auto slice_it = func->slices.begin();
  ++slice_it;
  const xlscc::GeneratedFunctionSlice& second_slice = *slice_it;

  EXPECT_TRUE(SliceInputsDecl(second_slice, "x", /*direct_in=*/false,
                              /*is_feedback=*/false, /*func=*/func,
                              /*decl_index=*/0));
  EXPECT_TRUE(SliceInputsDecl(second_slice, "x", /*direct_in=*/false,
                              /*is_feedback=*/true, /*func=*/func,
                              /*decl_index=*/0));
}

TEST_F(ContinuationsTest, StaticThisDecomposed) {
  const std::string content = R"(
    struct Block {
      int x;
      long y;

      void Run(__xls_channel<int>& in,
               __xls_channel<int>& out) {
        [[hls_pipeline_init_interval(1)]]
        for (int i=0;i<8;++i) {
          [[hls_pipeline_init_interval(1)]]
          for (int j=0;j<8;++j) {
            y += x;
          }
        }
        [[hls_pipeline_init_interval(1)]]
        for (int x = 0; x < 4; ++x) {
        }
      }
    };

    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      static Block block;
      block.Run(in, out);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  (void)func;
  ASSERT_EQ(func->slices.size(), 7);

  const xlscc::GeneratedFunctionSlice& last_slice = func->slices.back();

  EXPECT_TRUE(SliceInputsDecl(last_slice, "block", /*direct_in=*/std::nullopt,
                              /*is_feedback=*/std::nullopt,
                              /*func=*/std::nullopt, /*decl_index=*/0));
  EXPECT_TRUE(SliceInputsDecl(last_slice, "block", /*direct_in=*/std::nullopt,
                              /*is_feedback=*/std::nullopt,
                              /*func=*/std::nullopt, /*decl_index=*/1));
}

class DecomposeTest : public XlsccTestBase {
 public:
  void SetUp() override {
    XlsccTestBase::SetUp();
    package_ = std::make_unique<xls::Package>("my_package");
  }
};

TEST_F(DecomposeTest, TypeIsDecomposable) {
  xls::Type* u32 = package_->GetBitsType(32);
  xls::Type* u8 = package_->GetBitsType(8);
  xls::Type* tuple = package_->GetTupleType({u32, u8});
  xls::Type* nested = package_->GetTupleType({tuple, u32});
  xls::Type* array = package_->GetArrayType(2, u32);

  EXPECT_FALSE(TypeIsDecomposable(u32));
  EXPECT_TRUE(TypeIsDecomposable(tuple));
  EXPECT_TRUE(TypeIsDecomposable(nested));
  EXPECT_FALSE(TypeIsDecomposable(array));
}

TEST_F(DecomposeTest, DecomposeTupleTypes) {
  xls::Type* u32 = package_->GetBitsType(32);
  xls::Type* u8 = package_->GetBitsType(8);
  xls::Type* tuple = package_->GetTupleType({u32, u8});
  xls::Type* nested = package_->GetTupleType({tuple, u32});
  xls::Type* array = package_->GetArrayType(2, u32);

  {
    auto decomposed = DecomposeTupleTypes(u32);
    ASSERT_EQ(decomposed.size(), 1);
    EXPECT_EQ(decomposed[0], u32);
  }
  {
    auto decomposed = DecomposeTupleTypes(tuple);
    ASSERT_EQ(decomposed.size(), 2);
    EXPECT_EQ(decomposed[0], u32);
    EXPECT_EQ(decomposed[1], u8);
  }
  {
    auto decomposed = DecomposeTupleTypes(nested);
    ASSERT_EQ(decomposed.size(), 3);
    EXPECT_EQ(decomposed[0], u32);
    EXPECT_EQ(decomposed[1], u8);
    EXPECT_EQ(decomposed[2], u32);
  }
  {
    auto decomposed = DecomposeTupleTypes(array);
    ASSERT_EQ(decomposed.size(), 1);
    EXPECT_EQ(decomposed[0], array);
  }
}

TEST_F(DecomposeTest, DecomposeValue) {
  xls::Type* u32 = package_->GetBitsType(32);
  xls::Type* u8 = package_->GetBitsType(8);
  xls::Type* tuple_t = package_->GetTupleType({u32, u8});
  xls::Type* nested_t = package_->GetTupleType({tuple_t, u32});

  xls::Value v_u32 = xls::Value(xls::UBits(10, 32));
  xls::Value v_u8 = xls::Value(xls::UBits(5, 8));
  xls::Value v_tuple = xls::Value::Tuple({v_u32, v_u8});
  xls::Value v_nested = xls::Value::Tuple({v_tuple, v_u32});

  {
    XLS_ASSERT_OK_AND_ASSIGN(auto decomposed, DecomposeValue(u32, v_u32));
    ASSERT_EQ(decomposed.size(), 1);
    EXPECT_EQ(decomposed[0], v_u32);
  }
  {
    XLS_ASSERT_OK_AND_ASSIGN(auto decomposed, DecomposeValue(tuple_t, v_tuple));
    ASSERT_EQ(decomposed.size(), 2);
    EXPECT_EQ(decomposed[0], v_u32);
    EXPECT_EQ(decomposed[1], v_u8);
  }
  {
    XLS_ASSERT_OK_AND_ASSIGN(auto decomposed,
                             DecomposeValue(nested_t, v_nested));
    ASSERT_EQ(decomposed.size(), 3);
    EXPECT_EQ(decomposed[0], v_u32);
    EXPECT_EQ(decomposed[1], v_u8);
    EXPECT_EQ(decomposed[2], v_u32);
  }
}

TEST_F(DecomposeTest, DecomposeTuples) {
  xls::FunctionBuilder fb("test", package_.get());
  xls::Type* u32 = package_->GetBitsType(32);
  xls::Type* u8 = package_->GetBitsType(8);
  xls::Type* tuple_t = package_->GetTupleType({u32, u8});
  xls::Type* nested_t = package_->GetTupleType({tuple_t, u32});
  xls::Type* array_t = package_->GetArrayType(2, u32);

  NATIVE_BVAL param = fb.Param("p", nested_t);

  XLS_ASSERT_OK_AND_ASSIGN(auto decomposed, DecomposeTuples(param.node()));
  ASSERT_EQ(decomposed.size(), 3);
  EXPECT_TRUE(decomposed[0]->GetType()->IsEqualTo(u32));
  EXPECT_TRUE(decomposed[1]->GetType()->IsEqualTo(u8));
  EXPECT_TRUE(decomposed[2]->GetType()->IsEqualTo(u32));

  NATIVE_BVAL param_arr = fb.Param("p_arr", array_t);
  XLS_ASSERT_OK_AND_ASSIGN(auto decomposed_arr,
                           DecomposeTuples(param_arr.node()));
  ASSERT_EQ(decomposed_arr.size(), 1);
  EXPECT_TRUE(decomposed_arr[0]->GetType()->IsEqualTo(array_t));
}

TEST_F(DecomposeTest, ComposeTuples) {
  xls::FunctionBuilder fb("test", package_.get());
  xls::Type* u32 = package_->GetBitsType(32);
  xls::Type* u8 = package_->GetBitsType(8);
  xls::Type* tuple_t = package_->GetTupleType({u32, u8});
  xls::Type* nested_t = package_->GetTupleType({tuple_t, u32});

  NATIVE_BVAL p0 = fb.Param("p0", u32);
  NATIVE_BVAL p1 = fb.Param("p1", u8);
  NATIVE_BVAL p2 = fb.Param("p2", u32);

  absl::InlinedVector<xls::Node*, 1> nodes = {p0.node(), p1.node(), p2.node()};

  XLS_ASSERT_OK_AND_ASSIGN(xls::Node * composed,
                           ComposeTuples("composed", nested_t, fb.function(),
                                         xls::SourceInfo(), nodes));

  EXPECT_TRUE(composed->GetType()->IsEqualTo(nested_t));
}

TEST_F(ContinuationsTest, StructTypes) {
  const std::string content = R"(
    struct Inner {
      int x;
      int y;
    };
    struct Outer {
      Inner foo;
      Inner bar;
    };

    #pragma hls_top
    Outer my_package() {
      return Outer();
    })";

  XLS_ASSERT_OK_AND_ASSIGN(const xlscc::GeneratedFunction* func,
                           GenerateTopFunction(content));

  ASSERT_EQ(func->slices.size(), 1);
  const xls::Function* xls_func = func->slices.front().function;

  xls::Type* u32 = package_->GetBitsType(32);
  xls::Type* outer_type = xls_func->return_type();
  ASSERT_TRUE(outer_type->IsTuple());
  ASSERT_EQ(outer_type->AsTupleOrDie()->size(), 2);
  xls::Type* inner_type = outer_type->AsTupleOrDie()->element_type(0);

  EXPECT_TRUE(TypeIsDecomposable(inner_type));
  EXPECT_TRUE(TypeIsDecomposable(outer_type));

  // DecomposeTupleTypes
  {
    auto decomposed = DecomposeTupleTypes(outer_type);
    ASSERT_EQ(decomposed.size(), 4);
    EXPECT_EQ(decomposed[0], u32);
    EXPECT_EQ(decomposed[1], u32);
    EXPECT_EQ(decomposed[2], u32);
    EXPECT_EQ(decomposed[3], u32);
  }

  // DecomposeValue
  xls::Value v_u32_1 = xls::Value(xls::UBits(1, 32));
  xls::Value v_u32_2 = xls::Value(xls::UBits(2, 32));
  xls::Value v_u32_3 = xls::Value(xls::UBits(3, 32));
  xls::Value v_u32_4 = xls::Value(xls::UBits(4, 32));
  xls::Value v_inner_1 = xls::Value::Tuple({v_u32_1, v_u32_2});
  xls::Value v_inner_2 = xls::Value::Tuple({v_u32_3, v_u32_4});
  xls::Value v_outer = xls::Value::Tuple({v_inner_1, v_inner_2});

  {
    XLS_ASSERT_OK_AND_ASSIGN(auto decomposed,
                             DecomposeValue(outer_type, v_outer));
    ASSERT_EQ(decomposed.size(), 4);
    EXPECT_EQ(decomposed[0], v_u32_1);
    EXPECT_EQ(decomposed[1], v_u32_2);
    EXPECT_EQ(decomposed[2], v_u32_3);
    EXPECT_EQ(decomposed[3], v_u32_4);
  }

  // DecomposeTuples & ComposeTuples
  xls::FunctionBuilder fb("test_structs", package_.get());
  NATIVE_BVAL param = fb.Param("p", outer_type);

  XLS_ASSERT_OK_AND_ASSIGN(auto decomposed_nodes,
                           DecomposeTuples(param.node()));
  ASSERT_EQ(decomposed_nodes.size(), 4);
  EXPECT_TRUE(decomposed_nodes[0]->GetType()->IsEqualTo(u32));

  XLS_ASSERT_OK_AND_ASSIGN(xls::Node * composed,
                           ComposeTuples("composed", outer_type, fb.function(),
                                         xls::SourceInfo(), decomposed_nodes));

  EXPECT_TRUE(composed->GetType()->IsEqualTo(outer_type));
}

}  // namespace
}  // namespace xlscc
