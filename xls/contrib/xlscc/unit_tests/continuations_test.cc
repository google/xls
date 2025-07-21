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
#include <optional>
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
#include "xls/contrib/xlscc/tracked_bvalue.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/translator_types.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"

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
      for (const clang::NamedDecl* decl : continuation_out.decls) {
        if (decl->getNameAsString() == name &&
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
      for (const clang::NamedDecl* decl : continuation_out.decls) {
        if (decl->getNameAsString() == name) {
          ++count;
        }
      }
    }
    return count;
  };

  bool SliceInputsDecl(const xlscc::GeneratedFunctionSlice& slice,
                       std::string_view name,
                       std::optional<bool> direct_in = std::nullopt) {
    for (const xlscc::ContinuationInput& continuation_in :
         slice.continuations_in) {
      for (const clang::NamedDecl* decl : continuation_in.decls) {
        if (decl->getNameAsString() == name &&
            (!direct_in.has_value() ||
             direct_in.value() ==
                 continuation_in.continuation_out->direct_in)) {
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
      for (const clang::NamedDecl* decl : continuation_in.decls) {
        if (decl->getNameAsString() == name) {
          ++count;
        }
      }
    }
    return count;
  }

  bool SliceInputDoesNotInputBothDecls(
      const xlscc::GeneratedFunctionSlice& slice, std::string_view name_a,
      std::string_view name_b) {
    absl::flat_hash_set<const clang::NamedDecl*> decls_found;
    bool found = false;
    for (const xlscc::ContinuationInput& continuation_in :
         slice.continuations_in) {
      for (const clang::NamedDecl* decl : continuation_in.decls) {
        if (decl->getNameAsString() == name_a) {
          decls_found = continuation_in.decls;
          found = true;
          continue;
        }
      }
    }
    if (!found) {
      return true;
    }
    for (const clang::NamedDecl* decl : decls_found) {
      if (decl->getNameAsString() == name_b) {
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
  EXPECT_TRUE(SliceOutputsDecl(third_slice, "x"));
  EXPECT_FALSE(SliceOutputsDecl(fourth_slice, "x"));
  EXPECT_FALSE(SliceOutputsDecl(fifth_slice, "x"));
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
  EXPECT_TRUE(SliceInputsDecl(third_slice, "a"));
  EXPECT_TRUE(SliceInputsDecl(third_slice, "i"));

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
  EXPECT_EQ(SliceInputsDeclCount(fourth_slice, "a"), 2);
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

  EXPECT_TRUE(SliceOutputsDecl(first_slice, "a"));
  EXPECT_TRUE(SliceOutputsDecl(first_slice, "i"));
  EXPECT_FALSE(SliceOutputsDecl(first_slice, "c"));

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
  EXPECT_EQ(SliceInputsDeclCount(fourth_slice, "a"), 2);
  EXPECT_EQ(SliceInputsDeclCount(fourth_slice, "j"), 2);
  EXPECT_EQ(SliceOutputsDeclCount(fourth_slice, "a"), 1);
  EXPECT_EQ(SliceOutputsDeclCount(fourth_slice, "j"), 1);

  EXPECT_EQ(SliceInputsDeclCount(fifth_slice, "ctrl"), 1);
  EXPECT_EQ(SliceInputsDeclCount(fifth_slice, "i"), 2);
  EXPECT_FALSE(SliceOutputsDecl(fifth_slice, "a"));
  EXPECT_FALSE(SliceInputsDecl(fifth_slice, "a"));
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

}  // namespace
}  // namespace xlscc
