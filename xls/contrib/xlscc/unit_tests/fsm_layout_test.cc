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
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "clang/include/clang/AST/Decl.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/generate_fsm.h"
#include "xls/contrib/xlscc/tracked_bvalue.h"
#include "xls/contrib/xlscc/translator_types.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/state_element.h"

namespace xlscc {

class FSMLayoutTest : public XlsccTestBase {
 public:
  absl::StatusOr<NewFSMLayout> GenerateTopFunction(
      std::string_view content,
      absl::flat_hash_map<const clang::NamedDecl*, xls::StateElement*>*
          state_element_for_static_out = nullptr) {
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

    NewFSMGenerator generator(*translator_, *translator_,
                              DebugIrTraceFlags_FSMStates,
                              split_states_on_channel_ops_);

    absl::flat_hash_map<const clang::NamedDecl*, xls::StateElement*>
        state_element_for_static;

    xls::ProcBuilder pb("test", package_.get());

    for (const auto& [decl, init_val] : func->static_values) {
      NATIVE_BVAL state_read_bval = pb.StateElement(
          decl->getNameAsString(), init_val.rvalue(), xls::SourceInfo());
      state_element_for_static[decl] =
          state_read_bval.node()->As<xls::StateRead>()->state_element();
    }

    if (state_element_for_static_out != nullptr) {
      *state_element_for_static_out = state_element_for_static;
    }

    return generator.LayoutNewFSM(*func, state_element_for_static,
                                  xls::SourceInfo());
  }

  std::vector<int64_t> FilterStates(
      const NewFSMLayout& layout, const int64_t find_slice_index,
      const absl::flat_hash_set<int64_t>& find_jumped_from_slice_indices) {
    std::vector<int64_t> ret;
    for (int64_t state_index = 0; state_index < layout.states.size();
         ++state_index) {
      const NewFSMState& state = layout.states.at(state_index);
      absl::flat_hash_set<int64_t> jumped_from_slice_indices;
      for (const JumpInfo& jump_info : state.jumped_from_slice_indices) {
        jumped_from_slice_indices.insert(jump_info.from_slice);
      }
      if (state.slice_index == find_slice_index &&
          jumped_from_slice_indices == find_jumped_from_slice_indices) {
        ret.push_back(state_index);
      }
    }
    return ret;
  }

  bool StateInputsDecl(const NewFSMLayout& layout, int64_t slice_index,
                       absl::flat_hash_set<int64_t> jumped_from_slice_indices,
                       std::string_view name,
                       std::optional<int64_t> from_slice_index = std::nullopt) {
    std::vector<int64_t> state_indices =
        FilterStates(layout, slice_index, jumped_from_slice_indices);
    EXPECT_EQ(state_indices.size(), 1L);
    const NewFSMState& state = layout.states.at(state_indices.front());
    for (const auto& [param, continuation_value] :
         state.current_inputs_by_input_param) {
      if (from_slice_index.has_value() &&
          from_slice_index.value() !=
              layout.output_slice_index_by_value.at(continuation_value)) {
        continue;
      }
      for (const auto& decl : continuation_value->decls) {
        if (decl->getNameAsString() == name) {
          return true;
        }
      }
    }
    return false;
  }

  // Returned the ContinuationValue saved, or nullptr if not found
  const ContinuationValue* StateSavesDecl(
      const NewFSMLayout& layout, int64_t slice_index,
      absl::flat_hash_set<int64_t> jumped_from_slice_indices,
      std::string_view name,
      std::optional<int64_t> from_slice_index = std::nullopt) {
    std::vector<int64_t> state_indices =
        FilterStates(layout, slice_index, jumped_from_slice_indices);
    EXPECT_EQ(state_indices.size(), 1L);
    const NewFSMState& state = layout.states.at(state_indices.front());
    for (const ContinuationValue* value : state.values_to_save) {
      if (from_slice_index.has_value() &&
          from_slice_index.value() !=
              layout.output_slice_index_by_value.at(value)) {
        continue;
      }
      for (const auto& decl : value->decls) {
        if (decl->getNameAsString() == name) {
          return value;
        }
      }
    }

    return nullptr;
  }

  int64_t StateSavesDeclCount(const NewFSMState& state, std::string_view name) {
    int64_t count = 0;
    for (const ContinuationValue* value : state.values_to_save) {
      bool count_value = false;
      for (const auto& decl : value->decls) {
        if (decl->getNameAsString() == name) {
          count_value = true;
        }
      }
      if (count_value) {
        ++count;
      }
    }

    return count;
  }

  void ExpectSliceTransition(const NewFSMLayout layout,
                             int64_t from_slice_index, int64_t to_slice_index,
                             bool unconditional) {
    EXPECT_TRUE(
        layout.transition_by_slice_from_index.contains(from_slice_index));
    const NewFSMActivationTransition& transition =
        layout.transition_by_slice_from_index.at(from_slice_index);
    EXPECT_EQ(transition.from_slice, from_slice_index);
    EXPECT_EQ(transition.to_slice, to_slice_index);
    EXPECT_EQ(transition.unconditional_forward, unconditional);
  }
};

namespace {

TEST_F(FSMLayoutTest, Basic) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int x = in.read();
      const int y = in.read();
      out.write(y);
      out.write(x);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(NewFSMLayout layout, GenerateTopFunction(content));

  EXPECT_FALSE(StateInputsDecl(layout, /*slice_index=*/0,
                               /*jumped_from_slice_indices=*/{}, "x"));
  EXPECT_FALSE(StateInputsDecl(layout, /*slice_index=*/1,
                               /*jumped_from_slice_indices=*/{}, "x"));
  EXPECT_FALSE(StateInputsDecl(layout, /*slice_index=*/2,
                               /*jumped_from_slice_indices=*/{}, "x"));
  EXPECT_TRUE(StateInputsDecl(layout, /*slice_index=*/3,
                              /*jumped_from_slice_indices=*/{}, "x",
                              /*from_slice_index=*/1));
  EXPECT_FALSE(StateInputsDecl(layout, /*slice_index=*/4,
                               /*jumped_from_slice_indices=*/{}, "x"));

  EXPECT_FALSE(StateSavesDecl(layout, /*slice_index=*/0,
                              /*jumped_from_slice_indices=*/{}, "x"));
  const ContinuationValue* x_value = nullptr;
  ASSERT_TRUE(x_value = StateSavesDecl(layout, /*slice_index=*/1,
                                       /*jumped_from_slice_indices=*/{}, "x",
                                       /*from_slice_index=*/1));
  EXPECT_FALSE(layout.state_element_by_continuation_value.contains(x_value));

  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/2,
                             /*jumped_from_slice_indices=*/{}, "x",
                             /*from_slice_index=*/1));
  EXPECT_FALSE(StateSavesDecl(layout, /*slice_index=*/3,
                              /*jumped_from_slice_indices=*/{}, "x"));
  EXPECT_FALSE(StateSavesDecl(layout, /*slice_index=*/4,
                              /*jumped_from_slice_indices=*/{}, "x"));
}

TEST_F(FSMLayoutTest, BasicReuse) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int x = in.read();
      int y = in.read();
      ++x;
      out.write(y);
      out.write(x);

      x += in.read();
      out.write(x);
    })";

  split_states_on_channel_ops_ = true;

  XLS_ASSERT_OK_AND_ASSIGN(NewFSMLayout layout, GenerateTopFunction(content));

  const ContinuationValue* x0_value = nullptr;
  ASSERT_TRUE(x0_value = StateSavesDecl(layout, /*slice_index=*/2,
                                        /*jumped_from_slice_indices=*/{}, "x",
                                        /*from_slice_index=*/2));
  ASSERT_TRUE(layout.state_element_by_continuation_value.contains(x0_value));

  const ContinuationValue* x1_value = nullptr;
  ASSERT_TRUE(x1_value = StateSavesDecl(layout, /*slice_index=*/4,
                                        /*jumped_from_slice_indices=*/{}, "x",
                                        /*from_slice_index=*/4));
  ASSERT_TRUE(layout.state_element_by_continuation_value.contains(x1_value));

  EXPECT_EQ(layout.state_element_by_continuation_value.at(x0_value),
            layout.state_element_by_continuation_value.at(x1_value));
}

TEST_F(FSMLayoutTest, BasicReuseInSubroutine) {
  const std::string content = R"(
    void Subroutine(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      int x = in.read();
      int y = in.read();
      ++x;
      out.write(y);
      out.write(x);

      x += in.read();
      out.write(x);
    }

    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      Subroutine(in, out);
    })";

  split_states_on_channel_ops_ = true;

  XLS_ASSERT_OK_AND_ASSIGN(NewFSMLayout layout, GenerateTopFunction(content));

  const ContinuationValue* x0_value = nullptr;
  ASSERT_TRUE(x0_value = StateSavesDecl(layout, /*slice_index=*/2,
                                        /*jumped_from_slice_indices=*/{}, "x",
                                        /*from_slice_index=*/2));
  ASSERT_TRUE(layout.state_element_by_continuation_value.contains(x0_value));

  const ContinuationValue* x1_value = nullptr;
  ASSERT_TRUE(x1_value = StateSavesDecl(layout, /*slice_index=*/4,
                                        /*jumped_from_slice_indices=*/{}, "x",
                                        /*from_slice_index=*/4));
  ASSERT_TRUE(layout.state_element_by_continuation_value.contains(x1_value));

  EXPECT_EQ(layout.state_element_by_continuation_value.at(x0_value),
            layout.state_element_by_continuation_value.at(x1_value));
}

TEST_F(FSMLayoutTest, DirectIn) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(int&direct_in,
                    __xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int x = in.read();
      const int y = in.read();
      out.write(y);
      out.write(x + direct_in);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(NewFSMLayout layout, GenerateTopFunction(content));

  for (const NewFSMState& state : layout.states) {
    EXPECT_EQ(StateSavesDeclCount(state, "direct_in"), 0);
  }
}

TEST_F(FSMLayoutTest, PipelinedLoop) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int r = in.read();
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for (int i=0;i<10;++i) {
        a += r;
      }
      out.write(a);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(NewFSMLayout layout, GenerateTopFunction(content));

  ExpectSliceTransition(layout, /*from_slice_index=*/2, /*to_slice_index=*/2,
                        /*unconditional=*/false);

  EXPECT_TRUE(StateInputsDecl(layout, /*slice_index=*/2,
                              /*jumped_from_slice_indices=*/{}, "r",
                              /*from_slice_index=*/1));
  EXPECT_TRUE(StateInputsDecl(layout, /*slice_index=*/2,
                              /*jumped_from_slice_indices=*/{}, "a",
                              /*from_slice_index=*/1));
  EXPECT_TRUE(StateInputsDecl(layout, /*slice_index=*/2,
                              /*jumped_from_slice_indices=*/{}, "i",
                              /*from_slice_index=*/1));

  EXPECT_TRUE(StateInputsDecl(layout, /*slice_index=*/2,
                              /*jumped_from_slice_indices=*/{2}, "r",
                              /*from_slice_index=*/1));
  EXPECT_TRUE(StateInputsDecl(layout, /*slice_index=*/2,
                              /*jumped_from_slice_indices=*/{2}, "a",
                              /*from_slice_index=*/2));
  EXPECT_TRUE(StateInputsDecl(layout, /*slice_index=*/2,
                              /*jumped_from_slice_indices=*/{2}, "i",
                              /*from_slice_index=*/2));

  EXPECT_TRUE(StateInputsDecl(layout, /*slice_index=*/3,
                              /*jumped_from_slice_indices=*/{}, "a",
                              /*from_slice_index=*/2));
  EXPECT_FALSE(StateInputsDecl(layout, /*slice_index=*/3,
                               /*jumped_from_slice_indices=*/{}, "r"));

  EXPECT_FALSE(StateSavesDecl(layout, /*slice_index=*/1,
                              /*jumped_from_slice_indices=*/{}, "a",
                              /*from_slice_index=*/1));
  EXPECT_FALSE(StateSavesDecl(layout, /*slice_index=*/1,
                              /*jumped_from_slice_indices=*/{}, "i",
                              /*from_slice_index=*/1));
  const ContinuationValue* r_value = nullptr;
  ASSERT_TRUE(r_value = StateSavesDecl(layout, /*slice_index=*/1,
                                       /*jumped_from_slice_indices=*/{}, "r",
                                       /*from_slice_index=*/1));
  ASSERT_TRUE(layout.state_element_by_continuation_value.contains(r_value));
  const int64_t r_index =
      layout.state_element_by_continuation_value.at(r_value);

  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/2,
                             /*jumped_from_slice_indices=*/{}, "a",
                             /*from_slice_index=*/2));
  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/2,
                             /*jumped_from_slice_indices=*/{}, "i",
                             /*from_slice_index=*/2));
  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/2,
                             /*jumped_from_slice_indices=*/{}, "r",
                             /*from_slice_index=*/1));

  const ContinuationValue* a_value = nullptr;
  ASSERT_TRUE(a_value = StateSavesDecl(layout, /*slice_index=*/2,
                                       /*jumped_from_slice_indices=*/{2}, "a",
                                       /*from_slice_index=*/2));
  ASSERT_TRUE(layout.state_element_by_continuation_value.contains(a_value));
  const int64_t a_index =
      layout.state_element_by_continuation_value.at(a_value);
  EXPECT_NE(r_index, a_index);

  const ContinuationValue* i_value = nullptr;
  ASSERT_TRUE(i_value = StateSavesDecl(layout, /*slice_index=*/2,
                                       /*jumped_from_slice_indices=*/{2}, "i",
                                       /*from_slice_index=*/2));
  ASSERT_TRUE(layout.state_element_by_continuation_value.contains(i_value));
  const int64_t i_index =
      layout.state_element_by_continuation_value.at(i_value);
  EXPECT_NE(i_index, a_index);
  EXPECT_NE(i_index, r_index);

  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/2,
                             /*jumped_from_slice_indices=*/{2}, "r",
                             /*from_slice_index=*/1));

  EXPECT_FALSE(StateSavesDecl(layout, /*slice_index=*/3,
                              /*jumped_from_slice_indices=*/{}, "r"));
}

TEST_F(FSMLayoutTest, PipelinedLoopBypass) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int r = in.read();
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for (int i=0;i<10;++i) {
        a += 5;
      }
      out.write(a + r);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(NewFSMLayout layout, GenerateTopFunction(content));

  ExpectSliceTransition(layout, /*from_slice_index=*/2, /*to_slice_index=*/2,
                        /*unconditional=*/false);

  EXPECT_FALSE(StateInputsDecl(layout, /*slice_index=*/2,
                               /*jumped_from_slice_indices=*/{}, "r"));
  EXPECT_FALSE(StateInputsDecl(layout, /*slice_index=*/2,
                               /*jumped_from_slice_indices=*/{2}, "r"));
  EXPECT_TRUE(StateInputsDecl(layout, /*slice_index=*/3,
                              /*jumped_from_slice_indices=*/{}, "r",
                              /*from_slice_index=*/1));

  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/1,
                             /*jumped_from_slice_indices=*/{}, "r",
                             /*from_slice_index=*/1));
  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/2,
                             /*jumped_from_slice_indices=*/{}, "r",
                             /*from_slice_index=*/1));
  EXPECT_TRUE(StateSavesDecl(layout, /*slice_index=*/2,
                             /*jumped_from_slice_indices=*/{2}, "r",
                             /*from_slice_index=*/1));

  EXPECT_FALSE(StateSavesDecl(layout, /*slice_index=*/3,
                              /*jumped_from_slice_indices=*/{}, "r"));
}

TEST_F(FSMLayoutTest, PipelinedLoopNested) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int r = in.read();
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for (int i=0;i<10;++i) {
        #pragma hls_pipeline_init_interval 1
        for (int j=0;j<10;++j) {
          a += r;
        }
      }
      out.write(a);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(NewFSMLayout layout, GenerateTopFunction(content));

  ExpectSliceTransition(layout, /*from_slice_index=*/3, /*to_slice_index=*/3,
                        /*unconditional=*/false);
  ExpectSliceTransition(layout, /*from_slice_index=*/4, /*to_slice_index=*/2,
                        /*unconditional=*/false);

  for (const NewFSMState& state : layout.states) {
    EXPECT_LE(StateSavesDeclCount(state, "j"), 1);
    EXPECT_LE(StateSavesDeclCount(state, "i"), 1);
    EXPECT_LE(StateSavesDeclCount(state, "a"), 1);
  }
}

TEST_F(FSMLayoutTest, PipelinedLoopSerialShared) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int r = in.read();
      int a = 0;
      #pragma hls_pipeline_init_interval 1
      for (int i=0;i<10;++i) {
        a += 5;
      }
      #pragma hls_pipeline_init_interval 1
      for (int i=0;i<10;++i) {
        a += r;
      }
      out.write(a);
    })";

  XLS_ASSERT_OK_AND_ASSIGN(NewFSMLayout layout, GenerateTopFunction(content));

  const ContinuationValue* a0_value = nullptr;
  ASSERT_TRUE(a0_value = StateSavesDecl(layout, /*slice_index=*/2,
                                        /*jumped_from_slice_indices=*/{}, "a",
                                        /*from_slice_index=*/2));
  ASSERT_TRUE(layout.state_element_by_continuation_value.contains(a0_value));
  const int64_t a0_index =
      layout.state_element_by_continuation_value.at(a0_value);

  const ContinuationValue* a1_value = nullptr;
  ASSERT_TRUE(a1_value = StateSavesDecl(layout, /*slice_index=*/4,
                                        /*jumped_from_slice_indices=*/{}, "a",
                                        /*from_slice_index=*/4));
  ASSERT_TRUE(layout.state_element_by_continuation_value.contains(a1_value));
  const int64_t a1_index =
      layout.state_element_by_continuation_value.at(a1_value);

  EXPECT_EQ(a0_index, a1_index);

  const ContinuationValue* i0_value = nullptr;
  ASSERT_TRUE(i0_value = StateSavesDecl(layout, /*slice_index=*/2,
                                        /*jumped_from_slice_indices=*/{}, "i",
                                        /*from_slice_index=*/2));
  ASSERT_TRUE(layout.state_element_by_continuation_value.contains(i0_value));
  const int64_t i0_index =
      layout.state_element_by_continuation_value.at(i0_value);

  const ContinuationValue* i1_value = nullptr;
  ASSERT_TRUE(i1_value = StateSavesDecl(layout, /*slice_index=*/4,
                                        /*jumped_from_slice_indices=*/{}, "i",
                                        /*from_slice_index=*/4));
  ASSERT_TRUE(layout.state_element_by_continuation_value.contains(i1_value));
  const int64_t i1_index =
      layout.state_element_by_continuation_value.at(i1_value);

  // State element sharing is by decl
  EXPECT_NE(i0_index, i1_index);

  const ContinuationValue* r_value = nullptr;
  ASSERT_TRUE(r_value = StateSavesDecl(layout, /*slice_index=*/1,
                                       /*jumped_from_slice_indices=*/{}, "r",
                                       /*from_slice_index=*/1));
  EXPECT_TRUE(layout.state_element_by_continuation_value.contains(r_value));

  EXPECT_EQ(layout.output_slice_index_by_value.at(r_value), 1);
}

TEST_F(FSMLayoutTest, IOActivationTransitions) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int x = in.read();
      const int y = in.read();
      out.write(y);
      out.write(x);
    })";
  split_states_on_channel_ops_ = true;

  XLS_ASSERT_OK_AND_ASSIGN(NewFSMLayout layout, GenerateTopFunction(content));

  ExpectSliceTransition(layout, /*from_slice_index=*/2, /*to_slice_index=*/3,
                        /*unconditional=*/true);
  ExpectSliceTransition(layout, /*from_slice_index=*/6, /*to_slice_index=*/7,
                        /*unconditional=*/true);

  const ContinuationValue* x_value = nullptr;
  ASSERT_TRUE(x_value = StateSavesDecl(layout, /*slice_index=*/2,
                                       /*jumped_from_slice_indices=*/{}, "x",
                                       /*from_slice_index=*/2));
  EXPECT_TRUE(layout.state_element_by_continuation_value.contains(x_value));
}

TEST_F(FSMLayoutTest, IOActivationTransitionsMultiTransition) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      const int x = in.read();
      out.write(x);
      const int y = in.read();
      out.write(y);
    })";
  split_states_on_channel_ops_ = true;

  XLS_ASSERT_OK_AND_ASSIGN(NewFSMLayout layout, GenerateTopFunction(content));

  ExpectSliceTransition(layout, /*from_slice_index=*/4, /*to_slice_index=*/5,
                        /*unconditional=*/true);
}

TEST_F(FSMLayoutTest, IOActivationTransitionsNoTransition) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in1,
                    __xls_channel<int>& in2,
                    __xls_channel<int>& out1,
                    __xls_channel<int>& out2) {
      const int x = in1.read();
      out1.write(x);
      const int y = in2.read();
      out2.write(y);
    })";
  split_states_on_channel_ops_ = true;

  XLS_ASSERT_OK_AND_ASSIGN(NewFSMLayout layout, GenerateTopFunction(content));

  EXPECT_TRUE(layout.transition_by_slice_from_index.empty());
}

TEST_F(FSMLayoutTest, StaticStateElementShared) {
  const std::string content = R"(
    #pragma hls_top
    void my_package(__xls_channel<int>& in,
                    __xls_channel<int>& out) {
      static int count = 0;
      const int x = in.read();
      count += x;
      const int y = in.read();
      out.write(count + y);
    })";
  split_states_on_channel_ops_ = true;

  absl::flat_hash_map<const clang::NamedDecl*, xls::StateElement*>
      state_element_for_static;

  XLS_ASSERT_OK_AND_ASSIGN(
      NewFSMLayout layout,
      GenerateTopFunction(content, &state_element_for_static));

  ASSERT_EQ(layout.state_elements.size(), 1);
  ASSERT_EQ(state_element_for_static.size(), 1);
  EXPECT_EQ(layout.state_elements.front().existing_state_element,
            state_element_for_static.begin()->second);
}

}  // namespace
}  // namespace xlscc
