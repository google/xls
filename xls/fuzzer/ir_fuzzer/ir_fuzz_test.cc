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

#include <memory>

#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/matchers.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/verifier.h"
#include "xls/passes/reassociation_pass.h"

namespace xls {
namespace {

// Perform tests on the Package object which contains the IR. This is a general
// test that just verifies if the Package object is valid.
void VerifyIrFuzzPackage(PackageAndFuzzProgram package_and_fuzz_program) {
  std::unique_ptr<Package>& p = package_and_fuzz_program.p;
  VLOG(3) << "IR Fuzzer-2: IR:" << "\n" << p->DumpIr() << "\n";
}
// Use of gtest FUZZ_TEST to randomly generate IR while being compatible with
// Google infrastructure. The IrFuzzTest function is called and represents the
// main test logic. A domain is specified to define the range of possible values
// that the FuzzProgram protobuf can have when generating random values.
FUZZ_TEST(IrFuzzTest, VerifyIrFuzzPackage).WithDomains(IrFuzzDomain());

// This test makes sure that the OptimizationPassChangesOutputs test works for
// this specific example.
TEST(IrFuzzTest, PassChangesOutputsWithBitsParam) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_list_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            type {
              bits {
                bit_width: 10
              }
            }
          }
        }
      )");
  ReassociationPass pass;
  XLS_ASSERT_OK(
      PassChangesOutputsWithProto(proto_string, /*arg_set_count=*/10, pass));
}

TEST(IrFuzzTest, PassChangesOutputsWithTwoBitsParams) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_list_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            type {
              bits {
                bit_width: 10
              }
            }
          }
        }
        fuzz_ops {
          param {
            type {
              bits {
                bit_width: 20
              }
            }
          }
        }
      )");
  ReassociationPass pass;
  XLS_ASSERT_OK(
      PassChangesOutputsWithProto(proto_string, /*arg_set_count=*/10, pass));
}

TEST(IrFuzzTest, PassChangesOutputsWithTupleParam) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_list_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            type {
              tuple {
                tuple_elements {
                  bits {
                    bit_width: 10
                  }
                }
                tuple_elements {
                  bits {
                    bit_width: 20
                  }
                }
              }
            }
          }
        }
      )");
  ReassociationPass pass;
  XLS_ASSERT_OK(
      PassChangesOutputsWithProto(proto_string, /*arg_set_count=*/10, pass));
}

TEST(IrFuzzTest, PassChangesOutputsWithArrayParam) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_list_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            type {
              array {
                array_size: 2
                array_element {
                  bits {
                    bit_width: 10
                  }
                }
              }
            }
          }
        }
      )");
  ReassociationPass pass;
  XLS_ASSERT_OK(
      PassChangesOutputsWithProto(proto_string, /*arg_set_count=*/10, pass));
}

TEST(IrFuzzTest, PassChangesOutputsWithNestedTupleParam) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_list_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            type {
              tuple {
                tuple_elements {
                  tuple {
                    tuple_elements {
                      bits {
                        bit_width: 10
                      }
                    }
                  }
                }
                tuple_elements {
                  bits {
                    bit_width: 20
                  }
                }
              }
            }
          }
        }
      )");
  ReassociationPass pass;
  XLS_ASSERT_OK(
      PassChangesOutputsWithProto(proto_string, /*arg_set_count=*/10, pass));
}

TEST(IrFuzzTest, PassChangesOutputsWithEmptyArrayParam) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_list_method: TUPLE_LIST_METHOD
        fuzz_ops {
          param {
            type {
              array {
                array_size: 0
                array_element {
                  bits {
                    bit_width: 10
                  }
                }
              }
            }
          }
        }
      )");
  ReassociationPass pass;
  XLS_ASSERT_OK(
      PassChangesOutputsWithProto(proto_string, /*arg_set_count=*/10, pass));
}

}  // namespace
}  // namespace xls
