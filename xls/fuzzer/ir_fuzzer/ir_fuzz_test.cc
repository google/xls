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
void VerifyIrFuzzPackage(std::shared_ptr<Package> p) {
  XLS_ASSERT_OK(VerifyPackage(p.get()));
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
        args_bytes: "\xf7\x80\x6a\x4a\x69\xe3\x12\x20\x0f\xbe\xab\x08\xcc\xeb\xf5\x14\xda\x6d\x08\xa5\x0c\x0e\xd0\xa5\x64\x02\xed\x35\x2c\xad\xa3\x23"
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
        args_bytes: "\xa0\x02\xca\x71\xf7\x9b\x6e\xf7\x5d\xf6\xf8\xee\xe3\xa9\xc8\x2e\xe3\xf9\x52\x2d\xc5\x0f\x63\x05\x48\x7e\x25\x1d\x9d\xe3\x54\xa5\xeb\x53\xae\xc3\x6f\x1d\x8c\xc0\x47\xfd\x88\x57\x91\x3b\x0a\x3a\xeb\x3f\xa3\xf2\x0c\xa1\x31\x45\x80\x75\xf5\xc5\x51\xb7\xf9\x1c"
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

}  // namespace
}  // namespace xls
