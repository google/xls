#include "xls/ir/op_specification.h"

#include "gtest/gtest.h"

namespace xls {

TEST(OpSpecificationTest, CanGetSingleton) {
    const auto& singleton = GetOpClassKindsSingleton();
    EXPECT_NE(singleton.size(), 0);
}

}  // namespace xls