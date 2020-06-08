workspace(name = "com_google_xls")

# Load and configure a hermetic LLVM based C/C++ toolchain. This is done here
# and not in load_external.bzl because it requires several sequential steps of
# declaring archives and using things in them, which is awkward to do in .bzl
# files because it's not allowed to use `load` inside of a function.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "com_grail_bazel_toolchain",
    urls = [
        "https://github.com/grailbio/bazel-toolchain/archive/f4c17a3ae40f927ff62cc0fb8fe22b1530871807.zip",
    ],
    strip_prefix = "bazel-toolchain-f4c17a3ae40f927ff62cc0fb8fe22b1530871807",
    sha256 = "715fd98d566ed1304cb53e0c640427cf0916ec6db89588e3ac2b6a87632276d4",
    patches = ["//dependency_support/com_grail_bazel_toolchain:google_workstation_workaround.patch"],
)
load("@com_grail_bazel_toolchain//toolchain:deps.bzl", "bazel_toolchain_dependencies")
bazel_toolchain_dependencies()
load("@com_grail_bazel_toolchain//toolchain:rules.bzl", "llvm_toolchain")
llvm_toolchain(
    name = "llvm_toolchain",
    llvm_version = "10.0.0",
)
load("@llvm_toolchain//:toolchains.bzl", "llvm_register_toolchains")
llvm_register_toolchains()


load("//dependency_support:load_external.bzl", "load_external_repositories")
load_external_repositories()

load("//dependency_support:initialize_external.bzl", "initialize_external_repositories")
initialize_external_repositories()

load("@xls_pip_deps//:requirements.bzl", "pip_install")
pip_install()
