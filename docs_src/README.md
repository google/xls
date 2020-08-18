# XLS: Accelerated HW Synthesis

Note: This is not an officially supported Google product. Expect bugs and
sharp edges. Please help by trying it out, reporting bugs, and letting us know
what you think!

The XLS (Accelerated HW Synthesis) toolchain aims to enable the rapid
development of hardware IP via "software style" methodology. XLS is a High Level
Synthesis (HLS) toolchain which produces synthesizable designs from flexible,
high-level descriptions of functionality.

## Building From Source

Currently, XLS must be built from source using the Bazel build system.

The following instructions are for the Ubuntu 20.04 (Focal) Linux distribution:

```
# Follow the bazel install instructions:
# https://docs.bazel.build/versions/master/install-ubuntu.html
#
# Afterwards we observe:

$ bazel --version
bazel 3.2.0

$ sudo apt install python3-dev python3-distutils python3-dev libtinfo5

# py_binary currently assume they can refer to /usr/bin/env python
# even though Ubuntu 20.04 has no `python`, only `python3`.
# See https://github.com/bazelbuild/bazel/issues/8685

$ mkdir -p $HOME/opt/bin/
$ ln -s $(which python3) $HOME/opt/bin/python
$ echo 'export PATH=$HOME/opt/bin:$PATH' >> ~/.bashrc
$ source ~/.bashrc

$ bazel test -c opt ...
```
