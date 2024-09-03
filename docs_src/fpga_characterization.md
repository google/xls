# FPGA Characterization

Note that right now the in-tree yosys and nextpnr-ice40 builds plugins aren't
registering properly, see
[issue #188](https://github.com/google/xls/issues/188). As a result, we have to
use out-of-tree `yosys` and `nextpnr-ice40` builds for the moment.

```
$ bazel build -c opt //xls/synthesis/yosys:yosys_server_main
$ ./bazel-bin/xls/synthesis/yosys/yosys_server_main \
    --yosys_path $(which yosys) \
    --nextpnr_path $(which nextpnr-ice40) \
    --synthesis_target=ice40 \
    --alsologtostderr
```

The above runs a gRPC service, so in another terminal pane, we run the
characterization driver:

```
$ bazel run -c opt //xls/synthesis:timing_characterization_client_main
$ ./bazel-bin/xls/synthesis/timing_characterization_client_main \
    > ./xls/estimators/delay_model/models/ice40.textproto
```

This produces a textual representation of the delay model protobuf.

## Building In-Tree Binaries

Note that these cannot currently be used for the above characterization flow,
see [issue #188](https://github.com/google/xls/issues/188)

Build `yosys` and `nextpnr-ice40`:

```
$ bazel build -c opt @at_clifford_yosys//:yosys @nextpnr//:nextpnr-ice40
```
