# Proc-scoped channels in XLS IR

[TOC]

Written on 2023/5/27

Relevant issues: [#869](https://github.com/google/xls/issues/869).

Channels in the IR should be changed from package-scoped to proc-scoped.
Proc-scoped channels have many advantages which outweigh the significant
refactoring effort required to make the change.

Currently in XLS IR channels are declared globally at the package level. For
example:

```
package foo

chan in_ch(bits[32], id=0, kind=streaming, ops=receive_only)
chan out_ch(bits[32], id=1, kind=streaming, ops=send_only)
chan a_ch(bits[32], id=2, kind=streaming, ops=send_receive)

proc bar(t: token, s: ()) {
  ...
  rcv: (token, bits[32]) = receive(t, channel_id=0)
  ...
  snd0: token = send(t, data0, channel_id=2)
  ...
  snd1: token = send(t, data1, channel_id=1)
  ...
}

proc quux(t: token, s: ()) {
  ...
  rcv: (token, bits[32]) = receive(t, channel_id=2)
  ...
}

```

All channels have a globally unique name and id. These channels are used inside
of procs in send and receive nodes. Each channel can be used by at most a single
send node and a single receive node.

Changing to proc-scoped channels would change the structure of the IR to more
closely match DSLX (one of the proposal’s numerous advantages). Specifically,
procs would be organized hierarchically where procs spawn other procs. Each proc
has zero or more channel parameters with which the proc is spawned. These
channel parameters define the interface of the proc.

Possible syntax with proc-scoped channels for the proc example above:

```
package foo

proc quux<in_ch: bits[32]>(t: token, st: ()) {
  ...
  rcv: (token, bits[32]) = receive(t, channel=in_ch)
  ...
}

proc bar<in_ch: bits[32] in,
         out_ch: bits[32] out>(t: token, st: ()) {
  // Channel declaration.
  chan a_ch(bits[32])

  // Spawn a child proc. Naming the spawn statement (“quux_inst”) provides a
  // mechanism for referring to a particular proc instance via a path of named
  // instantiations.
  quux_inst: spawn quux<a_ch>()

  rcv: (token, bits[32]) = receive(t, channel=in_ch)
  ...
  snd0: token = send(t, data0, channel=a_ch)
  ...
  snd1: token = send(t, data1, channel=out_ch)
  ...
}
```

### Advantages of proc-scoped channels

*   In DSLX, an empty top proc can be used to stitch together procs via spawn
    statements (`next` method is empty). Currently this is not representable in
    the IR because the empty top proc has no connections to the spawned procs in
    the IR. This means some designs supported in DSLX are not representable in
    the IR.

*   The hierarchical structure of procs naturally maps to the hierarchical
    structure of Verilog modules and the structure of procs in the frontend.

*   Avoids channel name collisions. Currently whenever a new channel is created
    (for example, during IR conversion or transformation) care must be taken to
    avoid name collisions resulting in awkward channel names. The DSLX frontend
    for example will name channels based on line and column number in the source
    code.. With this change channel names can have shorter, more natural names
    which translate to shorter more natural Verilog module port names.

*   The channel interface of the proc is clearly defined via the spawn
    arguments. These channels become the port interface of the Verilog module
    generated for the proc. Grouping the proc interface channels in one place
    also provides a natural point at which to specify metadata about the
    channels. For example, SRAMs have several different kinds of channels: read
    request, write request, read response and write response. These could be
    specified as metadata on the proc.

*   Multiple instantiations of the same proc only require a single IR definition
    of the proc. This could extend down to the emitted Verilog resulting in a
    more compact, more easily verified RTL.

*   Scoping channels to procs makes it easier to hold a channel pointer in send
    and receive nodes. Currently send and receive nodes hold a channel ID which
    requires translation to a pointer before using. Associating a pointer held
    by a node with a package-scoped construct is difficult in the face of
    addition, deletion and modification of procs and nodes.

*   Supports unused ports on Verilog modules. This is represented with an unused
    spawn channel argument. Currently an unused channel is not associated with
    any proc which prevents these channels from representing unused ports.

### Disadvantages/complexities:

*   This will be a major refactor which will likely require at least a couple
    weeks worth of work.

*   With proc scoped channels there is no longer a one-to-one correspondence
    between the Proc objects and instantiation of procs in the design. The same
    is true for channels. This will require changes in the interpreter and JIT
    (at least). Specifying a specific channel/proc instance requires a path of
    spawn statements.

*   Channels can be declared in a couple ways: in the spawn arguments of the
    proc where each argument represents half a channel, and channels can be
    declared inside of procs (which represents both sides of the channel).
    Likely multiple channel representations in the code will be required.
    Perhaps the spawn argument is a reference to one half of a channel, and the
    channel declaration in a proc creates the channel object itself.

## Representation in the IR

The IR data structures will have to change to support proc-scoped channels. A
new construct which is a reference to one side of a channel needs to be added.
This construct would be used for the channel parameters of the procs. These are
bound to actual channels by spawn statements. The channel metadata (fifo depth,
etc) would remain attached to the channel. The channel reference would only hold
direction and name.

Possible C++ implementation:

```
struct ChannelRef {
  // In (send) or out (receive).
  Direction direction;
  std::string name;
};

class Proc {
  ...
  // List of channels declared inside the proc.
  std::vector<std::unique_ptr<Channel>> channel_declarations_;

  // Spawn parameters of this proc. For example:
  //
  //   proc foo<a: chan<bits[32]>, b: chan<bits[42]>> (...)
  //
  // Spawn parameters are `a` and `b`.
  std::vector<ChannelRef> spawn_parameters_;
};
```

Send and receive nodes in the IR would naturally correspond to a ChannelRef
however it’s not clear whether ChannelRef data structures should be explicitly
constructed for the send and receive nodes.

Proc-scoped channels break the one-to-one correspondence between definitions of
channels (or procs) and instantiations of channels (or procs). Some XLS
components such as the interpreter and JIT naturally operate on channel/proc
instantiations rather than definitions. To construct instantiations, an
elaboration process walks the proc hierarchy and creates instantiation objects
for instance of channels and procs. Possible instance implementation:

```
class ChannelInstance {
  // Pointer to the channel construct in the IR.
  Channel* channel;
};

class ProcInstance {
  Proc* proc;
  std::vector<ChannelInstance*> spawn_arguments;

  // Channels declared in this proc instance.
  std::vector<ChannelInstance> channel_instances;

  // Proc instances spawned by this proc instance.
  std::vector<ProcInstance> spawned_procs;

  // Proc which spawned this instance.
  ProcInstance* parent;
};
```

The elaboration process would return a ProcInstance for the top-level proc with
child instances underneath.

The interpreter and JIT would use this elaboration during execution. Each proc
instance gets its own continuation and each channel instance gets its own
channel queue.

The elaboration would be constructed as needed from the IR rather than trying to
maintain a persistent elaboration in sync with the IR.

One unanswered question is how the interface channels of the top-level proc
should be handled. All channels deeper in the proc hierarchy necessarily have a
corresponding channel declaration within a proc in which metadata can be stored.
However the top-level proc is by definition not spawned by another proc so the
interface channels of the top-level proc have no corresponding channel
declarations. There are several possible solutions:

1.  Declare top-level channels at the package level along with a spawn statement
    of the top-level proc:

    ```
    package the_package

    chan x(bits[32], ...);
    chan y(bits[32], ...);
    spawn the_top_proc<x, y>();
    ```

1.  Create a degenerate proc with no interface channels which spawns the
    top-level proc:

    ```
    proc __fake_top<>(t: token, s: ()) {
      chan x(bits[32], ...);
      chan y(bits[32], ...);
      spawn the_top_proc<x, y>();

      next (t, s);
    }
    ```

1.  No top-level channels are declared in IR. Instead top-level Channel objects
    are created as part of the elaboration. Any necessary metadata (if there is
    any) of the top-level procs would be passed as options to the compilation or
    evaluation process.

Option (3) seems the cleanest. (1) requires adding channel and spawn support at
the package level as a weird special case. (2) is likely a better option than
(1) but it does require carrying around a dummy proc which would need to be
identified in some way as the “fake top”.

## Incremental roll out

This change is too large to be done in a single change and should be done
incrementally. Below are possible steps for an incremental roll out:

*   Add bit on Proc indicating it is a new style proc and add new fields to proc
    as described above which are only set if the bit is set. Initially, only
    support a proc being spawned once. This preserves the one-to-one
    correspondence between channels/procs and instantiations of channels/procs.

*   Add parsing and serialization support (channel declarations, spawn
    statements, etc).

*   Add interpreter and JIT support.

*   Add pass which converts the old-style procs of a package into a hierarchy of
    new style procs.

*   Add proc conversion pass to end of pipeline and update codegen to support
    new style procs. Then incrementally move the pass earlier and earlier in the
    pipeline updating passes as the conversion pass is moved.

*   After the conversion pass has been moved to the front of the pipeline,
    update the frontends.

*   Remove support for old style procs.

*   Add elaboration and change interpreter and jit to use elaboration results.
    This enables spawning a proc multiple times.
