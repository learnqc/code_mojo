# Circuit CLI (stateful)

This is a small, stateful CLI that builds and simulates circuits using the
Butterfly Mojo simulator. It is designed to be driven by a human or by an AI
tool-caller that can send line-based commands to a long-running process.

Run it:

```
pixi run mojo run -I . tools/circuit_cli.mojo -- --prompt
```

Example session:

```
create 3
h 0
x 1
cp 0 2 pi/4
show circuit
show state
show grid 2 --log --bin
```

Supported commands:

- create/reset/clear
- strategy/threads
- h/x/y/z, p/rx/ry/rz
- cx/cy/cz, cp/crx/cry/crz, ccx, mcp
- swap, measure, bitrev, qrev, permute
- show circuit/state/grid
- help, quit/exit

Notes:
- This CLI is stateful only for the lifetime of the process.
- For AI function calling, keep the process alive and stream commands over
  stdin to preserve circuit state across tool invocations.
