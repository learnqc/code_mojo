# Testing

Run a single Mojo test file with pixi:

```sh
pixi run mojo run -I . path/to/test.mojo
```

Example:

```sh
pixi run mojo run -I . tests/test_state.mojo
```

## Benchmarks

Design rule: benchmark files should only wire existing APIs. Do not embed
algorithmic logic inside benches; keep logic in `butterfly/core` and reuse it.
