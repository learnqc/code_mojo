# Benchmarks

Bench files live at `benches/bench_<topic>.mojo`.

Run a bench:

```sh
pixi run mojo run -I . benches/bench_<topic>.mojo
```

Save CSV output:

```sh
pixi run mojo run -I . benches/bench_<topic>.mojo --autosave
```

Suite definitions (optional) go in `benches/suites/`.
CSV output is written to `benches/results/`.
