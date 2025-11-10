# Butterfly: A Mojo Quantum Simulator

Minimal code to get started with the functional version which will be wrapped in quantum circuits.

## Repository structure
<pre>
├── README.md
├── butterfly                           
│   ├── __init__.mojo                          
│   ├── algos
│   │   ├── __init__.mojo             
│   │   ├── value_encoding.mojo                            
│   ├── core
│   │   ├── __init__.mojo             
│   │   ├── gates.mojo   
│   │   └── state.mojo                            
│   ├── utils
│   │   ├── __init__.mojo             
│   │   ├── utils.mojo  
├── benches                           
│   ├── __init__.mojo                          
│   ├── bench.mojo
├── tests                           
│   ├── __init__.mojo                          
│   ├── test_basic.mojo                       
│   ├── test_value_encoding.mojo                       
│   ├── test_measure.mojo
├── ...              
</pre>

## Using pixi as a package manager
Official documentation:

https://docs.modular.com/max/get-started

Other resources:

https://mojo-lang.com/miji/start/pixi.html

Adjust the content of pixi.toml as needed:

```aiignore
[workspace]
authors = ["Logical Guess"]
channels = ["https://conda.modular.com/max", "https://repo.prefix.dev/modular-community", "conda-forge"]
name = "butterfly"
platforms = ["osx-arm64"]
version = "0.1.0"

[tasks]
package = "pixi run mojo package butterfly"
bench = "pixi run package && cp butterfly.mojopkg tests/ && pixi run mojo run benches/bench.mojo -I tests/ && rm benches/butterfly.mojopkg"
test_basic = "pixi run package && cp butterfly.mojopkg tests/ && pixi run mojo run tests/test_basic.mojo -I tests/ && rm tests/butterfly.mojopkg"
test_encode_value = "pixi run package && cp butterfly.mojopkg tests/ && pixi run mojo run tests/test_encode_value.mojo -I tests/ && rm tests/butterfly.mojopkg"
test_measure = "pixi run package && cp butterfly.mojopkg tests/ && pixi run mojo run tests/test_measure.mojo -I tests/ && rm tests/butterfly.mojopkg"

[dependencies]
mojo = "==0.25.6"
```

## Running tests and benchmarks

[<img src="/assets/value_encoding.png">]()

```bash
 pixi run test_basic
```

```bash
 pixi run test_encode_value
```

```bash
test_adjacent_pairs
```

```bash
 pixi run test_measure
```

```bash
 pixi run bench_target
```



