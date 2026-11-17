# Browser L-BFGS-B prototype

A standalone Rust/WebAssembly adapter for [Basin's L-BFGS-B](https://basin.rs/).
The adapter accepts a differentiable objective with finite box bounds. The
JavaScript callback returns the value and its full gradient for the same point.
The optimizer starts at the supplied feasible point and returns a local result.

Build with Rust 1.87 or newer:

```sh
cd tools/browser-optimizer
rustup target add wasm32-unknown-unknown
cargo install --locked --version 0.2.130 wasm-bindgen-cli
./build-wasm.sh
node smoke.mjs
python3 -m http.server 8000
```

Open <http://localhost:8000/> for the bounded Rosenbrock example. To deploy
the example, copy `index.html`, `optimizer.mjs`, `example.mjs` and the generated
`target/web/` directory, preserving their relative paths. The adapter lives in
a separate Cargo workspace with its own Rust and WASM dependencies.

```js
import { createOptimizer } from "./optimizer.mjs";

const optimizer = await createOptimizer();
const result = optimizer.minimize({
  initial: [0, 0],
  lower: [-2, -2],
  upper: [2, 2],
  evaluate: ([x, y]) => ({
    value: (x - 1) ** 2 + (y + 3) ** 2,
    gradient: [2 * (x - 1), 2 * (y + 3)],
  }),
  maxIterations: 100,
  gradientTolerance: 1e-6,
});
```

The result includes the point, objective value, iteration and evaluation
counts, and a status (`converged`, `iteration-limit`, or `stopped`). Callback
exceptions and non-finite values are reported to the caller. The callback runs
synchronously, so create the optimizer inside a Web Worker for substantial
problems.

For SVM hyperparameter optimization, supply the objective and gradient in the
coordinates the optimizer sees, for example log λ and log kernel scaling.
The existing Python implementation provides exact CV hypergradients.
Browser integration requires an evaluator that computes the corresponding CV
objective and derivatives in JavaScript or WebAssembly, and should run in a
worker alongside the training solver. The Rosenbrock demo exercises the
optimizer and its bound handling directly.
