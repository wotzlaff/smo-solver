import init, { minimize as minimizeWasm } from "./target/web/svm_browser_optimizer.mjs";

let loading;
let running = false;

/**
 * Load the WebAssembly optimizer. With no argument, load the WASM file next
 * to the generated bindings. Node callers can pass the compiled WASM bytes.
 */
export async function createOptimizer(wasm) {
  loading ??= (wasm === undefined ? init() : init({ module_or_path: wasm })).catch((error) => {
    loading = undefined;
    throw error;
  });
  await loading;
  return { minimize };
}

/**
 * Minimize a differentiable objective on a finite box.
 * evaluate(x) returns { value, gradient }. Trial points and the result use Float64Array.
 */
function minimize({ initial, lower, upper, evaluate, maxIterations = 100, gradientTolerance = 1e-6 }) {
  if (running) throw new Error("The optimizer is already running.");
  const n = initial.length;
  if (!Number.isSafeInteger(n) || n < 1 || n > 0x0fffffff ||
      lower.length !== n || upper.length !== n || typeof evaluate !== "function" ||
      !Number.isInteger(maxIterations) || maxIterations < 1 || maxIterations > 0xffffffff ||
      !Number.isFinite(gradientTolerance) || gradientTolerance <= 0) {
    throw new TypeError("Invalid dimensions, objective or optimizer options.");
  }

  function coordinates(values) {
    const result = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(values[i])) throw new TypeError("Coordinates and bounds must be finite.");
      result[i] = values[i];
    }
    return result;
  }

  running = true;
  try {
    const start = coordinates(initial);
    const lo = coordinates(lower);
    const hi = coordinates(upper);
    if (!start.every((x, i) => lo[i] <= x && x <= hi[i])) {
      throw new RangeError("Bounds must be ordered and contain the initial point.");
    }
    const result = minimizeWasm(start, lo, hi, maxIterations, gradientTolerance, (x) => {
      const { value, gradient } = evaluate(x);
      if (!Number.isFinite(value) || gradient.length !== n) {
        throw new TypeError("The objective must return a finite value and a gradient of length n.");
      }
      const output = new Float64Array(n + 1);
      output[0] = value;
      for (let i = 0; i < n; i++) {
        const entry = gradient[i];
        if (!Number.isFinite(entry)) throw new TypeError("The gradient must be finite.");
        output[i + 1] = entry;
      }
      return output;
    });
    return {
      x: result.slice(4),
      value: result[1],
      iterations: result[2],
      evaluations: result[3],
      status: ["converged", "iteration-limit", "stopped"][result[0]],
    };
  } finally {
    running = false;
  }
}
