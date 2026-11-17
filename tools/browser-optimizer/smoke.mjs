import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createOptimizer } from "./optimizer.mjs";
import { rosenbrock } from "./example.mjs";

const wasm = await readFile(new URL(
  "./target/web/svm_browser_optimizer_bg.wasm", import.meta.url,
));
const optimizer = await createOptimizer(wasm);

const shortRun = optimizer.minimize({ ...rosenbrock, maxIterations: 1 });
assert.equal(shortRun.status, "iteration-limit");
assert.ok(Number.isFinite(shortRun.value));
assert.throws(() => optimizer.minimize({ ...rosenbrock, initial: [1, 1] }), RangeError);

const failure = new Error("Objective evaluation failed");
assert.throws(() => optimizer.minimize({
  ...rosenbrock,
  evaluate() { throw failure; },
}), (error) => error === failure);
assert.throws(() => optimizer.minimize({
  ...rosenbrock,
  evaluate() { return { value: 0, gradient: [0, NaN] }; },
}), /gradient must be finite/);

let evaluations = 0;
const result = optimizer.minimize({
  ...rosenbrock,
  evaluate(x) {
    assert.ok(x.every((v, i) => v >= rosenbrock.lower[i] && v <= rosenbrock.upper[i]));
    evaluations++;
    return rosenbrock.evaluate(x);
  },
});
assert.equal(result.status, "converged");
assert.ok(Math.abs(result.x[0] - 0.8) < 1e-6);
assert.ok(Math.abs(result.x[1] - 0.64) < 1e-6);
assert.ok(Math.abs(result.value - 0.04) < 1e-8);
assert.equal(result.evaluations, evaluations);
console.log("WASM optimizer smoke passed", result);
