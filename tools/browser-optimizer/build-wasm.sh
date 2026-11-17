#!/bin/sh
set -eu
cd "$(dirname "$0")"

cargo build --locked --release --target wasm32-unknown-unknown --target-dir target
wasm-bindgen --target web --no-typescript --out-dir target/web \
  target/wasm32-unknown-unknown/release/svm_browser_optimizer.wasm
mv target/web/svm_browser_optimizer.js target/web/svm_browser_optimizer.mjs
