# Backend Support Matrix

Fuse ships with three execution backends. This matrix summarises what works
today so you can pick the right engine (or spot missing features before filing
issues).

| Capability                         | NumPy Runner (`backend="numpy"`)                             | Torch FX Runner (`backend="torch"`)                                                | JAX Runner (`backend="jax"`)                              |
|------------------------------------|--------------------------------------------------------------|-------------------------------------------------------------------------------------|-----------------------------------------------------------|
| Core execution                     | ✅ Full coverage (evaluator implements all builtins)         | 🧩 FX graph lowering for dense programs; falls back to NumPy for streaming/logical  | 🧩 Experimental lowering; relies on JAX availability       |
| Gradient support                   | ✅ Symbolic gradients via `generate_gradient_program`        | ⚠️ Autograd integration limited; FX trace exposes graph for external autodiff      | ⚠️ Limited: JIT works for many ops, but gradients rely on NumPy path |
| Streaming / rolling indices        | ✅ Demand + fixpoint modes                                   | ❌ Not yet supported (forced NumPy fallback)                                        | ⚠️ Partially supported (compiled as pure JAX where possible) |
| Boolean Datalog operators          | ✅ Supported                                                  | ⚠️ Falls back to NumPy                                                              | ⚠️ Requires NumPy fallback                                 |
| Monte Carlo projection             | ✅ `ExecutionConfig(projection_strategy="monte_carlo")`      | 🧩 Falls back to NumPy evaluator                                                     | 🧩 Falls back to NumPy evaluator                           |
| Dtype support                      | `float32` default, `float16`/`bfloat16` via manual casts     | Input tensors follow Torch dtype; internal ops use `float32` by default             | `float32` default; respects JAX dtype policy               |
| Memory-mapped sources (`.npy/.npz`)| ✅ Respects `RuntimePolicies` (with strict mmap option)      | ✅ Inherits manifest handling from shared policy layer                              | ✅ Same manifest layer                                     |
| Device selection                   | CPU only                                                     | CPU/GPU via Torch’s device strings (`cpu`,`cuda`,`mps`, …)                          | CPU/GPU/TPU via `ExecutionConfig(device="...")`            |
| FX / export tooling                | N/A                                                          | ✅ FX graph available; packaging helpers in `fuse.interop`                          | ⚠️ Experimental ONNX export via NumPy fallback             |
| Known constraints                  | —                                                            | *Demand mode & Monte Carlo fall back to NumPy*<br>*Streaming unsupported*           | *Demand mode & Monte Carlo fall back to NumPy*<br>*Boolean logic falls back*  |

Legend:

* ✅ — implemented and used in the test suite
* 🧩 — partially implemented; expect sharp edges or fallbacks
* ⚠️ — implemented with caveats; consult source/tests
* ❌ — not supported today

### Notes

* All backends share manifest loading, runtime policies, and the DSL parser.
  Backend-specific code lives under `src/fuse/*_backend/`.
* Torch and JAX backends both default to 32-bit floats. When compiling with
  mixed precision, ensure the weight manifests supply appropriately typed arrays.
* The Torch backend returns FX `GraphModule`s and caches artifacts on demand;
  reuse the cache directory when packaging models.
* JAX compilation requires `jax[jaxlib] >= 0.4.30`. When JAX is absent, Fuse
  transparently falls back to NumPy execution.

Use this matrix to choose the backend that matches your deployment target, and
to understand which features might still need contribution work.
