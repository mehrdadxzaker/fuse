# Fuse benchmark helpers

PYTHON ?= python3
BENCH_SCRIPT ?= benchmarks/attention_benchmark.py
SEQ ?= 1024
MEM ?= $(SEQ)
D_MODEL ?= 128
VALUE_DIM ?= 128
ITERATIONS ?= 30
WARMUP ?= 5
BACKEND ?= all
DEVICE ?= auto
SEED ?= 2024
EXTRA_ARGS ?=

.PHONY: bench-attention bench-attention-torch bench-attention-jax check-gpu

bench-attention:
	@$(PYTHON) $(BENCH_SCRIPT) \
		--backend $(BACKEND) \
		--device $(DEVICE) \
		--seq $(SEQ) \
		--mem $(MEM) \
		--d-model $(D_MODEL) \
		--value-dim $(VALUE_DIM) \
		--iterations $(ITERATIONS) \
		--warmup $(WARMUP) \
		--seed $(SEED) \
		$(EXTRA_ARGS)

bench-attention-torch:
	@$(MAKE) bench-attention BACKEND=torch DEVICE=$(DEVICE)

bench-attention-jax:
	@$(MAKE) bench-attention BACKEND=jax DEVICE=$(DEVICE)

check-gpu:
	@$(PYTHON) - <<'PY'
try:
    import torch
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    current = torch.cuda.current_device() if cuda else None
    name = torch.cuda.get_device_name(current) if cuda else None
    print(f"torch: version={torch.__version__}, cuda_available={cuda}, mps_available={mps}, device_name={name}")
except Exception as exc:
    print(f"torch: unavailable ({exc})")
try:
    import jax
    devices = jax.devices()
    if not devices:
        print("jax: no devices found")
    else:
        summary = ", ".join(f"{d.platform}:{getattr(d, 'id', '?')}" for d in devices)
        print(f"jax: version={jax.__version__}, devices=[{summary}]")
except Exception as exc:
    print(f"jax: unavailable ({exc})")
PY
