# TODO

## Done (this session)
- [x] Installed CUDA toolkit 13.3 (`/opt/cuda`)
- [x] Built `libmlir_cuda_runtime.so` (LLVM reconfigured with `-DMLIR_ENABLE_CUDA_RUNNER=ON -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc`)
- [x] Added LLVM + CUDA bin dirs to PATH (fish: `~/.config/fish/config.fish`)
- [x] Fixed all GPU scripts: RWTH paths → local paths
  - `externals/torch-mlir/build/lib` → `llvm-project/build/lib`
  - `/cvmfs/.../CUDA/12.6.3/lib64/` → `/opt/cuda/lib64`
- [x] Fixed `src/flan-t5-small/benchmark.sh` CPU path

## Environment status
- LLVM tools: `/home/adi/Projects/Honours/MLIR/llvm-project/build/bin`
- CUDA: `/opt/cuda` (13.3)
- `tutorial-opt`: `build-ninja/tools/tutorial-opt`
- Venv: `dev/` (Python 3.11.15)
- Available libs: `libmlir_c_runner_utils.so`, `libmlir_runner_utils.so`, `libmlir_cuda_runtime.so` in LLVM build

## Pending: build & verify all models

### CPU models
- [ ] `src/cnn` — has linalg MLIR, needs pipeline run
- [ ] `src/mnist` — has linalg MLIR, needs pipeline run
- [ ] `src/resnet18` — needs torch→linalg lowering + pipeline
- [ ] `src/bert-base-uncased` — needs torch→linalg lowering + pipeline
- [ ] `src/gpt` — needs torch→linalg lowering + pipeline
- [ ] `src/flan-t5-small` — needs torch→linalg lowering + pipeline (PIC shared lib)

### GPU models (infra ready, need artifacts)
- [ ] `src/sample/gpu` — needs GPU pipeline + compile
- [ ] `src/cnn/gpu` — needs GPU pipeline + compile
- [ ] `src/mnist/gpu` — needs GPU pipeline + compile
- [ ] `src/resnet18/gpu` — needs GPU pipeline + compile
- [ ] `src/flan-t5-small/gpu` — needs GPU pipeline + compile

### Tests
- [x] `matmul_to_blas.mlir` — passes
- [ ] Add more tests?
