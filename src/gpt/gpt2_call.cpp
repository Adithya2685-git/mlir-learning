// GPT-2 GPU host program — calls compiled MLIR model
// Links with libmlir_cuda_runtime.so and libcuda.so/libcudart.so
//
// Compile:
//   g++ -o gpt2_run gpt2_call.cpp /home/adi/tmp/b5_final.o \
//       -L/opt/cuda/lib64 -lcudart -lcuda \
//       -L/home/adi/Projects/Honours/MLIR/llvm-project/build/lib \
//       -lmlir_cuda_runtime -lmlir_c_runner_utils -lm

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cuda.h>

// The compiled function signature (decomposed memref ABI):
//   gpt2_decoder(
//     input_ids: memref<1x?xi64> → allocated_ptr, aligned_ptr, offset, s0, s1, stride0, stride1
//     attn_mask: memref<1x?xi64> → allocated_ptr, aligned_ptr, offset, s0, s1, stride0, stride1
//   ) → struct { ptr, ptr, offset, sizes[3], strides[3] }  (memref<1x?x50257xf32>)
extern "C" {
struct MemRef3D {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[3];
  int64_t strides[3];
};

MemRef3D gpt2_decoder(
    // Input IDs: memref<1x?xi64>
    int64_t *in_alloc, int64_t *in_align, int64_t in_offs,
    int64_t in_s0, int64_t in_s1, int64_t in_str0, int64_t in_str1,
    // Attn mask: memref<1x?xi64>
    int64_t *mask_alloc, int64_t *mask_align, int64_t mask_offs,
    int64_t mask_s0, int64_t mask_s1, int64_t mask_str0, int64_t mask_str1);
}

int main() {
  // Initialize CUDA driver API
  CUresult err = cuInit(0);
  if (err != CUDA_SUCCESS) {
    printf("cuInit failed: %d\n", err);
    return 1;
  }
  printf("CUDA initialized OK\n");

  const int seq_len = 5;
  const int vocab = 50257;

  // Token IDs for "The capital of France is" (matches PyTorch reference)
  // PyTorch tokenizer output: [464, 3139, 286, 4881, 318]
  int64_t input_ids[] = {464, 3139, 286, 4881, 318};
  int64_t attention_mask[] = {1, 1, 1, 1, 1};

  int64_t sizes[2] = {1, seq_len};
  int64_t strides[2] = {seq_len, 1};

  printf("Calling GPT-2 model (seq_len=%d)...\n", seq_len);

  MemRef3D output = gpt2_decoder(
      input_ids, input_ids, 0, sizes[0], sizes[1], strides[0], strides[1],
      attention_mask, attention_mask, 0, sizes[0], sizes[1], strides[0], strides[1]);

  // Last token logits
  int64_t last_offset = (seq_len - 1) * vocab;
  float *last_logits = output.aligned + last_offset;

  printf("\nLast token logits (first 10 vocab):\n");
  for (int i = 0; i < 10; i++)
    printf("  vocab[%5d] = %14.6f\n", i, last_logits[i]);

  // Top-5
  printf("\nTop-5 predictions:\n");
  struct { float val; int idx; } top5[5] = {{-1e30f, 0}, {-1e30f, 0}, {-1e30f, 0}, {-1e30f, 0}, {-1e30f, 0}};
  for (int i = 0; i < vocab; i++) {
    float v = last_logits[i];
    // Find insertion point in sorted top5
    int pos = 4;
    while (pos >= 0 && v > top5[pos].val) pos--;
    pos++;
    if (pos < 5) {
      for (int k = 4; k > pos; k--) top5[k] = top5[k-1];
      top5[pos] = {v, i};
    }
  }

  printf("\nTop-5 predictions:\n");
  for (int i = 0; i < 5; i++)
    printf("  #%d: token_id=%5d  logit=%.4f\n", i+1, top5[i].idx, top5[i].val);

  // Free output memory (was allocated inside the model on CPU via malloc)
  free(output.allocated);

  return 0;
}
