#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {
void _mlir_ciface_transformer_model(
    MemRefDescriptor<float, 2> *output, MemRefDescriptor<int64_t, 2> *input_ids,
    MemRefDescriptor<int64_t, 2> *attention_mask,
    MemRefDescriptor<int64_t, 2> *decoder_input_ids);
}

int main(int argc, char *argv[]) {
  // Example input IDs ("translate English to German: How are you?")
  int64_t input_ids[1][10] = {{13959, 1566, 12, 2968, 10, 571, 33, 25, 58, 1}};
  int64_t attention_mask[1][10] = {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  int64_t offset = 0;
  MemRefDescriptor<int64_t, 2> input_ids_MemRef = {
      (int64_t *)input_ids, (int64_t *)input_ids, offset, {1, 10}, {10, 1}};
  MemRefDescriptor<int64_t, 2> attention_mask_MemRef = {
      (int64_t *)attention_mask,
      (int64_t *)attention_mask,
      offset,
      {1, 10},
      {10, 1}};

  std::vector<int64_t> decoder_input_ids = {0};
  std::vector<float> outputData(32128);

  MemRefDescriptor<int64_t, 2> decoder_input_ids_MemRef = {
      decoder_input_ids.data(), decoder_input_ids.data(), 0, {1, 1}, {1, 1}};

  MemRefDescriptor<float, 2> outputMemRef = {
      outputData.data(), outputData.data(), 0, {1, 32128}, {32128, 1}};

  const int runs = 10;

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    _mlir_ciface_transformer_model(&outputMemRef, &input_ids_MemRef,
                                   &attention_mask_MemRef,
                                   &decoder_input_ids_MemRef);
  }

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < runs; ++i) {
    _mlir_ciface_transformer_model(&outputMemRef, &input_ids_MemRef,
                                   &attention_mask_MemRef,
                                   &decoder_input_ids_MemRef);
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  double seconds = elapsed.count();

  std::cout << "Avg inference time: " << std::fixed << std::setprecision(6)
            << (seconds / runs) << " sec\n";
  return 0;
}