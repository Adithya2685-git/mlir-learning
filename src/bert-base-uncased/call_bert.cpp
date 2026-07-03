#include <fstream>
#include <iostream>
#include <vector>

#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <math.h>

// Load the buffers from CSV file that was saved from PyTorch (get_buffers.py)
std::vector<std::vector<int64_t>> load_tensor() {
  std::vector<std::vector<int64_t>> result;
  std::ifstream file("tensor.csv");
  std::string line;

  std::stringstream ss;
  std::string token;

  while (std::getline(file, line)) {
    ss = std::stringstream(line);
    std::vector<int64_t> row;

    while (std::getline(ss, token, ',')) {
      row.push_back(std::stoi(token));
    }

    result.push_back(row);
  }
  return result;
}

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

struct Output {
  MemRefDescriptor<float, 3> output_last_hidden_state_MemRef;
  MemRefDescriptor<float, 2> output_pooler_output_MemRef;
};

extern "C" {
void _mlir_ciface_bert_model(
    Output *result,
    MemRefDescriptor<int64_t, 2> *buffer_embeddings_position_ids,
    MemRefDescriptor<int64_t, 2> *buffer_embeddings_token_type_ids,
    MemRefDescriptor<int64_t, 2> *input_ids,
    MemRefDescriptor<int64_t, 2> *token_type_ids,
    MemRefDescriptor<int64_t, 2> *attention_mask);
}

int main(int argc, char *argv[]) {
  // Load buffers
  auto tensor = load_tensor();

  // Initialize input tensors
  int64_t input_ids[1][12] = {
      {101, 2023, 2003, 1037, 7099, 7953, 2005, 14324, 2944, 9167, 1012, 102}};
  int64_t token_type_ids[1][12] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  int64_t attention_mask[1][12] = {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  // Output tensor
  float last_hidden_state[1][12][768];
  float pooler_output[1][768];

  // Create MemRef descriptors
  int64_t offset = 0;

  MemRefDescriptor<int64_t, 2> buffer_embeddings_position_ids = {
      tensor[0].data(), tensor[0].data(), offset, {1, 512}, {512, 1}};

  MemRefDescriptor<int64_t, 2> buffer_embeddings_token_type_ids = {
      tensor[1].data(), tensor[1].data(), offset, {1, 512}, {512, 1}};

  MemRefDescriptor<int64_t, 2> input_ids_MemRef = {
      (int64_t *)input_ids, (int64_t *)input_ids, offset, {1, 12}, {12, 1}};

  MemRefDescriptor<int64_t, 2> token_type_ids_MemRef = {
      (int64_t *)token_type_ids,
      (int64_t *)token_type_ids,
      offset,
      {1, 12},
      {12, 1}};

  MemRefDescriptor<int64_t, 2> attention_mask_MemRef = {
      (int64_t *)attention_mask,
      (int64_t *)attention_mask,
      offset,
      {1, 12},
      {12, 1}};

  MemRefDescriptor<float, 3> output_last_hidden_state_MemRef = {
      (float *)last_hidden_state,
      (float *)last_hidden_state,
      offset,
      {1, 12, 768},
      {768, 12, 1}};

  MemRefDescriptor<float, 2> output_pooler_output_MemRef = {
      (float *)pooler_output,
      (float *)pooler_output,
      offset,
      {1, 768},
      {768, 1}};

  Output result = {output_last_hidden_state_MemRef,
                   output_pooler_output_MemRef};

  // Call the model
  _mlir_ciface_bert_model(&result, &buffer_embeddings_position_ids,
                          &buffer_embeddings_token_type_ids, &input_ids_MemRef,
                          &token_type_ids_MemRef, &attention_mask_MemRef);

  float *output_last_hidden_state =
      (float *)result.output_last_hidden_state_MemRef.aligned;
  float *output_pooler_output =
      (float *)result.output_pooler_output_MemRef.aligned;

  // Print the first 10 values of each of the 12 output vectors
  int count = 0, i, j;
  std::cout << "Last hidden state (first 10 values of each of the 12 vectors):"
            << std::endl;
  for (i = 0; i < 12; ++i) {
    for (j = 0; j < 10; ++j) {
      std::cout << std::fixed << std::setprecision(5)
                << output_last_hidden_state[i * 768 + j] << ' ';
    }
    std::cout << std::endl;
  }

  // Print the first 10 values of the pooler output
  std::cout << "\nPooler output:" << std::endl;
  for (i = 0; i < 10; ++i) {
    std::cout << std::fixed << std::setprecision(5) << output_pooler_output[i]
              << ' ';
  }
  std::cout << std::endl;

  return 0;
}