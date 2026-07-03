#include <iomanip>
#include <iostream>

#include <cstdint>
#include <cstdio>

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {
void _mlir_ciface_sample_model(MemRefDescriptor<float, 2> *output,
                               MemRefDescriptor<float, 2> *input);
}

int main(int argc, char *argv[]) {
  float inputData[3][4];
  float outputData[3][5];

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      inputData[i][j] = 1.0;
    }
  }
  // Create MemRef descriptors
  int64_t offset = 0;
  int64_t input_sizes[2] = {3, 4};
  int64_t input_strides[2] = {4, 1}; // row-major layout

  int64_t output_size[2] = {3, 5};
  int64_t output_strides[2] = {5, 1}; // row-major layout

  MemRefDescriptor<float, 2> inputMemRef = {
      (float *)inputData,
      (float *)inputData,
      offset,
      {input_sizes[0], input_sizes[1]},
      {input_strides[0], input_strides[1]}};

  MemRefDescriptor<float, 2> outputMemRef = {
      (float *)outputData,
      (float *)outputData,
      offset,
      {output_size[0], output_size[1]},
      {output_strides[0], output_strides[1]}};

  // Call the model
  _mlir_ciface_sample_model(&outputMemRef, &inputMemRef);

  // Access output buffer
  float *output = (float *)outputMemRef.aligned;

  for (int64_t i = 0; i < output_size[0]; ++i) {
    for (int64_t j = 0; j < output_size[1]; ++j)
      std::cout << std::fixed << std::setprecision(5)
                << output[i * output_strides[0] + j] << ' ';
    std::cout << "\n";
  }

  return 0;
}