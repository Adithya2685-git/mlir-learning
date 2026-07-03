#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


const int64_t VOCAB_SIZE = 32128;
const int64_t PAD_TOKEN = 0;
const int64_t EOS_TOKEN = 1;
const int64_t MAX_DECODER_LEN = 32;

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {
void _mlir_ciface_flan_decoder(
    MemRefDescriptor<float, 3> *output,  // 3D output!
    MemRefDescriptor<int64_t, 2> *input_ids,
    MemRefDescriptor<int64_t, 2> *attention_mask,
    MemRefDescriptor<int64_t, 2> *decoder_input_ids_padded,
    MemRefDescriptor<int64_t, 2> *decoder_attention_mask);
}

std::vector<int64_t> generate_tokens(std::vector<int64_t>& input_ids, std::vector<int64_t>& attention_mask) {
  int64_t offset = 0;
  int64_t input_size = static_cast<int64_t>(input_ids.size());
  MemRefDescriptor<int64_t, 2> input_ids_MemRef = {
      input_ids.data(), input_ids.data(), offset, {1, input_size},{input_size, 1}};
  MemRefDescriptor<int64_t, 2> encoder_attention_mask_MemRef = {
      attention_mask.data(),
      attention_mask.data(),
      offset,
      {1, input_size},
      {input_size, 1}};

  // Decoder buffers
  int64_t decoder_input_ids_padded[1][MAX_DECODER_LEN];
  int64_t decoder_attention_mask[1][MAX_DECODER_LEN];
  
  for (int i = 0; i < MAX_DECODER_LEN; i++) {
    decoder_input_ids_padded[0][i] = PAD_TOKEN;
    decoder_attention_mask[0][i] = 0;
  }

  std::vector<int64_t> generated_tokens;
  generated_tokens.push_back(PAD_TOKEN);

  // Model outputs 3D tensor: (batch=1, seq_len=32, vocab=32128)
  std::vector<float> outputData(1 * MAX_DECODER_LEN * VOCAB_SIZE);
  
  const int max_steps = 20;

  for (int step = 0; step < max_steps; step++) {
    int64_t current_len = static_cast<int64_t>(generated_tokens.size());
    
    if (current_len > MAX_DECODER_LEN) {
      std::cout << "✗ ERROR: Generated sequence exceeds max length (" << MAX_DECODER_LEN << ")\n";
      break;
    }

    // Setup decoder input and attention mask
    for (int i = 0; i < current_len; i++) {
      decoder_input_ids_padded[0][i] = generated_tokens[i];
      decoder_attention_mask[0][i] = 1;
    }
    for (int i = current_len; i < MAX_DECODER_LEN; i++) {
      decoder_input_ids_padded[0][i] = PAD_TOKEN;
      decoder_attention_mask[0][i] = 0;
    }

    std::fill(outputData.begin(), outputData.end(), -999.0f);

    std::cout << "\n=== Step " << step << " ===\n";
    std::cout << "Decoder sequence (" << current_len << " tokens): [";
    for (size_t j = 0; j < generated_tokens.size(); ++j) {
      std::cout << generated_tokens[j];
      if (j < generated_tokens.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    MemRefDescriptor<int64_t, 2> decoder_input_ids_padded_MemRef = {
        (int64_t *)decoder_input_ids_padded,
        (int64_t *)decoder_input_ids_padded,
        0,
        {1, MAX_DECODER_LEN},
        {MAX_DECODER_LEN, 1}};

    MemRefDescriptor<int64_t, 2> decoder_attention_mask_MemRef = {
        (int64_t *)decoder_attention_mask,
        (int64_t *)decoder_attention_mask,
        0,
        {1, MAX_DECODER_LEN},
        {MAX_DECODER_LEN, 1}};

    // Output: 3D memref (batch=1, seq_len=32, vocab=32128)
    // Strides: [seq_len*vocab, vocab, 1] for row-major layout
    MemRefDescriptor<float, 3> outputMemRef = {
        outputData.data(), outputData.data(), 0, 
        {1, MAX_DECODER_LEN, VOCAB_SIZE}, 
        {MAX_DECODER_LEN * VOCAB_SIZE, VOCAB_SIZE, 1}};

    std::cout << "Calling model...\n";
    _mlir_ciface_flan_decoder(
        &outputMemRef,
        &input_ids_MemRef,
        &encoder_attention_mask_MemRef,
        &decoder_input_ids_padded_MemRef,
        &decoder_attention_mask_MemRef);

    std::cout << "Output shape: [" << outputMemRef.sizes[0] << ", " 
              << outputMemRef.sizes[1] << ", " << outputMemRef.sizes[2] << "]\n";

    float *output = outputMemRef.aligned;

    // Extract logits for the LAST real position: [batch=0, pos=current_len-1, :]
    // In the 3D array: output[0 * stride[0] + (current_len-1) * stride[1] + token_id * stride[2]]
    // With strides [seq_len*vocab, vocab, 1]: output[(current_len-1) * VOCAB_SIZE + token_id]
    int64_t last_pos = current_len - 1;
    int64_t offset_to_last_pos = last_pos * VOCAB_SIZE;

    // Validate output
    bool has_valid_output = false;
    int valid_count = 0;
    for (int j = 0; j < VOCAB_SIZE; j++) {
      float val = output[offset_to_last_pos + j];
      if (val != -999.0f && !std::isnan(val) && !std::isinf(val)) {
        has_valid_output = true;
        valid_count++;
      }
    }

    std::cout << "Valid logits at position " << last_pos << ": " << valid_count << " / " << VOCAB_SIZE << "\n";

    if (!has_valid_output) {
      std::cout << "✗ ERROR: No valid output!\n";
      break;
    }

    // Find top 5 predictions from the last position
    std::vector<std::pair<float, int>> scores;
    for (int j = 0; j < VOCAB_SIZE; j++) {
      float logit = output[offset_to_last_pos + j];
      if (!std::isnan(logit) && !std::isinf(logit)) {
        scores.push_back({logit, j});
      }
    }

    std::partial_sort(
        scores.begin(), 
        scores.begin() + std::min(5, (int)scores.size()),
        scores.end(),
        [](const auto &a, const auto &b) { return a.first > b.first; });

    std::cout << "Top 5 predictions:\n";
    for (int j = 0; j < std::min(5, (int)scores.size()); j++) {
      std::cout << "  [" << j << "] Token " << std::setw(5) << scores[j].second 
                << " logit: " << std::fixed << std::setprecision(4) << scores[j].first << "\n";
    }

    int64_t next_token = scores[0].second;
    std::cout << "→ Selected: " << next_token << "\n";

    generated_tokens.push_back(next_token);

    if (next_token == EOS_TOKEN) {
      std::cout << "✓ EOS token generated, stopping.\n";
      break;
    }
  }

  std::cout << "\n" << std::string(60, '=') << "\n";
  std::cout << "Final sequence (" << generated_tokens.size() << " tokens): [";
  for (size_t i = 0; i < generated_tokens.size(); i++) {
    std::cout << generated_tokens[i];
    if (i < generated_tokens.size() - 1) std::cout << ", ";
  }
  std::cout << "]\n";
  std::cout << std::string(60, '=') << "\n";

  return generated_tokens;
}

PYBIND11_MODULE(flan_call, m) {
    m.def("generate_tokens", &generate_tokens, "TODO");
}