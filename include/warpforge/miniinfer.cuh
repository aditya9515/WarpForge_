#pragma once

#include <warpforge/gemm.cuh>
#include <warpforge/runtime.cuh>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace warpforge {

struct MiniInferConfig final {
    std::size_t batch{1U};
    std::size_t sequence{128U};
    std::size_t hidden_size{512U};
    std::size_t attention_heads{8U};
    std::size_t head_dimension{64U};
    std::size_t intermediate_size{1536U};
    float rms_norm_epsilon{1.0e-5F};
    float rope_base{10000.0F};
    std::size_t position_offset{0U};
};

void validate_miniinfer_config(const MiniInferConfig& config);
[[nodiscard]] std::size_t miniinfer_token_count(const MiniInferConfig& config);
[[nodiscard]] std::size_t miniinfer_hidden_element_count(const MiniInferConfig& config);
[[nodiscard]] std::size_t miniinfer_intermediate_element_count(const MiniInferConfig& config);
[[nodiscard]] std::size_t miniinfer_score_element_count(const MiniInferConfig& config);

struct MiniInferHostWeights final {
    std::vector<float> attention_norm_weight;
    std::vector<float> query_weight;
    std::vector<float> key_weight;
    std::vector<float> value_weight;
    std::vector<float> output_weight;
    std::vector<float> ffn_norm_weight;
    std::vector<float> gate_weight;
    std::vector<float> up_weight;
    std::vector<float> down_weight;
};

class MiniInferWeights final {
public:
    explicit MiniInferWeights(const MiniInferConfig& config);
    ~MiniInferWeights() noexcept = default;

    MiniInferWeights(const MiniInferWeights&) = delete;
    MiniInferWeights& operator=(const MiniInferWeights&) = delete;
    MiniInferWeights(MiniInferWeights&&) = delete;
    MiniInferWeights& operator=(MiniInferWeights&&) = delete;

    void upload(const MiniInferHostWeights& weights, cudaStream_t stream);
    [[nodiscard]] bool uploaded() const noexcept { return uploaded_; }
    [[nodiscard]] std::size_t parameter_bytes() const noexcept;

    [[nodiscard]] const TensorView& attention_norm_view() const noexcept {
        return attention_norm_view_;
    }
    [[nodiscard]] const TensorView& query_view() const noexcept { return query_view_; }
    [[nodiscard]] const TensorView& key_view() const noexcept { return key_view_; }
    [[nodiscard]] const TensorView& value_view() const noexcept { return value_view_; }
    [[nodiscard]] const TensorView& output_view() const noexcept { return output_view_; }
    [[nodiscard]] const TensorView& ffn_norm_view() const noexcept { return ffn_norm_view_; }
    [[nodiscard]] const TensorView& gate_view() const noexcept { return gate_view_; }
    [[nodiscard]] const TensorView& up_view() const noexcept { return up_view_; }
    [[nodiscard]] const TensorView& down_view() const noexcept { return down_view_; }

private:
    MiniInferConfig config_;
    Tensor attention_norm_;
    Tensor query_;
    Tensor key_;
    Tensor value_;
    Tensor output_;
    Tensor ffn_norm_;
    Tensor gate_;
    Tensor up_;
    Tensor down_;
    TensorView attention_norm_view_;
    TensorView query_view_;
    TensorView key_view_;
    TensorView value_view_;
    TensorView output_view_;
    TensorView ffn_norm_view_;
    TensorView gate_view_;
    TensorView up_view_;
    TensorView down_view_;
    bool uploaded_{false};
};

struct MiniInferActivations final {
    TensorView input_norm;
    TensorView query;
    TensorView key;
    TensorView value;
    TensorView query_rope;
    TensorView key_rope;
    TensorView attention_scores;
    TensorView masked_scores;
    TensorView attention_probabilities;
    TensorView attention_context;
    TensorView attention_output;
    TensorView attention_residual;
    TensorView post_attention_norm;
    TensorView mlp_gate;
    TensorView mlp_up;
    TensorView mlp_swiglu;
    TensorView mlp_output;
};

class MiniInferWorkspace final {
public:
    explicit MiniInferWorkspace(const MiniInferConfig& config);
    ~MiniInferWorkspace() noexcept = default;

    MiniInferWorkspace(const MiniInferWorkspace&) = delete;
    MiniInferWorkspace& operator=(const MiniInferWorkspace&) = delete;
    MiniInferWorkspace(MiniInferWorkspace&&) = delete;
    MiniInferWorkspace& operator=(MiniInferWorkspace&&) = delete;

    [[nodiscard]] MiniInferActivations& activations() noexcept { return activations_; }
    [[nodiscard]] const MiniInferActivations& activations() const noexcept {
        return activations_;
    }
    [[nodiscard]] void* native_handle() noexcept { return storage_.native_handle(); }
    [[nodiscard]] std::size_t capacity_bytes() const noexcept {
        return storage_.capacity_bytes();
    }
    [[nodiscard]] std::size_t used_bytes() const noexcept { return storage_.used_bytes(); }

private:
    DeviceWorkspace storage_;
    MiniInferActivations activations_;
};

using MiniInferIntermediateObserver =
    std::function<void(std::string_view, const TensorView&, cudaStream_t)>;

class MiniInferBlock final {
public:
    explicit MiniInferBlock(MiniInferConfig config = {});
    ~MiniInferBlock() noexcept = default;

    MiniInferBlock(const MiniInferBlock&) = delete;
    MiniInferBlock& operator=(const MiniInferBlock&) = delete;
    MiniInferBlock(MiniInferBlock&&) = delete;
    MiniInferBlock& operator=(MiniInferBlock&&) = delete;

    void upload_weights(const MiniInferHostWeights& weights, cudaStream_t stream);
    void forward(
        const TensorView& input,
        TensorView& output,
        GemmBackend backend,
        cublasHandle_t cublas_handle,
        cudaStream_t stream,
        const MiniInferIntermediateObserver& observer = {});

    [[nodiscard]] const MiniInferConfig& config() const noexcept { return config_; }
    [[nodiscard]] const MiniInferWeights& weights() const noexcept { return weights_; }
    [[nodiscard]] MiniInferWorkspace& workspace() noexcept { return workspace_; }
    [[nodiscard]] const MiniInferWorkspace& workspace() const noexcept { return workspace_; }

private:
    MiniInferConfig config_;
    MiniInferWeights weights_;
    MiniInferWorkspace workspace_;
};

struct MiniInferFixture final {
    std::string preset;
    std::uint64_t seed{};
    MiniInferConfig config;
    MiniInferHostWeights weights;
    std::vector<float> input;
    std::map<std::string, std::vector<float>> intermediates;
};

[[nodiscard]] MiniInferFixture load_miniinfer_fixture(
    const std::filesystem::path& directory);
[[nodiscard]] std::vector<std::string_view> miniinfer_intermediate_names();
[[nodiscard]] Tolerance miniinfer_intermediate_tolerance(
    std::string_view name,
    const MiniInferConfig& config) noexcept;

}  // namespace warpforge
