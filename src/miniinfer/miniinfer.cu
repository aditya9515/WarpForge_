#include <warpforge/attention.cuh>
#include <warpforge/causal_mask.cuh>
#include <warpforge/elementwise.cuh>
#include <warpforge/fusion.cuh>
#include <warpforge/miniinfer.cuh>
#include <warpforge/rmsnorm.cuh>
#include <warpforge/rope.cuh>
#include <warpforge/softmax.cuh>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace warpforge {
namespace {

constexpr std::size_t workspace_alignment = 256U;
constexpr std::size_t activation_count = 17U;

[[nodiscard]] std::size_t checked_product(const std::size_t left, const std::size_t right,
                                          const char* label) {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(std::string(label) + " overflows size_t");
    }
    return left * right;
}

[[nodiscard]] std::size_t checked_sum(const std::size_t left, const std::size_t right,
                                      const char* label) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error(std::string(label) + " overflows size_t");
    }
    return left + right;
}

void validate_fp32_view(const TensorView& view, const TensorShape& expected_shape,
                        const char* label) {
    if (view.dtype() != DType::fp32 || view.shape().dimensions() != expected_shape.dimensions()) {
        throw std::invalid_argument(std::string(label) + " has the wrong shape or dtype");
    }
}

void validate_host_vector(const std::vector<float>& values, const std::size_t expected,
                          const char* label) {
    if (values.size() != expected) {
        throw std::invalid_argument(std::string(label) + " has " + std::to_string(values.size()) +
                                    " elements; expected " + std::to_string(expected));
    }
}

void upload_tensor(Tensor& destination, const std::vector<float>& source,
                   const cudaStream_t stream) {
    destination.copy_from_host_async(source.data(), source.size() * sizeof(float), stream);
}

[[nodiscard]] std::size_t workspace_capacity(const MiniInferConfig& config) {
    validate_miniinfer_config(config);
    const std::size_t hidden = miniinfer_hidden_element_count(config);
    const std::size_t intermediate = miniinfer_intermediate_element_count(config);
    const std::size_t scores = miniinfer_score_element_count(config);
    std::size_t elements = checked_product(hidden, 11U, "MiniInfer hidden workspace");
    elements = checked_sum(elements, checked_product(intermediate, 3U, "MiniInfer MLP workspace"),
                           "MiniInfer workspace elements");
    elements = checked_sum(elements, checked_product(scores, 3U, "MiniInfer attention workspace"),
                           "MiniInfer workspace elements");
    const std::size_t data_bytes =
        checked_product(elements, sizeof(float), "MiniInfer workspace bytes");
    const std::size_t alignment_slack = checked_product(activation_count, workspace_alignment - 1U,
                                                        "MiniInfer workspace alignment");
    return checked_sum(data_bytes, alignment_slack, "MiniInfer workspace capacity");
}

[[nodiscard]] TensorShape hidden_shape(const MiniInferConfig& config) {
    return TensorShape{config.batch, config.sequence, config.hidden_size};
}

[[nodiscard]] TensorShape qkv_shape(const MiniInferConfig& config) {
    return TensorShape{config.batch, config.sequence, config.attention_heads,
                       config.head_dimension};
}

[[nodiscard]] TensorShape score_shape(const MiniInferConfig& config) {
    return TensorShape{config.batch, config.attention_heads, config.sequence, config.sequence};
}

[[nodiscard]] TensorShape intermediate_shape(const MiniInferConfig& config) {
    return TensorShape{config.batch, config.sequence, config.intermediate_size};
}

[[nodiscard]] MiniInferConfig checked_config(MiniInferConfig config) {
    validate_miniinfer_config(config);
    return config;
}

} // namespace

void validate_miniinfer_config(const MiniInferConfig& config) {
    if (config.batch == 0U || config.sequence == 0U || config.hidden_size == 0U ||
        config.attention_heads == 0U || config.head_dimension == 0U ||
        config.intermediate_size == 0U) {
        throw std::invalid_argument("MiniInfer dimensions must be positive");
    }
    if (checked_product(config.attention_heads, config.head_dimension,
                        "MiniInfer attention width") != config.hidden_size) {
        throw std::invalid_argument("attention_heads * head_dimension must equal hidden_size");
    }
    if (config.head_dimension % 2U != 0U) {
        throw std::invalid_argument("MiniInfer head_dimension must be even for RoPE");
    }
    if (!std::isfinite(config.rms_norm_epsilon) || config.rms_norm_epsilon <= 0.0F) {
        throw std::invalid_argument("MiniInfer RMSNorm epsilon must be finite and positive");
    }
    if (!std::isfinite(config.rope_base) || config.rope_base <= 0.0F) {
        throw std::invalid_argument("MiniInfer RoPE base must be finite and positive");
    }
    if (config.position_offset > std::numeric_limits<std::size_t>::max() - (config.sequence - 1U)) {
        throw std::overflow_error("MiniInfer position range overflows size_t");
    }
    static_cast<void>(miniinfer_hidden_element_count(config));
    static_cast<void>(miniinfer_intermediate_element_count(config));
    static_cast<void>(miniinfer_score_element_count(config));
}

std::size_t miniinfer_token_count(const MiniInferConfig& config) {
    return checked_product(config.batch, config.sequence, "MiniInfer token count");
}

std::size_t miniinfer_hidden_element_count(const MiniInferConfig& config) {
    return checked_product(miniinfer_token_count(config), config.hidden_size,
                           "MiniInfer hidden elements");
}

std::size_t miniinfer_intermediate_element_count(const MiniInferConfig& config) {
    return checked_product(miniinfer_token_count(config), config.intermediate_size,
                           "MiniInfer intermediate elements");
}

std::size_t miniinfer_score_element_count(const MiniInferConfig& config) {
    const std::size_t head_groups =
        checked_product(config.batch, config.attention_heads, "MiniInfer attention head groups");
    const std::size_t rows =
        checked_product(head_groups, config.sequence, "MiniInfer attention rows");
    return checked_product(rows, config.sequence, "MiniInfer attention scores");
}

MiniInferWeights::MiniInferWeights(const MiniInferConfig& config)
    : config_(checked_config(config)),
      attention_norm_(TensorShape{config_.hidden_size}, DType::fp32),
      query_(TensorShape{config_.hidden_size, config_.hidden_size}, DType::fp32),
      key_(TensorShape{config_.hidden_size, config_.hidden_size}, DType::fp32),
      value_(TensorShape{config_.hidden_size, config_.hidden_size}, DType::fp32),
      output_(TensorShape{config_.hidden_size, config_.hidden_size}, DType::fp32),
      ffn_norm_(TensorShape{config_.hidden_size}, DType::fp32),
      gate_(TensorShape{config_.hidden_size, config_.intermediate_size}, DType::fp32),
      up_(TensorShape{config_.hidden_size, config_.intermediate_size}, DType::fp32),
      down_(TensorShape{config_.intermediate_size, config_.hidden_size}, DType::fp32),
      attention_norm_view_(attention_norm_.view()), query_view_(query_.view()),
      key_view_(key_.view()), value_view_(value_.view()), output_view_(output_.view()),
      ffn_norm_view_(ffn_norm_.view()), gate_view_(gate_.view()), up_view_(up_.view()),
      down_view_(down_.view()) {}

void MiniInferWeights::upload(const MiniInferHostWeights& weights, const cudaStream_t stream) {
    const std::size_t hidden_square =
        checked_product(config_.hidden_size, config_.hidden_size, "MiniInfer hidden weight");
    const std::size_t hidden_intermediate =
        checked_product(config_.hidden_size, config_.intermediate_size, "MiniInfer MLP weight");
    validate_host_vector(weights.attention_norm_weight, config_.hidden_size,
                         "attention_norm_weight");
    validate_host_vector(weights.query_weight, hidden_square, "query_weight");
    validate_host_vector(weights.key_weight, hidden_square, "key_weight");
    validate_host_vector(weights.value_weight, hidden_square, "value_weight");
    validate_host_vector(weights.output_weight, hidden_square, "output_weight");
    validate_host_vector(weights.ffn_norm_weight, config_.hidden_size, "ffn_norm_weight");
    validate_host_vector(weights.gate_weight, hidden_intermediate, "gate_weight");
    validate_host_vector(weights.up_weight, hidden_intermediate, "up_weight");
    validate_host_vector(weights.down_weight, hidden_intermediate, "down_weight");

    upload_tensor(attention_norm_, weights.attention_norm_weight, stream);
    upload_tensor(query_, weights.query_weight, stream);
    upload_tensor(key_, weights.key_weight, stream);
    upload_tensor(value_, weights.value_weight, stream);
    upload_tensor(output_, weights.output_weight, stream);
    upload_tensor(ffn_norm_, weights.ffn_norm_weight, stream);
    upload_tensor(gate_, weights.gate_weight, stream);
    upload_tensor(up_, weights.up_weight, stream);
    upload_tensor(down_, weights.down_weight, stream);
    uploaded_ = true;
}

std::size_t MiniInferWeights::parameter_bytes() const noexcept {
    return attention_norm_.byte_size() + query_.byte_size() + key_.byte_size() +
           value_.byte_size() + output_.byte_size() + ffn_norm_.byte_size() + gate_.byte_size() +
           up_.byte_size() + down_.byte_size();
}

MiniInferWorkspace::MiniInferWorkspace(const MiniInferConfig& config)
    : storage_(workspace_capacity(config)) {
    const TensorShape hidden = hidden_shape(config);
    const TensorShape qkv = qkv_shape(config);
    const TensorShape scores = score_shape(config);
    const TensorShape intermediate = intermediate_shape(config);
    activations_.input_norm = storage_.allocate_view(hidden, DType::fp32);
    activations_.query = storage_.allocate_view(qkv, DType::fp32);
    activations_.key = storage_.allocate_view(qkv, DType::fp32);
    activations_.value = storage_.allocate_view(qkv, DType::fp32);
    activations_.query_rope = storage_.allocate_view(qkv, DType::fp32);
    activations_.key_rope = storage_.allocate_view(qkv, DType::fp32);
    activations_.attention_scores = storage_.allocate_view(scores, DType::fp32);
    activations_.masked_scores = storage_.allocate_view(scores, DType::fp32);
    activations_.attention_probabilities = storage_.allocate_view(scores, DType::fp32);
    activations_.attention_context = storage_.allocate_view(qkv, DType::fp32);
    activations_.attention_output = storage_.allocate_view(hidden, DType::fp32);
    activations_.attention_residual = storage_.allocate_view(hidden, DType::fp32);
    activations_.post_attention_norm = storage_.allocate_view(hidden, DType::fp32);
    activations_.mlp_gate = storage_.allocate_view(intermediate, DType::fp32);
    activations_.mlp_up = storage_.allocate_view(intermediate, DType::fp32);
    activations_.mlp_swiglu = storage_.allocate_view(intermediate, DType::fp32);
    activations_.mlp_output = storage_.allocate_view(hidden, DType::fp32);
}

MiniInferBlock::MiniInferBlock(MiniInferConfig config)
    : config_(std::move(config)), weights_(config_), workspace_(config_) {
    validate_miniinfer_config(config_);
}

void MiniInferBlock::upload_weights(const MiniInferHostWeights& weights,
                                    const cudaStream_t stream) {
    weights_.upload(weights, stream);
}

void MiniInferBlock::forward(const TensorView& input, TensorView& output, const GemmBackend backend,
                             const cublasHandle_t cublas_handle, const cudaStream_t stream,
                             const MiniInferIntermediateObserver& observer) {
    if (!weights_.uploaded()) {
        throw std::logic_error("MiniInfer weights must be uploaded before forward");
    }
    const TensorShape expected_hidden = hidden_shape(config_);
    validate_fp32_view(input, expected_hidden, "MiniInfer input");
    validate_fp32_view(output, expected_hidden, "MiniInfer output");
    if (backend == GemmBackend::cublas && cublas_handle == nullptr) {
        throw std::invalid_argument("MiniInfer cuBLAS backend requires a valid handle");
    }
    if (backend != GemmBackend::custom && backend != GemmBackend::cublas) {
        throw std::invalid_argument("unknown MiniInfer GEMM backend");
    }

    auto& activation = workspace_.activations();
    const std::size_t tokens = miniinfer_token_count(config_);
    const std::size_t hidden_elements = miniinfer_hidden_element_count(config_);
    const std::size_t intermediate_elements = miniinfer_intermediate_element_count(config_);
    const GemmDispatch dispatch{backend, backend == GemmBackend::custom
                                             ? GemmVariant::register_blocked_fp32
                                             : GemmVariant::naive_fp32};
    const auto observe = [&](const std::string_view name, const TensorView& view) {
        if (observer) {
            observer(name, view, stream);
        }
    };

    const auto& attention_norm_weight = weights_.attention_norm_view();
    rmsnorm_cuda(input.data_as<float>(), attention_norm_weight.data_as<float>(),
                 activation.input_norm.data_as<float>(), tokens, config_.hidden_size,
                 config_.rms_norm_epsilon, RmsNormVariant::block, rmsnorm_default_block_size,
                 stream);
    observe("input_norm", activation.input_norm);

    const GemmProblem projection{tokens, config_.hidden_size, config_.hidden_size};
    const auto& query_weight = weights_.query_view();
    const auto& key_weight = weights_.key_view();
    const auto& value_weight = weights_.value_view();
    gemm_cuda(activation.input_norm, query_weight, activation.query, projection, dispatch,
              cublas_handle, stream);
    observe("query", activation.query);
    gemm_cuda(activation.input_norm, key_weight, activation.key, projection, dispatch,
              cublas_handle, stream);
    observe("key", activation.key);
    gemm_cuda(activation.input_norm, value_weight, activation.value, projection, dispatch,
              cublas_handle, stream);
    observe("value", activation.value);

    const RopeProblem rope_problem{config_.batch,           config_.sequence,
                                   config_.attention_heads, config_.head_dimension,
                                   config_.position_offset, config_.rope_base};
    rope_cuda(activation.query.data_as<float>(), activation.query_rope.data_as<float>(),
              rope_problem, rope_default_block_size, stream);
    observe("query_rope", activation.query_rope);
    rope_cuda(activation.key.data_as<float>(), activation.key_rope.data_as<float>(), rope_problem,
              rope_default_block_size, stream);
    observe("key_rope", activation.key_rope);

    const AttentionProblem attention_problem{config_.batch, config_.sequence,
                                             config_.attention_heads, config_.head_dimension};
    attention_scores_cuda(activation.query_rope.data_as<float>(),
                          activation.key_rope.data_as<float>(),
                          activation.attention_scores.data_as<float>(), attention_problem,
                          attention_default_block_size, stream);
    observe("attention_scores", activation.attention_scores);
    causal_mask_cuda(
        activation.attention_scores.data_as<float>(), activation.masked_scores.data_as<float>(),
        {config_.batch, config_.attention_heads, config_.sequence, config_.sequence, 0U},
        causal_mask_default_block_size, stream);
    observe("masked_scores", activation.masked_scores);
    softmax_cuda(activation.masked_scores.data_as<float>(),
                 activation.attention_probabilities.data_as<float>(),
                 config_.batch * config_.attention_heads * config_.sequence, config_.sequence,
                 SoftmaxVariant::block, softmax_default_block_size, stream);
    observe("attention_probabilities", activation.attention_probabilities);
    attention_value_cuda(activation.attention_probabilities.data_as<float>(),
                         activation.value.data_as<float>(),
                         activation.attention_context.data_as<float>(), attention_problem,
                         attention_default_block_size, stream);
    observe("attention_context", activation.attention_context);

    const auto& output_weight = weights_.output_view();
    gemm_cuda(activation.attention_context, output_weight, activation.attention_output, projection,
              dispatch, cublas_handle, stream);
    observe("attention_output", activation.attention_output);
    add_cuda(input.data_as<float>(), activation.attention_output.data_as<float>(),
             activation.attention_residual.data_as<float>(), hidden_elements,
             elementwise_default_block_size, stream);
    observe("attention_residual", activation.attention_residual);

    const auto& ffn_norm_weight = weights_.ffn_norm_view();
    rmsnorm_cuda(activation.attention_residual.data_as<float>(), ffn_norm_weight.data_as<float>(),
                 activation.post_attention_norm.data_as<float>(), tokens, config_.hidden_size,
                 config_.rms_norm_epsilon, RmsNormVariant::block, rmsnorm_default_block_size,
                 stream);
    observe("post_attention_norm", activation.post_attention_norm);

    const GemmProblem expand{tokens, config_.intermediate_size, config_.hidden_size};
    const auto& gate_weight = weights_.gate_view();
    const auto& up_weight = weights_.up_view();
    gemm_cuda(activation.post_attention_norm, gate_weight, activation.mlp_gate, expand, dispatch,
              cublas_handle, stream);
    observe("mlp_gate", activation.mlp_gate);
    gemm_cuda(activation.post_attention_norm, up_weight, activation.mlp_up, expand, dispatch,
              cublas_handle, stream);
    observe("mlp_up", activation.mlp_up);
    swiglu_fused_cuda(activation.mlp_gate.data_as<float>(), activation.mlp_up.data_as<float>(),
                      activation.mlp_swiglu.data_as<float>(), intermediate_elements,
                      elementwise_default_block_size, stream);
    observe("mlp_swiglu", activation.mlp_swiglu);

    const GemmProblem contract{tokens, config_.hidden_size, config_.intermediate_size};
    const auto& down_weight = weights_.down_view();
    gemm_cuda(activation.mlp_swiglu, down_weight, activation.mlp_output, contract, dispatch,
              cublas_handle, stream);
    observe("mlp_output", activation.mlp_output);
    add_cuda(activation.attention_residual.data_as<float>(), activation.mlp_output.data_as<float>(),
             output.data_as<float>(), hidden_elements, elementwise_default_block_size, stream);
    observe("output", output);
}

std::vector<std::string_view> miniinfer_intermediate_names() {
    return {"input_norm",
            "query",
            "key",
            "value",
            "query_rope",
            "key_rope",
            "attention_scores",
            "masked_scores",
            "attention_probabilities",
            "attention_context",
            "attention_output",
            "attention_residual",
            "post_attention_norm",
            "mlp_gate",
            "mlp_up",
            "mlp_swiglu",
            "mlp_output",
            "output"};
}

Tolerance miniinfer_intermediate_tolerance(const std::string_view name,
                                           const MiniInferConfig& config) noexcept {
    if (name == "input_norm" || name == "query" || name == "key" || name == "value") {
        return {2.0e-4, 2.0e-4};
    }
    if (name == "query_rope" || name == "key_rope") {
        return {3.0e-4, 2.0e-4};
    }
    if (name == "attention_scores" || name == "masked_scores") {
        return {5.0e-4, 5.0e-4};
    }
    if (name == "attention_probabilities" || name == "attention_context") {
        return {5.0e-4, 5.0e-4};
    }
    const double accumulation_scale =
        std::max(1.0, std::ceil(std::log2(static_cast<double>(
                          std::max(config.hidden_size, config.intermediate_size)))));
    if (name == "output" || name == "mlp_output" || name == "mlp_swiglu") {
        return {5.0e-4 * accumulation_scale, 1.0e-3};
    }
    return {2.5e-4 * accumulation_scale, 5.0e-4};
}

} // namespace warpforge
