#include <warpforge/miniinfer.cuh>

#include <cstddef>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace warpforge {
namespace {

[[nodiscard]] std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open MiniInfer manifest: " + path.string());
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

[[nodiscard]] std::string escape_regex(const std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size() * 2U);
    for (const char character : value) {
        if (std::string_view{".^$|()[]{}*+?\\"}.find(character) != std::string_view::npos) {
            escaped.push_back('\\');
        }
        escaped.push_back(character);
    }
    return escaped;
}

[[nodiscard]] std::string extract_token(const std::string& json, const std::string_view key,
                                        const std::string_view value_pattern) {
    const std::regex pattern{"\\\"" + escape_regex(key) + "\\\"\\s*:\\s*(" +
                             std::string(value_pattern) + ")"};
    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        throw std::runtime_error("MiniInfer manifest is missing key: " + std::string(key));
    }
    return match[1].str();
}

[[nodiscard]] std::size_t extract_size(const std::string& json, const std::string_view key) {
    return static_cast<std::size_t>(std::stoull(extract_token(json, key, R"([0-9]+)")));
}

[[nodiscard]] float extract_float(const std::string& json, const std::string_view key) {
    return std::stof(extract_token(json, key, R"([-+0-9.eE]+)"));
}

[[nodiscard]] std::string extract_string(const std::string& json, const std::string_view key) {
    const std::regex pattern{"\\\"" + escape_regex(key) + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\""};
    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        throw std::runtime_error("MiniInfer manifest is missing key: " + std::string(key));
    }
    return match[1].str();
}

[[nodiscard]] std::vector<float> read_fp32(const std::filesystem::path& directory,
                                           const std::string_view name,
                                           const std::size_t element_count) {
    const std::filesystem::path path = directory / (std::string(name) + ".bin");
    const std::uintmax_t expected_bytes =
        static_cast<std::uintmax_t>(element_count) * sizeof(float);
    std::error_code error;
    const std::uintmax_t actual_bytes = std::filesystem::file_size(path, error);
    if (error || actual_bytes != expected_bytes) {
        throw std::runtime_error("MiniInfer fixture size mismatch for " + path.string() +
                                 ": expected " + std::to_string(expected_bytes) + " bytes");
    }
    std::vector<float> values(element_count);
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(expected_bytes));
    if (!input) {
        throw std::runtime_error("failed to read MiniInfer fixture tensor: " + path.string());
    }
    return values;
}

} // namespace

MiniInferFixture load_miniinfer_fixture(const std::filesystem::path& directory) {
    const std::string manifest = read_text(directory / "manifest.json");
    if (extract_size(manifest, "schema_version") != 1U) {
        throw std::runtime_error("unsupported MiniInfer fixture schema version");
    }

    MiniInferFixture fixture;
    fixture.preset = extract_string(manifest, "preset");
    fixture.seed = extract_size(manifest, "seed");
    fixture.config.batch = extract_size(manifest, "batch");
    fixture.config.sequence = extract_size(manifest, "sequence");
    fixture.config.hidden_size = extract_size(manifest, "hidden_size");
    fixture.config.attention_heads = extract_size(manifest, "attention_heads");
    fixture.config.head_dimension = extract_size(manifest, "head_dimension");
    fixture.config.intermediate_size = extract_size(manifest, "intermediate_size");
    fixture.config.rms_norm_epsilon = extract_float(manifest, "rms_norm_epsilon");
    fixture.config.rope_base = extract_float(manifest, "rope_base");
    fixture.config.position_offset = extract_size(manifest, "position_offset");
    validate_miniinfer_config(fixture.config);

    const std::size_t hidden = fixture.config.hidden_size;
    const std::size_t intermediate = fixture.config.intermediate_size;
    const std::size_t hidden_square = TensorShape{hidden, hidden}.element_count();
    const std::size_t hidden_intermediate = TensorShape{hidden, intermediate}.element_count();
    fixture.weights.attention_norm_weight = read_fp32(directory, "attention_norm_weight", hidden);
    fixture.weights.query_weight = read_fp32(directory, "query_weight", hidden_square);
    fixture.weights.key_weight = read_fp32(directory, "key_weight", hidden_square);
    fixture.weights.value_weight = read_fp32(directory, "value_weight", hidden_square);
    fixture.weights.output_weight = read_fp32(directory, "output_weight", hidden_square);
    fixture.weights.ffn_norm_weight = read_fp32(directory, "ffn_norm_weight", hidden);
    fixture.weights.gate_weight = read_fp32(directory, "gate_weight", hidden_intermediate);
    fixture.weights.up_weight = read_fp32(directory, "up_weight", hidden_intermediate);
    fixture.weights.down_weight = read_fp32(directory, "down_weight", hidden_intermediate);
    fixture.input = read_fp32(directory, "input", miniinfer_hidden_element_count(fixture.config));

    const std::size_t hidden_elements = miniinfer_hidden_element_count(fixture.config);
    const std::size_t intermediate_elements = miniinfer_intermediate_element_count(fixture.config);
    const std::size_t score_elements = miniinfer_score_element_count(fixture.config);
    for (const std::string_view name : miniinfer_intermediate_names()) {
        std::size_t element_count = hidden_elements;
        if (name == "attention_scores" || name == "masked_scores" ||
            name == "attention_probabilities") {
            element_count = score_elements;
        } else if (name == "mlp_gate" || name == "mlp_up" || name == "mlp_swiglu") {
            element_count = intermediate_elements;
        }
        fixture.intermediates.emplace(std::string(name), read_fp32(directory, name, element_count));
    }
    return fixture;
}

} // namespace warpforge
