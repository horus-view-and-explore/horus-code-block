#ifndef TRANSFORMS_HPP
#define TRANSFORMS_HPP

#include <array>
#include <string>

template <size_t N>
static std::array<size_t, N> inverse_permutation(const std::array<size_t, N> &permutation)
{
    std::array<size_t, N> inverse_permutation;

    for (size_t i = 0; i < N; ++i)
    {
        inverse_permutation[permutation[i]] = i;
    }

    return inverse_permutation;
}

template <typename T, size_t N>
static std::array<T, N>
transpose_elements(const std::array<T, N> &elements, const std::array<size_t, N> &permutation)
{
    std::array<T, N> transposed_elements;

    for (T i = 0; i < N; ++i)
    {
        transposed_elements[i] = elements[permutation[i]];
    }

    return transposed_elements;
}

template <typename T, size_t N>
static std::array<T, N>
linear_to_multi_indices(const T &index, const std::array<size_t, N> &dimensions)
{
    std::array<T, N> indices;
    size_t remainder = index;

    for (int i = N - 1; i >= 0; --i)
    {
        indices[i] = static_cast<T>(remainder % dimensions[i]);
        remainder /= dimensions[i];
    }

    return indices;
}

template <typename T, size_t N>
static T
multi_to_linear_index(const std::array<T, N> &indices, const std::array<size_t, N> &dimensions)
{
    T index = 0;
    size_t multiplier = 1;

    for (int i = N - 1; i >= 0; --i)
    {
        index += static_cast<T>(indices[i] * multiplier);
        multiplier *= dimensions[i];
    }

    return index;
}

template <typename T, size_t N>
static std::vector<T> generate_transpose_mapping(
    const size_t &width,
    const size_t &height,
    const std::array<size_t, N> &dimensions,
    const std::array<size_t, N> &permutation)
{
    const auto transposed_dimensions = transpose_elements(dimensions, permutation);
    const auto inversed_permutation = inverse_permutation(permutation);

    std::vector<T> transposed_mapping(width * height);

    for (T x = 0; x < width; ++x)
    {
        for (T y = 0; y < height; ++y)
        {
            T index = static_cast<T>(y * width + x);
            transposed_mapping[index] = multi_to_linear_index(
                transpose_elements(
                    linear_to_multi_indices(index, transposed_dimensions), inversed_permutation),
                dimensions);
        }
    }

    return transposed_mapping;
}

#endif