#include <ATen/ATen.h>

#include <benchmark/benchmark.h>

static void quantize_per_channel(benchmark::State& state) {
  const size_t batchSize = 128;
  const size_t channels = static_cast<size_t>(state.range(0));
  const size_t nelem = static_cast<size_t>(state.range(1));

  at::Tensor a = at::rand({batchSize, channels, nelem});
  at::Tensor scale = at::rand({channels});
  at::Tensor zero_point = at::randint(-10, 10, {channels});

  at::Tensor qa;
  for (auto _ : state) {
    qa = at::native::quantize_per_channel_cpu(
        a, scale, zero_point, 1, at::ScalarType::QUInt8);
  }
}

static void GenerateSizes(benchmark::internal::Benchmark* b) {
  b->ArgNames({"C", "N"});

  for (size_t c = 8; c < 1024; c *= 2) {
    for (size_t n = 8; n < 1024; n *= 2) {
      b->Args({c, n});
    }
  }
}

BENCHMARK(quantize_per_channel)->Apply(GenerateSizes);
BENCHMARK_MAIN();
