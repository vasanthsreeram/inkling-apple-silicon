#!/bin/sh
# Fetch the pinned llama.cpp revision, apply the Inkling patch and build the tools into ./build.
# Needs: Xcode command-line tools, CMake, Ninja, curl. macOS on Apple Silicon.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
revision=0a9105de42d2c3e3cbc72cb1f398c743ada4f342
src="$root/build/source"
if [ ! -d "$src" ]; then
    mkdir -p "$src"
    curl -fsSL --retry 3 "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$revision" -o "$root/build/source.tar.gz"
    tar -xzf "$root/build/source.tar.gz" -C "$src" --strip-components 1
    patch -d "$src" -p1 < "$root/patches/inkling-apple-silicon.patch"
fi
cmake -S "$src" -B "$root/build/cmake" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_OPENMP=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$root/build/cmake" --target llama-server llama-bench llama-quantize -j "$(sysctl -n hw.ncpu)"
mkdir -p "$root/bin"
for t in llama-server llama-bench llama-quantize; do install -m 755 "$root/build/cmake/bin/$t" "$root/bin/$t"; done
echo "Built: $root/bin/{llama-server,llama-bench,llama-quantize}"
