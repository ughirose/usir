# Makefile for USIR WebAssembly SIMD 128-bit Acceleration Kernel
# Emscripten SIMD128 target

CXX = emcc
CXXFLAGS = -O3 -std=c++17 -msimd128 -flto --no-entry \
	-s STANDALONE_WASM \
	-s INITIAL_MEMORY=65536 \
	-s MAXIMUM_MEMORY=65536 \
	-s ALLOW_MEMORY_GROWTH=0 \
	-s EXPORTED_FUNCTIONS="['_usir_cholesky_sq_dist_24d','_usir_hamiltonian_24d','_usir_poe_step_24d','_malloc','_free']" \
	-s ERROR_ON_UNDEFINED_SYMBOLS=0

TARGET = usir_simd.wasm

.PHONY: all clean

all: $(TARGET)

$(TARGET): usir_simd.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
