# HLS Support Library
C++ header library for Vivado/Vitis HLS providing implementations of common
building blocks for the assembly of structured hardware-accelerated kernels.

## [Bit Utils](src/bit_utils.hpp)
- Bit-level operators for `ap_(u)int<N>`.

## [Streaming](src/streaming.hpp)
- `read()`-guarded, stream-to-stream processing templates that are customized by
  plugin lambdas.
- Stateful adapters are encapsulated as classes. They are *flit*-based.
  i.e. they are reacting on an asserted `last` flag accompanying the payload data.

## [Hashes](src/hashes.hpp)
- A collection of various hashing functors currently operating on 32-bit input data.

# Similar Project
- [Networking Template Library](https://github.com/acsl-technion/ntl)
- [hlslib](https://github.com/definelicht/hlslib)
