/****************************************************************************
 * Copyright (c) 2020, Systems Group, ETH Zurich
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ****************************************************************************
 * @brief
 *	Structural stream manipulation modules.
 * @author
 *	Thomas B. Preußer <tpreusser@inf.ethz.ch> <thomas.preusser@utexas.edu>
 */
#ifndef HLS_STREAMING_HPP
#define HLS_STREAMING_HPP

#include <hls_stream.h>
#include <ap_int.h>

#include <utility>

#include "bit_utils.hpp"

namespace hsl {

//===========================================================================
// Flit types with `last` markers to communicate job boundaries

// Basic Flit: value + last flag
template<typename T_VAL>
struct flit_v_t {
	bool	last;
	T_VAL	val;

public:
	flit_v_t(int _val = 0) : last(false), val(_val) {}
	flit_v_t(bool _last, T_VAL const& _val) : last(_last), val(_val) {}
};

// Key/Value Flit: key, value + last flag
template<typename T_KEY, typename T_VAL>
struct flit_kv_t {
	bool	last;
	T_KEY	key;
	T_VAL	val;

public:
	flit_kv_t(int _val = 0) : last(false), key(0), val(_val) {}
	flit_kv_t(bool _last, T_KEY const &_key, T_VAL const& _val)
	: last(_last), key(_key), val(_val) {}
};

//===========================================================================
// Elementary Transform Functors

/**
 * Const<int> f(1);
 * f(arg0, arg1, ...) returns 1
 */
template<typename T>
class Const {
	T const  m_val;
public:
	Const(T const& val) : m_val(val) {}
public:
	template<typename... Args>
	constexpr T const& operator()(Args&&...) const noexcept { return  m_val; }
};

/**
 * Arg<i>  f;
 * f(arg_0, arg_1, ...) returns arg_i
 * Maintains references except for rvalue refs, which are reduced to prvalues.
 */
template<unsigned i>
struct Arg {
	template<typename A, typename... Args>
	constexpr auto operator()(A&&, Args&&... args) const noexcept
	 -> decltype(Arg<i-1>()(std::forward<Args>(args)...)) {
		return  Arg<i-1>()(std::forward<Args>(args)...);
	}
};
template<>
struct Arg<0> {
	template<typename T> struct remove_rvref      { using type = T; };
	template<typename T> struct remove_rvref<T&&> { using type = T; };
	template<typename A, typename... Args>
	constexpr auto operator()(A&& a, Args&&... args) const noexcept
	 -> typename remove_rvref<A>::type {
		return  a;
	}
};

//===========================================================================
// Stateless Transforms

/**
 * Apply functional mapping `f` while passing data from `src` to `dst`.
 * y <= f(x) with f(TI) -> TO
 */
template<typename F = Arg<0>, typename TI, typename TO>
void map(hls::stream<TI> &src, hls::stream<TO> &dst, F&& f = F()) {
#pragma HLS pipeline II=1
	TI  x;
	if(src.read_nb(x))  dst.write(f(x));
}

/**
 * Copy data from `src` to all elements of `dst` allowing an
 * index-dependent functional mapping `f`.
 * y_i <= f(x, i) with f(TI, int) -> TO
 */
template<typename F = Arg<0>, typename TI, typename TO, int N>
void split(hls::stream<TI> &src, hls::stream<TO> (&dst)[N], F&& f = F()) {
#pragma HLS pipeline II=1
	TI  x;
	if(src.read_nb(x)) {
		for(int  i = 0; i < N; i++) {
#pragma HLS unroll
			dst[i].write(f(x, i));
		}
	}
}

/**
 * Performs parallel reads across all `src` inputs applying a sequential
 * assimilating reduction `f` before emitting the result to `dst`.
 * f(TO, TI) -> TO
 * y <- f(...f(f(TO(), x_0), x_1), ...)
 */
template<typename F, typename TI, typename TO, int N>
void reduce(hls::stream<TI> (&src)[N], hls::stream<TO> &dst, F&& f = F(), TO zero = TO()) {
#pragma HLS pipeline II=1
	ap_uint<N>  empty;
	for(int  i = 0; i < N; i++) {
#pragma HLS unroll
		empty[i] = src[i].empty();
	}
	if(empty == 0) {
		TO  y = zero;
		for(int  i = 0; i < N; i++) {
#pragma HLS unroll
			y = f(y, src[i].read());
		}
		dst.write(y);
	}
}

//===========================================================================
// Stateful Transforms

/**
 * Perform a sequential assimilating fold over the data received from
 * `src` until receiving an item with an asserted `last` flag, which
 * triggers the output of the Fold resultand resets the folding buffer.
 * f(TO, TI) -> TO
 * y <- f(...f(f(TO(), x_0), x_1), ...)
 */
template<typename TO>
class Fold {
	TO  res;

public:
	template<typename F, typename TI>
	void fold(hls::stream<TI> &src, hls::stream<TO> &dst, F&& f = F(), TO zero = TO()) {
#pragma HLS pipeline II=1
		if(!src.empty()) {
			TI const  x = src.read();
			res = f(res, x.val);
			if(x.last) {
				dst.write(res);
				res = zero;
			}
		}
	}
}; // class Fold

/**
 * Concatenates the data received from the parallel `src` streams
 * starting at index 0 and switching to the subsequent one upon
 * encountering an asserted `last` flag. Only the `last` flag of
 * the final stream is maintained in the output written to `dst`
 * when the process wraps back to stream 0.
 */
template<int N>
class Concat {
	ap_uint<clog2<N>::value>  idx = 0;
public:
	template<typename T>
	void concat(
		hls::stream<T> (&src)[N],
		hls::stream<T> &dst
	) {
#pragma HLS pipeline II=1
		T  x;
		if(src[idx].read_nb(x)) {
			bool const  last = x.last;
			bool const  wrap = last && (idx == N-1);
			x.last = wrap;
			dst.write(x);
			if(last) {
				if(wrap)  idx = 0;
				else  idx++;
			}
		}
	}

}; // class Concat

/**
 * Interleaves the data from the `src` streams onto the `dst` stream
 * batch by batch.
 */
template<int N, int BATCH_SIZE=1>
class Interleave {
	ap_uint<clog2<N>::value>           idx = 0;
	ap_uint<clog2<BATCH_SIZE>::value>  cnt = 0;
public:
	template<typename T>
	void interleave(
		hls::stream<T> (&src)[N],
		hls::stream<T> &dst
	) {
#pragma HLS pipeline II=1
		T  x;
		if(src[idx].read_nb(x)) {
			dst.write(x);
			if(cnt < BATCH_SIZE-1)  cnt++;
			else {
				cnt = 0;
				idx = (idx == N-1)? 0 : idx+1;
			}
		}
	}

}; // class Interleave

/**
 * Collects statistics over a key/value `src` stream until encountering
 * an asserted `last` flag, which triggers the statistics output to `dst`.
 */
template<typename T_KEY, typename T_VAL>
class Collect {
	// Consuming input vs. dumping sketch
	bool   dump = false;
	T_KEY  dump_ptr = 0;

	// Sketch memory
	T_VAL  Mem[1<<T_KEY::width] = {};
	T_KEY  z_key = 0, zz_key = 0, zzz_key = 0;
	T_VAL  z_val = 0, zz_val = 0, zzz_val = 0;

public:
	template<typename F, typename T_INC>
	void collect(
		hls::stream<flit_kv_t<T_KEY, T_INC>> &src,
		hls::stream<flit_v_t<T_VAL>> &dst,
		F&& f = F()
	) {
#pragma HLS pipeline II=1
#pragma HLS dependence variable=Mem inter false
		flit_kv_t<T_KEY, T_INC>  idat;
		bool  istb;
		T_VAL  curr_val; {
			T_KEY     madr;
			if(dump)  madr = dump_ptr;
			else {
				istb = src.read_nb(idat);
				madr = idat.key;
			}
			T_VAL const  mval = Mem[madr];
			bool  match = false;
			if(madr == zzz_key)  { match = true; curr_val = zzz_val; }
			if(madr == zz_key)   { match = true; curr_val = zz_val; }
			if(madr == z_key)    { match = true; curr_val = z_val; }
			if(!match)  curr_val = mval;
		}

		T_KEY  key = z_key;
		T_VAL  val = z_val;
		if(dump) {		// Dump sketch
			bool const  last = dump_ptr == T_KEY{-1};
			dst.write({last, curr_val});

			key = dump_ptr;
			val = 0;

			dump = !last;
			dump_ptr++;
		}
		else if(istb) { // Consume input
			key = idat.key;
			val = f(curr_val, idat.val);

			dump = idat.last;
		}

		zzz_key = zz_key; zzz_val = zz_val;
		zz_key = z_key; zz_val = z_val;
		z_key = key; z_val = val;
		Mem[key] = val;
	}
}; // class Collect

} // namespace hsl
#endif
