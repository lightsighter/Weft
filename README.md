Weft
====

A Sound and Complete Verification Tool for Warp-Specialized GPU Kernels

Weft is a sound and complete verification tool for warp-specialized 
kernels that use named barriers on NVIDIA GPUs. Warp-specialized 
kernels can encode arbitrary producer-consumer relationships between 
different subsets of warps within a kernel using named barriers.
This requires a more general analysis than most current GPU verification
tools provide.

Weft operates on the PTX code emitted by the CUDA compiler and verifies 
three important properties of any warp-specialized kernel.

 * Deadlock Freedom - the use of named barriers should not result in deadlock.
 * Safe Barrier Recycling - named barriers are a limited physical resource
                            and it is important to check that they are
                            safely recycled.
 * Race Freedom - checking that all shared memory accesses are properly
                  synchronized by named barriers.

Due to its generality, Weft is also capable of checking non-warp-specialized
code as well for race freedom. The one caveat is that Weft currently
does not attempt to check code that uses atomics.

To use Weft, request that the CUDA compiler generate PTX for a kernel
using the <em>--ptx</em> flag. The output PTX file can then be passed
directly to Weft. Weft assumes that there is only one kernel per file.
If Weft discovers any loops which it cannot statically bound, then it
unrolls them four times. The expected number of unroll times can be
set using the <em>--loop</em> flag.
