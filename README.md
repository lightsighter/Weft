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

Weft performs a fully static analysis which requires that the use of 
named barriers and shared memory accesses be statically analyzable.
All operations which are not statically analyzable are ignored and 
can optionally be reported. In practice, we have found that For most 
GPU kernels this is not an issue because synchronization and shared 
memory accesses are not dependent on program input and therefore
can be verified statically.

Due to its generality, Weft is also capable of checking non-warp-specialized
code as well for race freedom. The one caveat is that Weft currently
does not attempt to check code that uses atomics.

To use Weft, request that the CUDA compiler generate PTX for a kernel
using the <em>--ptx</em> flag. The output PTX file can then be passed
directly to Weft. Weft currently assumes that there is only one kernel 
per file.

The examples directory contains several CUDA kernels and Makefiles for
generating the associated PTX code.

Below is a summary of the command line flags that Weft accepts.

 * <em>-f</em>: specify the input PTX file (can be omitted if 
                the file is the last argument in the command line)
 * <em>-i</em>: instrument the execution of Weft to report the
                time taken and memory usage for each stage
 * <em>-n</em>: set the number of threads per CTA. This is required
                if the CUDA kernel did not have a 
                <em>\_\_launch_bounds\_\_</em> annotation
 * <em>-s</em>: assume warp-synchronous exeuction when checking for races
 * <em>-t</em>: set the size of the thread pool for Weft to use. In
                general, Weft is memory bound, so one thread per socket
                should be sufficient for achieving peak performance.
 * <em>-v</em>: enable verbose output
 * <em>-w</em>: enable warnings about PTX instructions that cannot be
                statically emulated (can result in large output)

