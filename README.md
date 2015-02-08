Weft
====

A Sound and Complete Verification Tool for Warp-Specialized GPU Kernels

Update! Our paper on Weft, **Verification of Producer-Consumer
Synchronization in GPU Programs** will be appearing at
[PLDI 2015](http://conf.researchr.org/home/pldi2015).

Navigation
----

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Downloading Weft](#downloading-and-building-weft)
4. [Using Weft](#using-weft)
5. [Command Line Arguments](#command-line-arguments)

Overview
----

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
can optionally be reported. In practice, we have found that for most 
GPU kernels this is not an issue because synchronization and shared 
memory accesses are not dependent on program input and therefore
can be verified statically.

Due to its generality, Weft is also capable of checking non-warp-specialized
code as well for race freedom. The one caveat is that Weft currently
does not attempt to check code that uses atomics.

Prerequisites
----

Weft requires an installation of the CUDA compiler for generating
input PTX files. The CUDA toolkit can be downloaded 
[here](https://developer.nvidia.com/cuda-downloads). Weft requires
CUDA version 5.0 or later.

Weft can be built with a standard C++ compiler. Weft has been tested
with g++ and clang on both Linux and Mac systems.

Downloading and Building Weft
----

Weft is available on github under the Apache Software License
version 2.0. To clone a copy of the Weft source type:

    $ git clone https://github.com/lightsighter/Weft.git

After cloning the repository, change into the `src` directory
and type:

    $ make

This will build the Weft binary `weft`. You may wish to add the 
directory containing the Weft binary to your path using the
following command.

    $ export PATH=$PATH:/<path_to_weft>/src

Using Weft
----

Using Weft to validate a CUDA source file is straightforward.
The first step is to use the CUDA compiler to generate a PTX
file for Weft to consume as input. Currently, Weft will only
analyze the first kernel that it finds in a PTX file, so files
containing multiple kernels should be divided into separate
source files.

To generate input for Weft, the CUDA compiler should be
invoked with the `-ptx` flag to create an output PTX file.
We also recommend the CUDA compiler be called with the 
`-lineinfo` flag so Weft can provide output based on CUDA 
source code line numbers instead of PTX line numbers. In 
some cases, the flags for compute architecture (`-arch`) and 
machine size (`-m`) may need to be specified depending on the
kernel being compiled. Below are the two ways that we invoke 
the CUDA compiler on all of our example kernels for the
Fermi and Kepler architectures respectively.

    $ nvcc -ptx -lineinfo -m64 -arch=compute_20 source.cu
    $ nvcc -ptx -lineinfo -m64 -arch=compute_35 source.cu

The resulting PTX file is the input to Weft. The PTX file name
can either be specified to Weft using the `-f` flag or as the
last argument.

    $ weft -f source.ptx -s -t 4
    $ weft source.ptx

As part of its validation, Weft needs to know how many threads
are in each CTA. For kernels with 1-D CTAs, Weft can infer this
information if the `__launch_bounds__` annotation was given on
the CUDA original kernel. However, if this declaration did not exits on
the original source kernel, then it must be explicitly specified
using the `-n` flag. As an example, our `saxpy_single.cu` source
file contains no `__launch_bounds__` declaration on its
kernel, therefore we must tell Weft that the kernel requires CTAs
containing 320 threads.

    $ weft -n 320 saxpy_single.ptx

Note that the `-n` flag should also be used to specify multi-dimensional
CTA shapes which cannot be captured by the `__launch_bounds__` 
annotation. Both of the following are valid examples:

    $ weft -n 320x1x1 saxpy_single.ptx
    $ weft -n 16x16 dgemm.ptx

Weft supports a large set of command line flags which we cover in
more detail [later](#command-line-arguments). We mention two flags
briefly now as they are often useful for many users. First, by default,
Weft does not assume <em>warp synchronous</em> execution where all
threads in a warp execute in lock-step. Many CUDA programs rely on 
this property for correctness. The warp synchronous execution assumption
can be enabled in Weft by passing the `-s` flag on the command line.
As an example, the Fermi chemistry kernel in `examples/DME/chem_fermi.cu`
will report races if run under normal assumptions, but will always be 
race free under a warp synchronous execution.

Another useful flag for Weft is the `-t` flag which controls the 
number of parallel threads that Weft will use when performing validation.
For most multi-core architectures we find that 2-4 threads is a good
option. Weft is primarily a memory bound application, and having two
threads per socket is usually sufficient to saturate memory bandwidth.

We have provided a set of test kernels for Weft in the `examples` 
directory. Each individual directory contains its own Makefile for
generating the PTX code for individual kernels. We also have a script 
called `run_examples.sh` in the main `examples` directory which will 
validate all of the example kernels. Note that some kernels will 
report races. The script may take between 30 minutes
and 1 hour (depending on the machine) to validate all of the kernels.

Command Line Arguments
----

Below is a summary of the command line flags that Weft supports.

 * `-b`: specify the CTA id to simulate (default 0x0x0)
 * `-d`: print detailed information when giving error output,
                including where threads are blocked for deadlock as
                well as per-thread and per-address information for races
 * `-f`: specify the input PTX file (can be omitted if 
                the file is the last argument in the command line)
 * `-g`: specify the grid dimensions for the kernel being simulated
                (this argument can be omitted in most cases as many kernels
                will not depend on these values)
 * `-i`: instrument the execution of Weft to report the
                time taken and memory usage for each stage
 * `-n`: set the number of threads per CTA. This is required
                if the CUDA kernel did not have a 
                `__launch_bounds__` annotation
 * `-s`: assume warp-synchronous execution when checking for races
 * `-t`: set the size of the thread pool for Weft to use; in
                general, Weft is memory bound, so one or two threads per socket
                should be sufficient for achieving peak performance.
 * `-v`: enable verbose output
 * `-w`: enable warnings about PTX instructions that cannot be
                statically emulated (can result in large output)

