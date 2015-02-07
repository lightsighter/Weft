#!/bin/sh

cd ../src
make
cd ../examples

#run saxpy
cd saxpy
make

echo "Running saxpy single..."
../../src/weft -t 4 -n 320 saxpy_single.ptx

echo "Running saxpy double..."
../../src/weft -t 4 -n 384 saxpy_double.ptx

make clean
cd ..

#run sgemv
cd sgemv
make

echo "Running sgemv vec single..."
../../src/weft -t 4 vec_single.ptx

echo "Running sgemv vec double..."
../../src/weft -t 4 vec_double.ptx

echo "Running sgemv vec manual..."
../../src/weft -t 4 vec_manual.ptx

echo "Running sgemv both single..."
../../src/weft -t 4 both_single.ptx

echo "Running sgemv both double..."
../../src/weft -t 4 both_double.ptx

echo "Running sgemv both manual..."
../../src/weft -t 4 both_manual.ptx

make clean
cd ..

#run RTM
cd RTM
make

echo "Running RTM one phase manual..."
../../src/weft -t 4 one_phase_manual_buffer.ptx

echo "Running RTM one phase single..."
../../src/weft -t 4 one_phase_single_buffer.ptx

echo "Running RTM two phase manual..."
../../src/weft -t 4 two_phase_manual_buffer.ptx

echo "Running RTM two phase quad..."
../../src/weft -t 4 two_phase_quad_buffer.ptx

echo "Running RTM two phase single..."
../../src/weft -t 4 two_phase_single_buffer.ptx

make clean
cd ..

#run DME
cd DME
make
#Fermi requires warp synchronization for verification
echo "Running DME diffusion fermi..."
../../src/weft -s -t 4 diff_fermi.ptx
echo "Running DME viscosity fermi..."
../../src/weft -s -t 4 visc_fermi.ptx
echo "Running DME chemistry fermi..."
../../src/weft -s -t 4 chem_fermi.ptx

#Kepler switches on warp synchronization when shuffles are detected
echo "Running DME diffusion kepler..."
../../src/weft -t 4 diff_kepler.ptx
echo "Running DME viscosity kepler..."
../../src/weft -t 4 visc_kepler.ptx
echo "Running DME chemistry kepler..."
../../src/weft -t 4 chem_kepler.ptx

make clean
cd ..

#run Heptane
cd Heptane
make

#Fermi requires warp synchronization for verification
echo "Running Heptane diffusion fermi..."
../../src/weft -s -t 4 diff_fermi.ptx
echo "Running Heptane viscosity fermi..."
../../src/weft -s -t 4 visc_fermi.ptx
echo "Running Heptane chemistry fermi..."
../../src/weft -s -t 4 chem_fermi.ptx

#Kepler switches on warp synchronization when shuffles are detected
echo "Running Heptane diffusion kepler..."
../../src/weft -t 4 diff_kepler.ptx
echo "Running Heptane viscosity kepler..."
../../src/weft -t 4 visc_kepler.ptx
echo "Running Heptane chemistry kepler..."
../../src/weft -t 4 chem_kepler.ptx

make clean
cd ..

#run PRF
cd PRF
make

#Fermi requires warp synchronization for verification
echo "Running PRF diffusion fermi..."
../../src/weft -s -t 4 diff_fermi.ptx
echo "Running PRF viscosity fermi..."
../../src/weft -s -t 4 visc_fermi.ptx

#Kepler switches on warp synchronization when shuffles are detected
echo "Running PRF diffusion kepler..."
../../src/weft -t 4 diff_kepler.ptx
echo "Running PRF viscosity kepler..."
../../src/weft -t 4 visc_kepler.ptx

make clean
cd ..

