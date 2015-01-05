#
# Copyright 2015 Stanford University and NVIDIA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

FERMI_INPUTS	:= diff_fermi.cu \
		   visc_fermi.cu \
		   fast_fermi.cu

KEPLER_INPUTS	:= diff_kepler.cu \
		   visc_kepler.cu \
		   fast_kepler.cu

FERMI_OUTPUTS 	:= $(FERMI_INPUTS:.cu=.ptx)
KEPLER_OUTPUTS	:= $(KEPLER_INPUTS:.cu=.ptx)

SMALLF_OUTPUTS  := $(FERMI_INPUTS:.cu=_small.ptx)
SMALLK_OUTPUTS	:= $(KEPLER_INPUTS:.cu=_small.ptx)
DYNAMICF_OUTPUTS:= $(FERMI_INPUTS:.cu=_dynamic.ptx)
DYNAMICK_OUTPUTS:= $(KEPLER_INPUTS:.cu=_dynamic.ptx)

%_fermi_small.ptx : %_fermi.cu
	nvcc -o $@ -ptx -m64 -arch=compute_20 -DTOTAL_STEPS=4 $<

%_fermi_dynamic.ptx : %_fermi.cu
	nvcc -o $@ -ptx -m64 -arch=compute_20 -DDYNAMIC $<

%_fermi.ptx : %_fermi.cu
	nvcc -o $@ -ptx -m64 -arch=compute_20 $<

%_kepler_small.ptx : %_kepler.cu
	nvcc -o $@ -ptx -m64 -arch=compute_35 -DTOTAL_STEPS=4 $<

%_kepler_dynamic.ptx : %_kepler.cu
	nvcc -o $@ -ptx -m64 -arch=compute_35 -DDYNAMIC $<

%_kepler.ptx : %_kepler.cu
	nvcc -o $@ -ptx -m64 -arch=compute_35 $<


.PHONY: all
all: normal

normal: $(FERMI_OUTPUTS) $(KEPLER_OUTPUTS)

.PHONY: small
small: $(SMALLF_OUTPUTS) $(SMALLK_OUTPUTS)

.PHONY: dynamic
dynamic: $(DYNAMICF_OUTPUTS) $(DYNAMICK_OUTPUTS)
	
clean:
	rm -f *.ptx

