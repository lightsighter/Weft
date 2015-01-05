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

INPUTS		:= vec_single.cu \
		   vec_double.cu \
		   vec_manual.cu \
		   both_single.cu \
		   both_double.cu \
		   both_manual.cu
OUTPUTS 	:= $(INPUTS:.cu=.ptx)
SMALL_OUTPUTS	:= $(INPUTS:.cu=_small.ptx)

%_small.ptx : %.cu
	nvcc -o $@ -ptx -m64 -arch=compute_20 -DSGEMV_ITERS=8 $<

%.ptx : %.cu
	nvcc -o $@ -ptx -m64 -arch=compute_20 $<

.PHONY: all
all: normal

normal: $(OUTPUTS)

.PHONY: small
small: $(SMALL_OUTPUTS)
	
clean:
	rm -f *.ptx

