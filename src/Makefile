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

OUTFILE := weft 

.PHONY: all
all: $(OUTFILE)

GCC = g++
CC_FLAGS = -O2 -Wall
LD_FLAGS = -O2 -lpthread 

UNAME = $(shell uname)
ifeq ($(UNAME),Linux)
LD_FLAGS += -lrt
endif

FILES = weft.cc \
	race.cc \
	graph.cc \
	program.cc \
	instruction.cc

OBJS := $(FILES:.cc=.o)

%.o : %.cc
	$(GCC) -c $(CC_FLAGS) $<

$(OUTFILE) : $(OBJS)
	$(GCC) -o $(OUTFILE) $(OBJS) $(LD_FLAGS)

clean:
	rm -f *.o $(OUTFILE)
