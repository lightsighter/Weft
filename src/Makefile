
OUTFILE := weft 

.PHONY: all
all: $(OUTFILE)

GCC = g++
CC_FLAGS = -ggdb -Wall
LD_FLAGS = -ggdb

FILES = weft.cc \
	program.cc \
	instruction.cc

OBJS := $(FILES:.cc=.o)

%.o : %.cc
	$(GCC) -c $(CC_FLAGS) $<

$(OUTFILE) : $(OBJS)
	$(GCC) -o $(OUTFILE) $(OBJS) $(LD_FLAGS)

clean:
	rm -f *.o $(OUTFILE)
