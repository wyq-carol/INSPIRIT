PROG=exemple
CUDA_SDK_ROOT=/usr/local/cuda/sdk/C
CUDA_HOME=/usr/local/cuda

.PHONY: all

all: $(PROG)


CC      := gcc
NVCC    := /usr/local/cuda/bin/nvcc
CFLAGS  := $$(pkg-config --cflags libstarpu) -O0 -g #-Wall
LDFLAGS := $$(pkg-config --libs libstarpu)
CUDADIR=$(CUDA_HOME)

COMMONFLAGS += -I. -I$(CUDADIR)/include  -I$(CUDA_SDK_ROOT)/common/inc -DUNIX -g -Xcompiler
NVCCFLAGS +=  -I$(CUDA_SDK_ROOT)/common/inc -I. -G

%.o: %.cu
	$(NVCC) $(CFLAGS) $(COMMONFLAGS) $(NVCCFLAGS) -o $@ -c $< 

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $< 

$(PROG): %: %.o %_kernel.o
	$(CC) -o $@ $^ $(LDFLAGS) -L$(CUDADIR)/lib -lcudart

clean:
	rm -f $(PROG) *.o
