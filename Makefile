
SDIR = ./src
IDIR = ./include
BDIR = ./bin
TDIR = ./test
CUDADIR = /usr/local/cuda/samples/common/inc/
CC = nvcc

DCIFLAGS = -use_fast_math -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES --std=c++11 -O2 -I $(IDIR)
DCITESTFLAGS = -use_fast_math -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES --std=c++11 -g -G -O2 -I $(IDIR)
QUERYFLAGS = -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES --std=c++11 -I $(CUDADIR) -I $(IDIR)
KMPSOFLAGS = -use_fast_math -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES --std=c++11 -O2 -I $(IDIR)

_CLUSTDESCRDEPS = cluster_descriptor.h
CLUSTDESCRDEPS = $(patsubst %,$(IDIR)/%,$(_CLUSTDESCRDEPS))
_CLUSTDESCRSRC = cluster_descriptor.cu
CLUSTDESCRSRC = $(patsubst %,$(SDIR)/%,$(_CLUSTDESCRSRC))
CLUSTDESCROBJ = cluster_descriptor.o

_CLUSTUTILSDEPS = cluster_utils.h
CLUSTUTILSDEPS = $(patsubst %,$(IDIR)/%,$(_CLUSTUTILSDEPS))
_CLUSTUTILSSRC = cluster_utils.cu
CLUSTUTILSSRC = $(patsubst %,$(SDIR)/%,$(_CLUSTUTILSSRC))
CLUSTUTILSOBJ = cluster_utils.o

_DCIDEPS = dci.h
DCIDEPS = $(patsubst %,$(IDIR)/%,$(_DCIDEPS))
_DCISRC = dci.cu
DCISRC = $(patsubst %,$(SDIR)/%,$(_DCISRC))
DCIOBJ = dci.o

_DCITESTSRC = dci_test.cu
DCITESTSRC = $(patsubst %,$(TDIR)/%,$(_DCITESTSRC))
DCITESTOBJ = dci_test.o

_QUERYDEPS = helper_cuda.h
QUERYDEPS = $(patsubst %,$(CUDADIR)/%,$(_QUERYDEPS))
_QUERYSRC = device_query.cpp
QUERYSRC = $(patsubst %,$(TDIR)/%,$(_QUERYSRC))
QUERYOBJ = device_query.o

_KMPSOSRC = kmpso.cu
KMPSOSRC = $(patsubst %,$(SDIR)/%,$(_KMPSOSRC))
KMPSOOBJ = kmpso.o

$(CLUSTDESCROBJ): $(CLUSTDESCRSRC) $(CLUSTDESCRDEPS)
	$(CC) -c -o $@ $< $(DCIFLAGS)

$(CLUSTUTILSOBJ): $(CLUSTUTILSSRC) $(CLUSTUTILSDEPS)
	$(CC) -c -o $@ $< $(DCIFLAGS)

$(DCIOBJ): $(DCISRC) $(DCIDEPS)
	$(CC) -c -o $@ $< $(DCIFLAGS)

dci: $(CLUSTDESCROBJ) $(CLUSTUTILSOBJ) $(DCIOBJ)
	$(CC) -o $@ $^ $(DCIFLAGS)
	mv $@ $(BDIR)
	rm $(DCIOBJ)
	rm $(CLUSTDESCROBJ)
	rm $(CLUSTUTILSOBJ)

$(DCITESTOBJ): $(DCITESTSRC) $(DCIDEPS)
	$(CC) -c -o $@ $< $(DCITESTFLAGS)

dcitest: $(DCITESTOBJ)
	$(CC) -o $@ $^ $(DCITESTFLAGS)
	mv $@ $(BDIR)
	rm $(DCITESTOBJ)

$(QUERYOBJ): $(QUERYSRC) $(QUERYDEPS)
	$(CC) -c -o $@ $< $(QUERYFLAGS)

query: $(QUERYOBJ)
	$(CC) -o $@ $^ $(QUERYFLAGS)
	mv $@ $(BDIR)
	rm $(QUERYOBJ)

$(KMPSOOBJ): $(KMPSOSRC) $(DCIDEPS)
	$(CC) -c -o $@ $< $(KMPSOFLAGS)

kmpso: $(CLUSTDESCROBJ) $(CLUSTUTILSOBJ) $(KMPSOOBJ)
	$(CC) -o $@ $^ $(KMPSOFLAGS)
	mv $@ $(BDIR)
	rm $(KMPSOOBJ)
