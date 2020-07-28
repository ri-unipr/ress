
SDIR = ./src
IDIR = ./include
BDIR = ./bin
DDIR = ./device
CUDADIR = /usr/local/cuda/samples/common/inc/
CC = nvcc

DCIFLAGS = -use_fast_math -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES --std=c++11 -O2 -I $(IDIR)
HOMGENFLAGS = -use_fast_math -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES --std=c++11 -O2 -I $(IDIR)
QUERYFLAGS = -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES --std=c++11 -I $(CUDADIR) -I $(IDIR)
KMPSOFLAGS = -use_fast_math -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES --std=c++11 -O2 -I $(IDIR)
TESTFLAGS = -use_fast_math -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES --std=c++11 -O2 -I $(IDIR)

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

_FILEUTILSDEPS = file_utils.h
FILEUTILSDEPS = $(patsubst %,$(IDIR)/%,$(_FILEUTILSDEPS))
_FILEUTILSSRC = file_utils.cu
FILEUTILSSRC = $(patsubst %,$(SDIR)/%,$(_FILEUTILSSRC))
FILEUTILSOBJ = file_utils.o

_COMMANDLINEDEPS = command_line.h
COMMANDLINEDEPS = $(patsubst %,$(IDIR)/%,$(_COMMANDLINEDEPS))
_COMMANDLINESRC = command_line.cu
COMMANDLINESRC = $(patsubst %,$(SDIR)/%,$(_COMMANDLINESRC))
COMMANDLINEOBJ = command_line.o

_DCISRC = dci.cu
DCISRC = $(patsubst %,$(SDIR)/%,$(_DCISRC))
DCIOBJ = dci.o

_HOMGENSRC = homgen.cu
HOMGENSRC = $(patsubst %,$(SDIR)/%,$(_HOMGENSRC))
HOMGENOBJ = homgen.o

_QUERYDEPS = helper_cuda.h
QUERYDEPS = $(patsubst %,$(CUDADIR)/%,$(_QUERYDEPS))
_QUERYSRC = device_query.cpp
QUERYSRC = $(patsubst %,$(DDIR)/%,$(_QUERYSRC))
QUERYOBJ = device_query.o

_KMPSOSRC = kmpso.cu
KMPSOSRC = $(patsubst %,$(SDIR)/%,$(_KMPSOSRC))
KMPSOOBJ = kmpso.o

_TESTSRC = dci_test.cu
TESTSRC = $(patsubst %,$(SDIR)/%,$(_TESTSRC))
TESTOBJ = dci_test.o

$(CLUSTDESCROBJ): $(CLUSTDESCRSRC) $(CLUSTDESCRDEPS)
	$(CC) -c -o $@ $< $(DCIFLAGS)

$(CLUSTUTILSOBJ): $(CLUSTUTILSSRC) $(CLUSTUTILSDEPS)
	$(CC) -c -o $@ $< $(DCIFLAGS)

$(FILEUTILSOBJ): $(FILEUTILSSRC) $(FILEUTILSDEPS)
	$(CC) -c -o $@ $< $(DCIFLAGS)

$(COMMANDLINEOBJ): $(COMMANDLINESRC) $(COMMANDLINEDEPS)
	$(CC) -c -o $@ $< $(DCIFLAGS)

$(DCIOBJ): $(DCISRC)
	$(CC) -c -o $@ $< $(DCIFLAGS)

dci: $(CLUSTDESCROBJ) $(CLUSTUTILSOBJ) $(FILEUTILSOBJ) $(COMMANDLINEOBJ) $(DCIOBJ)
	$(CC) -o $@ $^ $(DCIFLAGS)
	mv $@ $(BDIR)
	rm $(DCIOBJ)
	rm $(CLUSTDESCROBJ)
	rm $(CLUSTUTILSOBJ)
	rm $(FILEUTILSOBJ)
	rm $(COMMANDLINEOBJ)

$(HOMGENOBJ): $(HOMGENSRC)
	$(CC) -c -o $@ $< $(HOMGENFLAGS)

homgen: $(CLUSTDESCROBJ) $(CLUSTUTILSOBJ) $(FILEUTILSOBJ) $(COMMANDLINEOBJ) $(HOMGENOBJ)
	$(CC) -o $@ $^ $(DCIFLAGS)
	mv $@ $(BDIR)
	rm $(HOMGENOBJ)
	rm $(CLUSTDESCROBJ)
	rm $(CLUSTUTILSOBJ)
	rm $(FILEUTILSOBJ)
	rm $(COMMANDLINEOBJ)

$(QUERYOBJ): $(QUERYSRC) $(QUERYDEPS)
	$(CC) -c -o $@ $< $(QUERYFLAGS)

query: $(QUERYOBJ)
	$(CC) -o $@ $^ $(QUERYFLAGS)
	mv $@ $(BDIR)
	rm $(QUERYOBJ)

$(KMPSOOBJ): $(KMPSOSRC) $(DCIDEPS)
	$(CC) -c -o $@ $< $(KMPSOFLAGS)

kmpso: $(CLUSTDESCROBJ) $(CLUSTUTILSOBJ) $(FILEUTILSOBJ) $(KMPSOOBJ)
	$(CC) -o $@ $^ $(KMPSOFLAGS)
	mv $@ $(BDIR)
	rm $(KMPSOOBJ)
	rm $(CLUSTDESCROBJ)
	rm $(CLUSTUTILSOBJ)
	rm $(FILEUTILSOBJ)

$(TESTOBJ): $(TESTSRC) $(DCIDEPS)
	$(CC) -c -o $@ $< $(TESTFLAGS)

dcitest: $(CLUSTDESCROBJ) $(CLUSTUTILSOBJ) $(FILEUTILSOBJ) $(TESTOBJ)
	$(CC) -o $@ $^ $(TESTFLAGS)
	mv $@ $(BDIR)
	rm $(TESTOBJ)
	rm $(CLUSTDESCROBJ)
	rm $(CLUSTUTILSOBJ)
	rm $(FILEUTILSOBJ)

clean:
	rm $(BDIR)/*
