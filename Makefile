
SDIR = ./src
IDIR = ./include
BDIR = ./bin
DDIR = ./device
CUDADIR = /usr/local/cuda/samples/common/inc/
CC = nvcc

ERESSFLAGS = -use_fast_math -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES --std=c++11 -O2 -I $(IDIR)
HOMGENFLAGS = -use_fast_math -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES --std=c++11 -O2 -I $(IDIR)
KRESSFLAGS = -use_fast_math -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES --std=c++11 -O2 -I $(IDIR)
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

_ERESSSRC = eress.cu
ERESSSRC = $(patsubst %,$(SDIR)/%,$(_ERESSSRC))
ERESSOBJ = eress.o

_HOMGENSRC = homgen.cu
HOMGENSRC = $(patsubst %,$(SDIR)/%,$(_HOMGENSRC))
HOMGENOBJ = homgen.o

_KRESSSRC = kress.cu
KRESSSRC = $(patsubst %,$(SDIR)/%,$(_KRESSSRC))
KRESSOBJ = kress.o

$(CLUSTDESCROBJ): $(CLUSTDESCRSRC) $(CLUSTDESCRDEPS)
	$(CC) -c -o $@ $< $(ERESSFLAGS)

$(CLUSTUTILSOBJ): $(CLUSTUTILSSRC) $(CLUSTUTILSDEPS)
	$(CC) -c -o $@ $< $(ERESSFLAGS)

$(FILEUTILSOBJ): $(FILEUTILSSRC) $(FILEUTILSDEPS)
	$(CC) -c -o $@ $< $(ERESSFLAGS)

$(COMMANDLINEOBJ): $(COMMANDLINESRC) $(COMMANDLINEDEPS)
	$(CC) -c -o $@ $< $(ERESSFLAGS)

$(ERESSOBJ): $(ERESSSRC)
	$(CC) -c -o $@ $< $(ERESSFLAGS)

eress: $(CLUSTDESCROBJ) $(CLUSTUTILSOBJ) $(FILEUTILSOBJ) $(COMMANDLINEOBJ) $(ERESSOBJ)
	$(CC) -o $@ $^ $(ERESSFLAGS)
	mv $@ $(BDIR)
	rm $(ERESSOBJ)
	rm $(CLUSTDESCROBJ)
	rm $(CLUSTUTILSOBJ)
	rm $(FILEUTILSOBJ)
	rm $(COMMANDLINEOBJ)

$(HOMGENOBJ): $(HOMGENSRC)
	$(CC) -c -o $@ $< $(HOMGENFLAGS)

homgen: $(CLUSTDESCROBJ) $(CLUSTUTILSOBJ) $(FILEUTILSOBJ) $(HOMGENOBJ)
	$(CC) -o $@ $^ $(ERESSFLAGS)
	mv $@ $(BDIR)
	rm $(HOMGENOBJ)
	rm $(CLUSTDESCROBJ)
	rm $(CLUSTUTILSOBJ)
	rm $(FILEUTILSOBJ)

$(KRESSOBJ): $(KRESSSRC)
	$(CC) -c -o $@ $< $(KRESSFLAGS)

kress: $(CLUSTDESCROBJ) $(CLUSTUTILSOBJ) $(FILEUTILSOBJ) $(KRESSOBJ)
	$(CC) -o $@ $^ $(KRESSFLAGS)
	mv $@ $(BDIR)
	rm $(KRESSOBJ)
	rm $(CLUSTDESCROBJ)
	rm $(CLUSTUTILSOBJ)
	rm $(FILEUTILSOBJ)

clean:
	rm $(BDIR)/*
