all: 
	icc -o swimm -O3 -openmp arguments.c CPUsearch.c HETsearch.c MICsearch.c sequences.c submat.c swimm.c utils.c

clean: 
	rm -rf swimm


