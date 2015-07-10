#ifndef HETSEARCH_H_NOT_INCLUDED
#define HETSEARCH_H_NOT_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <omp.h>
#include <immintrin.h>
#include "submat.h"
#include "utils.h"
#include "CPUsearch.h"
#include "MICsearch.h"

// Heterogeneus search with: (1) SSE instructions and Score Profile in CPU (2) KNC instructions and Adaptive Profile in MIC
void het_search_sse_sp_knc_ap (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned int query_sequences_count, unsigned long int Q,
	unsigned int * query_disp, unsigned long int vect_sequences_db_count, char ** chunk_b, unsigned int chunk_count, unsigned int * chunk_vect_sequences_db_count,
	unsigned short int ** chunk_n, unsigned int ** chunk_b_disp, unsigned long int * chunk_disp, char * submat, int open_gap, int extend_gap,
	int cpu_threads, int cpu_block_size, int num_mics, int mic_threads, int * scores, double * workTime, unsigned short int query_length_threshold);

// Heterogeneus search with: (1) AVX2 instructions and Score Profile in CPU (2) KNC instructions and Adaptive Profile in MIC
void het_search_avx2_sp_knc_ap (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned int query_sequences_count, unsigned long int Q,
	unsigned int * query_disp, unsigned long int vect_sequences_db_count, char ** chunk_b, unsigned int chunk_count, unsigned int * chunk_vect_sequences_db_count,
	unsigned short int ** chunk_n, unsigned int ** chunk_b_disp, unsigned long int * chunk_disp, char * submat, int open_gap, int extend_gap,
	int cpu_threads, int cpu_block_size, int num_mics, int mic_threads, int * scores, double * workTime, unsigned short int query_length_threshold);


#endif