#ifndef MICSEARCH_H_INCLUDED
#define MICSEARCH_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <omp.h>
#include <immintrin.h>
#include "submat.h"
#include "utils.h"

#define MIC_MAX_NUM_THREADS 244

#define MIC_KNC_INT32_VECTOR_LENGTH 16
#define MIC_KNC_INT32_TO_CPU_AVX2_INT8_ADAPT_FACTOR 2
#define MIC_KNC_UNROLL_COUNT 6
#define MIC_KNC_BLOCK_SIZE 256

#define BLOSUM_ROWS_x_MIC_KNC_INT32_VECTOR_LENGTH 384

#define ALLOC alloc_if(1) free_if(0)
#define FREE alloc_if(0) free_if(1)
#define REUSE alloc_if(0) free_if(0)
#define ALLOC_AND_FREE alloc_if(1) free_if(1)

// MIC search with KNC instructions and Adaptive Profile technique for single chunk database
void mic_search_knc_ap_single_chunk (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned int query_sequences_count, unsigned long int Q,
	unsigned int * query_disp, unsigned long int vect_sequences_db_count, char ** chunk_b, unsigned int chunk_count, unsigned int * chunk_vect_sequences_db_count,
	unsigned short int ** chunk_n, unsigned int ** chunk_b_disp, unsigned long int * chunk_disp, char * submat, int open_gap, int extend_gap,
	int num_mics, int mic_threads, int * scores, double * workTime, unsigned short int query_length_threshold);

// MIC search with KNC instructions and Adaptive Profile technique for multiple chunks database
void mic_search_knc_ap_multiple_chunks (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned int query_sequences_count, unsigned long int Q,
	unsigned int * query_disp, unsigned long int vect_sequences_db_count, char ** chunk_b, unsigned int chunk_count, unsigned int * chunk_vect_sequences_db_count,
	unsigned short int ** chunk_n, unsigned int ** chunk_b_disp, unsigned long int * chunk_disp, char * submat, int open_gap, int extend_gap,
	int num_mics, int mic_threads, int * scores, double * workTime, unsigned short int query_length_threshold);

#endif