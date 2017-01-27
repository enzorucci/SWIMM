#ifndef CPUSEARCH_H_INCLUDED
#define CPUSEARCH_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <omp.h>
#include <immintrin.h>
#include "submat.h"
#include "utils.h"
#include "sequences.h"

#define CPU_SSE_INT8_VECTOR_LENGTH 16
#define CPU_SSE_INT16_VECTOR_LENGTH 8
#define CPU_SSE_INT32_VECTOR_LENGTH 4
#define CPU_SSE_UNROLL_COUNT 10
#define CPU_SSE_BLOCK_SIZE 128

#define BLOSUM_ROWS_x_CPU_SSE_INT8_VECTOR_LENGTH 384

#define CPU_AVX2_INT8_VECTOR_LENGTH 32
#define CPU_AVX2_INT16_VECTOR_LENGTH 16
#define CPU_AVX2_INT32_VECTOR_LENGTH 8
#define CPU_AVX2_UNROLL_COUNT 10
#define CPU_AVX2_BLOCK_SIZE 64

#define BLOSUM_ROWS_x_CPU_AVX2_INT8_VECTOR_LENGTH 768

// CPU search using SSE instrucions and Score Profile technique
void cpu_search_sse_sp (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned long int query_sequences_count, unsigned int * query_disp,
	char * vect_sequences_db, unsigned short int * vect_sequences_db_lengths, unsigned short int * vect_sequences_db_blocks, unsigned long int vect_sequences_db_count, unsigned long int * vect_sequences_db_disp,
	char * submat, int open_gap, int extend_gap, int n_threads, int cpu_block_size, int * scores, double * workTime);

// CPU search using AVX2 instrucions and Score Profile technique
void cpu_search_avx2_sp (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned long int query_sequences_count, unsigned int * query_disp,
	char * vect_sequences_db, unsigned short int * vect_sequences_db_lengths, unsigned short int * vect_sequences_db_blocks, unsigned long int vect_sequences_db_count, unsigned long int * vect_sequences_db_disp,
	char * submat, int open_gap, int extend_gap, int n_threads, int cpu_block_size, int * scores, double * workTime);

#endif
