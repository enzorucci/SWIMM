#ifndef DB_H_INCLUDED
#define DB_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <omp.h>
#include "arguments.h"
#include "utils.h"
#include "swimm.h"

#define BUFFER_SIZE 1000
#define ALLOCATION_CHUNK 1000

#define DUMMY_ELEMENT 'Z'+1
#define PREPROCESSED_DUMMY_ELEMENT 24

// DB preprocessing
void preprocess_db (char * input_filename, char * out_filename, int n_procs);

// DB assembly for CPU
void assemble_single_chunk_db (char * sequences_filename, int vector_length, unsigned long int * sequences_count,
				unsigned long int * D, unsigned short int * sequences_db_max_length, int * max_title_length, unsigned long int * vect_sequences_db_count, unsigned long int * vD, char **ptr_vect_sequences_db,
				unsigned short int ** ptr_vect_sequences_db_lengths, unsigned long int ** ptr_vect_sequences_db_disp, int n_procs);

// DB assembly for MIC or HET
void assemble_multiple_chunks_db (char * sequences_filename, int vector_length, unsigned long int max_chunk_size, unsigned long int * sequences_count,
				unsigned long int * D, unsigned short int * sequences_db_max_length, int * max_title_length, unsigned long int * vect_sequences_db_count, unsigned long int * vD, char ***ptr_chunk_b,
				unsigned int * chunk_count, unsigned int ** ptr_chunk_vect_sequences_db_count, unsigned long int ** ptr_chunk_disp, 
				unsigned short int *** ptr_chunk_vect_sequences_db_lengths, unsigned int *** ptr_chunk_vect_sequences_db_disp, int n_procs);

// Load DB headers
void load_database_headers (char * sequences_filename, unsigned long int sequences_count, int max_title_length, char *** ptr_sequences_db_headers);

void load_query_sequences(char * queries_filename, int execution_mode, char ** ptr_query_sequences, char *** ptr_query_headers, unsigned short int **ptr_query_sequences_lengths,
						unsigned short int **ptr_m, unsigned long int * query_sequences_count, unsigned long int * ptr_Q, unsigned int ** ptr_query_sequences_disp, int n_procs) ; 

// Functions for parallel sorting
void merge_sequences(char ** sequences, char ** titles, unsigned short int * sequences_lengths, unsigned long int size);

void mergesort_sequences_serial(char ** sequences, char ** titles, unsigned short int * sequences_lengths, unsigned long int size);

void sort_sequences (char ** sequences,  char ** titles, unsigned short int * sequences_lengths, unsigned long int size, int threads);

#endif
