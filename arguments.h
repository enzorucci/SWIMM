#ifndef ARGUMENTS_H_INCLUDED
#define ARGUMENTS_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <argp.h>
#include "submat.h"

#define MAX_CHUNK_SIZE 100663296 // 96Mb
#define VECTOR_LENGTH 16
#define CPU_ONLY 0
#define MIC_ONLY 1
#define HETEROGENEOUS 2
#define CPU_THREADS 4 
#define CPU_BLOCK_SIZE 120
#define NUM_MICS 1
#define MIC_THREADS 240
#define OPEN_GAP 10
#define EXTEND_GAP 2
#define TOP 10
#define QUERY_PROFILE 'Q'
#define SCORE_PROFILE 'S'
#define ADAPTIVE_PROFILE 'A'
#define QUERY_LENGTH_THRESHOLD 567


// Arguments parsing
void program_arguments_processing (int argc, char * argv[]);
static int parse_opt (int key, char *arg, struct argp_state *state);

// Global options
extern char * sequences_filename, * queries_filename, *input_filename, * output_filename, *op, * submat, submat_name[];
extern char profile;
extern int vector_length, execution_mode, cpu_block_size, cpu_threads, num_mics, mic_threads, open_gap, extend_gap;
extern unsigned short int query_length_threshold;
extern unsigned long int max_chunk_size, top;
extern char blosum45[], blosum50[], blosum62[], blosum80[], blosum90[], pam30[], pam70[], pam250[];

#endif