#include "swimm.h"

// Global options
char *sequences_filename=NULL, * queries_filename=NULL, *input_filename=NULL, * output_filename=NULL, *op=NULL, * submat=blosum62, submat_name[]="BLOSUM62", profile=0;
int vector_length=CPU_SSE_INT8_VECTOR_LENGTH, execution_mode=HETEROGENEOUS, cpu_threads=CPU_THREADS, num_mics=NUM_MICS, mic_threads=MIC_THREADS, cpu_block_size=0, open_gap=OPEN_GAP, extend_gap=EXTEND_GAP;
unsigned short int query_length_threshold = QUERY_LENGTH_THRESHOLD;
unsigned long int max_chunk_size=MAX_CHUNK_SIZE, top=TOP;

int main(int argc, char *argv[]) {

	unsigned long int i, j, sequences_count, D, vect_sequences_db_count, vD, * chunk_vD, * vect_sequences_db_disp, query_sequences_count, Q;
	unsigned int chunk_count, * chunk_vect_sequences_db_count, ** chunk_vect_sequences_db_disp, * query_sequences_disp;
	int max_title_length, *scores;
	unsigned short int ** chunk_vect_sequences_db_lengths, * vect_sequences_db_lengths, * vect_sequences_db_blocks, sequences_db_max_length, * query_sequences_lengths, *m;
	char ** chunk_vect_sequences_db, * vect_sequences_db, *query_sequences, ** query_headers, ** sequence_db_headers, ** tmp_sequence_db_headers;
    time_t current_time = time(NULL);
	double workTime, tick;
	
	/* Process program arguments */
	program_arguments_processing(argc,argv);

	/* Database preprocessing */
	if (strcmp(op,"preprocess") == 0)
		preprocess_db (input_filename,output_filename,cpu_threads); 
	else {
		/* Database search */
		// Print database search information
		printf("\nSWIMM v%s \n\n",VERSION);
		printf("Database file:\t\t\t%s\n",sequences_filename);

		// Set CPU block size (if corresponds)
		if (cpu_block_size == 0) {
			cpu_block_size = (vector_length == CPU_AVX2_INT8_VECTOR_LENGTH ? CPU_AVX2_BLOCK_SIZE : CPU_SSE_BLOCK_SIZE);
			cpu_block_size = (cpu_block_size/SEQ_LEN_MULT)*SEQ_LEN_MULT;
		}
		
		// Load query sequence from file in a
		load_query_sequences(queries_filename,execution_mode,&query_sequences,&query_headers,&query_sequences_lengths,&m,&query_sequences_count,&Q,&query_sequences_disp,cpu_threads);

		// Assemble database (single chunk for CPU, multiple chunks for MIC and Heterogeneous)
		if (execution_mode == CPU_ONLY) {
			assemble_single_chunk_db (sequences_filename, vector_length, &sequences_count, &D, &sequences_db_max_length, &max_title_length, &vect_sequences_db_count, &vD, 
				&vect_sequences_db,	&vect_sequences_db_lengths,	&vect_sequences_db_blocks, &vect_sequences_db_disp, cpu_threads, cpu_block_size);
		}
		else 
			assemble_multiple_chunks_db (sequences_filename, vector_length, max_chunk_size, &sequences_count, &D, &sequences_db_max_length, &max_title_length,
				&vect_sequences_db_count, &vD, &chunk_vect_sequences_db, &chunk_count, &chunk_vect_sequences_db_count, &chunk_vD, &chunk_vect_sequences_db_lengths, 
				&chunk_vect_sequences_db_disp, cpu_threads);

		// Allocate buffers 
		top = (sequences_count < top ? sequences_count : top);
		scores = (int*) _mm_malloc(query_sequences_count*(vect_sequences_db_count*vector_length)*sizeof(int), (execution_mode == MIC_ONLY ? 64 : 32));
		tmp_sequence_db_headers = (char**) malloc(sequences_count*sizeof(char *));

		// Print database search information
		printf("Database size:\t\t\t%ld sequences (%ld residues) \n",sequences_count,D);
		printf("Longest database sequence: \t%d residues\n",sequences_db_max_length);
		printf("Substitution matrix:\t\t%s\n",submat_name);
		printf("Gap open penalty:\t\t%d\n",open_gap);
		printf("Gap extend penalty:\t\t%d\n",extend_gap);
		printf("Query filename:\t\t\t%s\n",queries_filename);

		workTime = dwalltime();

		// CPU search
		if (execution_mode == CPU_ONLY) {
			if (vector_length == CPU_SSE_INT8_VECTOR_LENGTH)
				// CPU search using SSE instrucions and Score Profile technique
				cpu_search_sse_sp (query_sequences, m, query_sequences_count, query_sequences_disp, vect_sequences_db,
					vect_sequences_db_lengths, vect_sequences_db_blocks, vect_sequences_db_count, vect_sequences_db_disp, submat, open_gap, extend_gap, cpu_threads, cpu_block_size, scores,
					&workTime);
			else
				// CPU search using AVX2 instrucions and Score Profile technique
				cpu_search_avx2_sp (query_sequences, m, query_sequences_count, query_sequences_disp, vect_sequences_db,
					vect_sequences_db_lengths, vect_sequences_db_blocks, vect_sequences_db_count, vect_sequences_db_disp, submat, open_gap, extend_gap, cpu_threads, cpu_block_size, scores,
					&workTime);
		} else {
			// MIC search
			if (execution_mode == MIC_ONLY) {
				// determine query length threshold
				if (profile == QUERY_PROFILE)
					query_length_threshold = query_sequences_lengths[query_sequences_count-1]+1;
				else 
					if (profile == SCORE_PROFILE)
						query_length_threshold = 0;
				// MIC search using KNC instrucions and Adaptive Profile technique
				if (chunk_count > 1)
					mic_search_knc_ap_multiple_chunks (query_sequences, m, query_sequences_count, Q, query_sequences_disp, vect_sequences_db_count, 
						chunk_vect_sequences_db, chunk_count, chunk_vect_sequences_db_count, chunk_vect_sequences_db_lengths, chunk_vect_sequences_db_disp,
						chunk_vD, submat, open_gap, extend_gap, num_mics, mic_threads, scores, &workTime, query_length_threshold);
				else
					mic_search_knc_ap_single_chunk (query_sequences, m, query_sequences_count, Q, query_sequences_disp, vect_sequences_db_count, 
						chunk_vect_sequences_db, chunk_count, chunk_vect_sequences_db_count, chunk_vect_sequences_db_lengths, chunk_vect_sequences_db_disp,
						chunk_vD, submat, open_gap, extend_gap, num_mics, mic_threads, scores, &workTime, query_length_threshold);

			} else { // Heterogeneous search

				// determine query length threshold
				if (profile == QUERY_PROFILE)
					query_length_threshold = query_sequences_lengths[query_sequences_count-1]+1;
				else 
					if (profile == SCORE_PROFILE)
						query_length_threshold = 0;

				if (vector_length == CPU_SSE_INT8_VECTOR_LENGTH) 
					// Heterogeneus search with: (1) SSE instructions and Score Profile in CPU (2) KNC instructions and Adaptive Profile in MIC
					het_search_sse_sp_knc_ap (query_sequences, m, query_sequences_count, Q, query_sequences_disp, 
						vect_sequences_db_count, chunk_vect_sequences_db, chunk_count, chunk_vect_sequences_db_count, chunk_vect_sequences_db_lengths,
						chunk_vect_sequences_db_disp, chunk_vD, submat, open_gap, extend_gap, cpu_threads, cpu_block_size, num_mics, mic_threads, scores, 
						&workTime, query_length_threshold);
				else
					// Heterogeneus search with: (1) AVX2 instructions and Score Profile in CPU (2) KNC instructions and Adaptive Profile in MIC
					het_search_avx2_sp_knc_ap (query_sequences, m, query_sequences_count, Q, query_sequences_disp, 
						vect_sequences_db_count, chunk_vect_sequences_db, chunk_count, chunk_vect_sequences_db_count, chunk_vect_sequences_db_lengths,
						chunk_vect_sequences_db_disp, chunk_vD, submat, open_gap, extend_gap, cpu_threads, cpu_block_size, num_mics, mic_threads, scores, 
						&workTime, query_length_threshold);

			}
		}


		// Free allocated memory
		_mm_free(query_sequences);
		_mm_free(query_sequences_disp);
		_mm_free(m);
		if (execution_mode == CPU_ONLY){
			_mm_free(vect_sequences_db);
			_mm_free(vect_sequences_db_lengths);
			_mm_free(vect_sequences_db_blocks);
			_mm_free(vect_sequences_db_disp);
		} else {
			_mm_free(chunk_vect_sequences_db[0]);
			_mm_free(chunk_vect_sequences_db);
			for (i=0; i< chunk_count ; i++ ) 
				_mm_free(chunk_vect_sequences_db_lengths[i]);
			_mm_free(chunk_vect_sequences_db_lengths);
			for (i=0; i< chunk_count ; i++ ) 
				_mm_free(chunk_vect_sequences_db_disp[i]);
			_mm_free(chunk_vect_sequences_db_disp);
			free(chunk_vect_sequences_db_count);
			free(chunk_vD);
		}

		// Load database headers
		load_database_headers (sequences_filename, sequences_count, max_title_length, &sequence_db_headers);

		// allow nested paralelism
		omp_set_nested(1);

		// Print top scores
		for (i=0; i<query_sequences_count ; i++ ) {
			memcpy(tmp_sequence_db_headers,sequence_db_headers,sequences_count*sizeof(char *));
			sort_scores(scores+i*vect_sequences_db_count*vector_length,tmp_sequence_db_headers,sequences_count,cpu_threads);
			printf("\nQuery no.\t\t\t%d\n",i+1);
			printf("Query description: \t\t%s\n",query_headers[i]+1);
			printf("Query length:\t\t\t%d residues\n",query_sequences_lengths[i]);
			printf("\nScore\tSequence description\n");
			for (j=0; j<top; j++) 
				printf("%d\t%s",scores[i*vect_sequences_db_count*vector_length+j],tmp_sequence_db_headers[j]+1);
		}
		printf("\nSearch date:\t\t\t%s",ctime(&current_time));
		printf("Search time:\t\t\t%lf seconds\n",workTime);
		printf("Search speed:\t\t\t%.2lf GCUPS\n",(Q*D) / (workTime*1000000000));
		if (execution_mode == CPU_ONLY) {
			printf("Execution mode:\t\t\tXeon only (%d threads, block width = %d)\n",cpu_threads,cpu_block_size);
			printf("Profile technique:\t\tScore Profile\n");
			printf("Instruction set:\t\t%s (vector length = %d)\n",(vector_length == 16 ? "SSE" : "AVX2"),vector_length);
		}
		else {
			if (execution_mode == MIC_ONLY) {
				printf("Execution mode:\t\t\tXeon Phi only (%d Xeon Phis with %d threads each)\n",num_mics,mic_threads);
				printf("Profile technique:\t\t%s",(profile == QUERY_PROFILE ? "Query Profile" : (profile == SCORE_PROFILE ? "Score Profile" : "Adaptive Profile")));
				if (profile == ADAPTIVE_PROFILE)
					printf(" (threshold = %d)",query_length_threshold);
				printf("\nInstruction set:\t\tKNC (vector length = %d)\n",vector_length);
				printf("Max. chunk size:\t\t%ld bytes\n",max_chunk_size);
				printf("Chunk count:\t\t\t%ld \n",chunk_count);
			} else {
				printf("Execution mode:\t\t\tHybrid (%d CPU threads (block width = %d) and %d Xeon Phis with %d threads each)\n",cpu_threads,cpu_block_size,num_mics,mic_threads);
				printf("Profile technique:\t\tScore Profile in Xeon, %s in Xeon Phi",(profile == QUERY_PROFILE ? "Query Profile" : (profile == SCORE_PROFILE ? "Score Profile" : "Adaptive Profile")));
				if (profile == ADAPTIVE_PROFILE)
					printf(" (threshold = %d)",query_length_threshold);
				if (vector_length == 16)
					printf("\nInstruction set:\t\tSSE+KNC (vector length = 16)\n");
				else
					printf("\nInstruction set:\t\tAVX2 (vector length = 32) + KNC (vector length = 16)\n");
				printf("Chunk count:\t\t\t%ld \n",chunk_count);
				printf("Max. chunk size:\t\t%ld bytes\n",max_chunk_size);
			}
		}


		// Free allocated memory
		_mm_free(query_sequences_lengths);
		_mm_free(scores); 	
		for (i=0; i<query_sequences_count ; i++ ) 
			free(query_headers[i]);
		free(query_headers);
		for (i=0; i<sequences_count ; i++ ) 
			free(sequence_db_headers[i]);
		free(sequence_db_headers);
		free(tmp_sequence_db_headers);

	}

	return 0;
}

