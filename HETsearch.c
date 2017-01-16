#include "HETsearch.h"


// Heterogeneus search with: (1) SSE instructions and Score Profile in CPU (2) KNC instructions and Adaptive Profile in MIC
void het_search_sse_sp_knc_ap (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned int query_sequences_count, unsigned long int Q,
	unsigned int * query_disp, unsigned long int vect_sequences_db_count, char ** chunk_b, unsigned int chunk_count, unsigned int * chunk_vect_sequences_db_count,
	unsigned short int ** chunk_n, unsigned int ** chunk_b_disp, unsigned long int * chunk_vD, char * submat, int open_gap, int extend_gap,
	int cpu_threads, int cpu_block_size, int num_mics, int mic_threads, int * scores, double * workTime, unsigned short int query_length_threshold) {

	unsigned long int offload_max_vD=0, * chunk_accum_vect_sequences_db_count;
	long int i=0, ii, j=0, k=0, jj;
	double tick;
	unsigned short int * m, *n, sequences_db_max_length, query_sequences_max_length; 
	unsigned int * a_disp, * b_disp = NULL, offload_max_vect_sequences_db_count=0, qp_count, sp_count;
	unsigned int offload_vect_sequences_db_count;
	char *a, * b, * queryProfiles;

	a = query_sequences;			b = chunk_b[0];				
	m = query_sequences_lengths;	n = chunk_n[0];				
	a_disp = query_disp;			b_disp = chunk_b_disp[0];	

	query_sequences_max_length = query_sequences_lengths[query_sequences_count-1];
	sequences_db_max_length = chunk_n[chunk_count-1][chunk_vect_sequences_db_count[chunk_count-1]-1];

	// calculate maximum chunk size
	for (i=0; i<chunk_count ; i++) 
		offload_max_vD = (offload_max_vD > chunk_vD[i] ? offload_max_vD : chunk_vD[i]);

	// calculate maximum chunk sequences count
	for (i=0; i<chunk_count ; i++) 
		offload_max_vect_sequences_db_count = (offload_max_vect_sequences_db_count > chunk_vect_sequences_db_count[i] ? offload_max_vect_sequences_db_count : chunk_vect_sequences_db_count[i]);

	// calculate number of query sequences that are processed with query and score profile
	i = 0;
	while ((i < query_sequences_count) && (query_sequences_lengths[i] < query_length_threshold))
		i++;
	qp_count = i;
	sp_count = query_sequences_count-qp_count;

	// build query profile's
	queryProfiles = (char *)_mm_malloc(Q*BLOSUM_COLS*sizeof(char), 64);
	for (i=0; i<Q ; i++)
		memcpy(queryProfiles+i*BLOSUM_COLS,submat+a[i]*BLOSUM_COLS,BLOSUM_COLS*sizeof(char));

	// Allocate memory for CPU buffers
	chunk_accum_vect_sequences_db_count = (unsigned long int *)_mm_malloc(chunk_count*sizeof(unsigned long int), 32);

	chunk_accum_vect_sequences_db_count[0] = 0;
	for (i=1; i<chunk_count ; i++)
		chunk_accum_vect_sequences_db_count[i] = chunk_accum_vect_sequences_db_count[i-1] + chunk_vect_sequences_db_count[i-1];

	// allow nested parallelism
	omp_set_nested(1);

	tick = dwalltime();

	#pragma omp parallel default(none) shared(queryProfiles, submat, a,m,a_disp,query_sequences_count,b,n,b_disp,vect_sequences_db_count,scores, cpu_block_size, num_mics, mic_threads, open_gap, extend_gap, cpu_threads, qp_count,sp_count, chunk_b, chunk_n, chunk_b_disp, chunk_vD, query_sequences_max_length, sequences_db_max_length, Q, chunk_vect_sequences_db_count, chunk_accum_vect_sequences_db_count, chunk_count, offload_max_vD, offload_max_vect_sequences_db_count, query_length_threshold) num_threads(num_mics+1)
	{

		// data for MIC thread
		__declspec(align(64)) __m512i  *mic_row_ptrs[MIC_MAX_NUM_THREADS]={NULL}, *mic_maxCol_ptrs[MIC_MAX_NUM_THREADS]={NULL}, *mic_maxRow_ptrs[MIC_MAX_NUM_THREADS]={NULL}, *mic_lastCol_ptrs[MIC_MAX_NUM_THREADS]={NULL};
		__declspec(align(64)) char * mic_scoreProfile_ptrs[MIC_MAX_NUM_THREADS]={NULL};
		unsigned long int offload_vD, scores_offset;
		unsigned int mic_chunks=0, offload_vect_sequences_db_count; 
		int * mic_scores;
		// data for CPU thread
		__m128i  *cpu_row_ptrs[cpu_threads], *cpu_row2_ptrs[cpu_threads], *cpu_row3_ptrs[cpu_threads];
		__m128i  *cpu_maxCol_ptrs[cpu_threads], *cpu_maxRow_ptrs[cpu_threads], *cpu_lastCol_ptrs[cpu_threads];
		char * cpu_scoreProfile_ptrs[cpu_threads];
		int cpu_chunks=0;
		unsigned int cpu_vect_sequences_db_count;
		// common data
		unsigned long int i, c;
		unsigned int * ptr_chunk_b_disp;
		int tid;
		unsigned short int * ptr_chunk_n;
		char * ptr_chunk_b;

		tid = omp_get_thread_num();

		if (tid < num_mics){

			// allocate buffers for MIC thread
			mic_scores = (int*) _mm_malloc(query_sequences_count*(offload_max_vect_sequences_db_count*MIC_KNC_INT32_VECTOR_LENGTH)*sizeof(int), 64);

			// pre-allocate buffers and transfer common thread data to corresponding MIC
			#pragma offload_transfer target(mic:tid) in(submat: length(BLOSUM_ELEMS) ALLOC) in(queryProfiles: length(Q*BLOSUM_COLS) ALLOC) \
				in(a: length(Q) ALLOC) in(m:length(query_sequences_count) ALLOC) in(a_disp: length(query_sequences_count) ALLOC) \
				nocopy(b:length(offload_max_vD) ALLOC)  nocopy(n:length(offload_max_vect_sequences_db_count) ALLOC) nocopy(b_disp: length(offload_max_vect_sequences_db_count) ALLOC) \
				nocopy(mic_scores: length(query_sequences_count*offload_max_vect_sequences_db_count*MIC_KNC_INT32_VECTOR_LENGTH) ALLOC) \
				nocopy(mic_row_ptrs, mic_maxCol_ptrs, mic_maxRow_ptrs, mic_lastCol_ptrs, mic_scoreProfile_ptrs: ALLOC)

		} 

		// distribute database chunk between MICs and CPU using dynamic scheduling
		#pragma omp for schedule(dynamic) nowait
		for (c=0; c < chunk_count ; c++) {

			ptr_chunk_b = chunk_b[c];
			ptr_chunk_n = chunk_n[c];
			ptr_chunk_b_disp = chunk_b_disp[c];
			scores_offset = chunk_accum_vect_sequences_db_count[c];
			
			if (tid < num_mics){ // MIC thread

				offload_vD = chunk_vD[c];
				offload_vect_sequences_db_count = chunk_vect_sequences_db_count[c];

				// process database chunk in MIC 
				#pragma offload target(mic:tid) in(ptr_chunk_b[0:offload_vD] : into(b[0:offload_vD]) REUSE) \ 
					in(ptr_chunk_n[0:offload_vect_sequences_db_count] : into(n[0:offload_vect_sequences_db_count]) REUSE) \
					in(ptr_chunk_b_disp[0:offload_vect_sequences_db_count] : into(b_disp[0:offload_vect_sequences_db_count]) REUSE) \
					out(mic_scores: length(query_sequences_count*offload_vect_sequences_db_count*MIC_KNC_INT32_VECTOR_LENGTH) REUSE) \
					in(a: length(0) REUSE) in(m: length(0) REUSE) in(a_disp: length(0) REUSE) in(submat: length(0) REUSE) in(queryProfiles: length(0) REUSE) \
					nocopy(mic_row_ptrs, mic_maxCol_ptrs, mic_maxRow_ptrs, mic_lastCol_ptrs, mic_scoreProfile_ptrs: REUSE) 
					#pragma omp parallel shared(c, mic_chunks, offload_vect_sequences_db_count, query_sequences_count, open_gap, extend_gap, query_sequences_max_length, sequences_db_max_length, query_length_threshold) num_threads(mic_threads)
					{
						__m512i  *row, *maxCol, *maxRow, *lastCol;
						int  * ptr_scores;
						char * ptr_a, * ptr_b, *ptr_b_block, * scoreProfile, *queryProfile, *ptr_scoreProfile;

						__declspec(align(64)) __m512i vzero = _mm512_setzero_epi32(), score, previous, current, aux1, aux2, aux3, aux4, auxLastCol;
						__declspec(align(64)) __m512i vextend_gap = _mm512_set1_epi32(extend_gap), vopen_extend_gap = _mm512_set1_epi32(open_gap+extend_gap);
						__declspec(align(64)) __m512i v16 = _mm512_set1_epi32(16), submat_hi, submat_lo, b_values;
						__mmask16 mask;

						unsigned int tid, i, j, ii, jj, k, disp_1, disp_2, disp_3, disp_4, dim, nbb;
						unsigned long int t, s, q; 

						tid = omp_get_thread_num();

						// if this is the first offload, allocate auxiliary buffers
						if (mic_chunks == 0)	{
							mic_row_ptrs[tid] = (__m512i *) _mm_malloc((MIC_KNC_BLOCK_SIZE+1)*sizeof(__m512i), 64);
							mic_maxCol_ptrs[tid] = (__m512i *) _mm_malloc((MIC_KNC_BLOCK_SIZE+1)*sizeof(__m512i), 64);
							mic_maxRow_ptrs[tid] = (__m512i *) _mm_malloc((query_sequences_max_length)*sizeof(__m512i), 64);
							mic_lastCol_ptrs[tid] = (__m512i *) _mm_malloc((query_sequences_max_length)*sizeof(__m512i), 64);
							if (query_sequences_max_length >= query_length_threshold)
								mic_scoreProfile_ptrs[tid] = (char *) _mm_malloc(BLOSUM_ROWS_x_MIC_KNC_INT32_VECTOR_LENGTH*MIC_KNC_BLOCK_SIZE*sizeof(char),16);
						}
						
						row = mic_row_ptrs[tid];
						maxCol = mic_maxCol_ptrs[tid];
						maxRow = mic_maxRow_ptrs[tid];
						lastCol = mic_lastCol_ptrs[tid];
						scoreProfile = mic_scoreProfile_ptrs[tid];

						// calculate chunk alignments using query profile technique
						#pragma omp for schedule(dynamic) nowait
						for (t=0; t< qp_count*offload_vect_sequences_db_count; t++) {

							q = (qp_count-1) - (t % qp_count);
							s = (offload_vect_sequences_db_count-1) - (t / qp_count);

							queryProfile = queryProfiles + a_disp[q]*BLOSUM_COLS;
							ptr_b = b + b_disp[s];
							ptr_scores = mic_scores + (q*offload_vect_sequences_db_count+s)*MIC_KNC_INT32_VECTOR_LENGTH;

							// init buffers
							#pragma unroll(MIC_KNC_UNROLL_COUNT)
							for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm512_setzero_epi32(); // index 0 is not used
							#pragma unroll(MIC_KNC_UNROLL_COUNT)
							for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm512_setzero_epi32();
							
							// set score to 0
							score = _mm512_setzero_epi32();

							// calculate number of blocks
							nbb = ceil( (double) n[s] / (double) MIC_KNC_BLOCK_SIZE);

							for (k=0; k < nbb; k++){

								// calculate dim
								disp_1 = k*MIC_KNC_BLOCK_SIZE;
								dim = (MIC_KNC_BLOCK_SIZE < n[s]-disp_1 ? MIC_KNC_BLOCK_SIZE : n[s]-disp_1);

								// get b block
								ptr_b_block = ptr_b + disp_1*MIC_KNC_INT32_VECTOR_LENGTH;

								// init buffers
								#pragma unroll(MIC_KNC_UNROLL_COUNT)
								for (i=1; i<dim+1 ; i++ ) maxCol[i] = _mm512_setzero_epi32(); //index 0 is not used
								#pragma unroll(MIC_KNC_UNROLL_COUNT)
								for (i=0; i<dim ; i++ ) row[i] = _mm512_setzero_epi32();
								auxLastCol = _mm512_setzero_epi32();

								for( i = 0; i < m[q]; i++){
							
									// previous must start in 0
									previous = _mm512_setzero_epi32();
									// update row[0] with lastCol elements
									row[0] = lastCol[i];
									// load submat values corresponding to current a residue
									disp_1 = i*BLOSUM_COLS;

									#if __MIC__
									submat_lo = _mm512_extload_epi32(queryProfile+disp_1, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0);
									submat_hi = _mm512_extload_epi32(queryProfile+disp_1+16, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0);
									#endif

									#pragma unroll(MIC_KNC_UNROLL_COUNT)
									for( jj=1; jj < dim+1;  jj++) {
										//calcuate the diagonal value
										#if __MIC__
										b_values = _mm512_extload_epi32(ptr_b_block+(jj-1)*MIC_KNC_INT32_VECTOR_LENGTH, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0);
										#endif
										mask = _mm512_cmpge_epi32_mask(b_values,v16);
										aux1 = _mm512_permutevar_epi32(b_values, submat_lo);
										aux1 = _mm512_mask_permutevar_epi32(aux1, mask, b_values, submat_hi);
										current = _mm512_add_epi32(row[jj-1], aux1);								
										// calculate current max value
										current = _mm512_max_epi32(current, maxRow[i]);
										current = _mm512_max_epi32(current, maxCol[jj]);
										current = _mm512_max_epi32(current, vzero);
										// update maxRow and maxCol
										maxRow[i] = _mm512_sub_epi32(maxRow[i], vextend_gap);
										maxCol[jj] = _mm512_sub_epi32(maxCol[jj], vextend_gap);
										aux1 = _mm512_sub_epi32(current, vopen_extend_gap);
										maxRow[i] = _mm512_max_epi32(maxRow[i], aux1);
										maxCol[jj] =  _mm512_max_epi32(maxCol[jj], aux1);	
										// update row buffer
										row[jj-1] = previous;
										previous = current;
										// update max score
										score = _mm512_max_epi32(score,current);
									}
									// update lastCol
									lastCol[i] = auxLastCol;
									auxLastCol = current;
								}
							}
							// store max value
							_mm512_store_epi32(ptr_scores, score);
						}

						// calculate chunk alignments using score profile technique
						#pragma omp for schedule(dynamic) nowait
						for (t=0; t< sp_count*offload_vect_sequences_db_count; t++) {

							q = qp_count + (sp_count-1) - (t % sp_count);
							s = (offload_vect_sequences_db_count-1) - (t / sp_count);

							ptr_a = a + a_disp[q];
							ptr_b = b + b_disp[s];
							ptr_scores = mic_scores + (q*offload_vect_sequences_db_count+s)*MIC_KNC_INT32_VECTOR_LENGTH;

							// init buffers
							#pragma unroll(MIC_KNC_UNROLL_COUNT)
							for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm512_setzero_epi32(); // index 0 is not used
							#pragma unroll(MIC_KNC_UNROLL_COUNT)
							for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm512_setzero_epi32();
							
							// set score to 0
							score = _mm512_setzero_epi32();

							// calculate number of blocks
							nbb = ceil( (double) n[s] / (double) MIC_KNC_BLOCK_SIZE);

							for (k=0; k < nbb; k++){

								// calculate dim
								disp_2 = k*MIC_KNC_BLOCK_SIZE;
								dim = (MIC_KNC_BLOCK_SIZE < n[s]-disp_2 ? MIC_KNC_BLOCK_SIZE : n[s]-disp_2);

								// init buffers
								#pragma unroll(MIC_KNC_UNROLL_COUNT)
								for (i=1; i<dim+1 ; i++ ) maxCol[i] = _mm512_setzero_epi32(); //index 0 is not used
								#pragma unroll(MIC_KNC_UNROLL_COUNT)
								for (i=0; i<dim ; i++ ) row[i] = _mm512_setzero_epi32();
								auxLastCol = _mm512_setzero_epi32();

								// build score profile
								disp_1 = dim*MIC_KNC_INT32_VECTOR_LENGTH;
								for (i=0; i< dim ;i++ ) {
									#if __MIC__
									aux1 = _mm512_extload_epi32(ptr_b+(disp_2+i)*MIC_KNC_INT32_VECTOR_LENGTH, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0);
									#endif
									disp_3 = i*MIC_KNC_INT32_VECTOR_LENGTH;
									#pragma unroll(MIC_KNC_UNROLL_COUNT)
									for (j=0; j< BLOSUM_ROWS-1; j++) {
										#if __MIC__
										aux2 = _mm512_i32extgather_epi32(aux1, submat + j*BLOSUM_COLS, _MM_UPCONV_EPI32_SINT8  , 1, 0);
										_mm512_extstore_epi32(scoreProfile+disp_3+j*disp_1, aux2, _MM_DOWNCONV_EPI32_SINT8 , _MM_HINT_NONE );
										#endif
									}
									#if __MIC__
									_mm512_extstore_epi32(scoreProfile+disp_3+(BLOSUM_ROWS-1)*disp_1, vzero, _MM_DOWNCONV_EPI32_SINT8 , _MM_HINT_NONE );
									#endif
								}

								for( i = 0; i < m[q]; i++){
							
									// previous must start in 0
									previous = _mm512_setzero_epi32();
									// update row[0] with lastCol elements
									row[0] = lastCol[i];
									// calculate i displacement
									ptr_scoreProfile = scoreProfile + ((int)(ptr_a[i]))*disp_1;

									#pragma unroll(MIC_KNC_UNROLL_COUNT)
									for( jj=1; jj < dim+1; jj++) {
										//calcuate the diagonal value
										#if __MIC__
										current = _mm512_add_epi32(row[jj-1], _mm512_extload_epi32(ptr_scoreProfile+(jj-1)*MIC_KNC_INT32_VECTOR_LENGTH, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0));
										#endif							
										// calculate current max value
										current = _mm512_max_epi32(current, maxRow[i]);
										current = _mm512_max_epi32(current, maxCol[jj]);
										current = _mm512_max_epi32(current, vzero);
										// update maxRow and maxCol
										maxRow[i] = _mm512_sub_epi32(maxRow[i], vextend_gap);
										maxCol[jj] = _mm512_sub_epi32(maxCol[jj], vextend_gap);
										aux4 = _mm512_sub_epi32(current, vopen_extend_gap);
										maxRow[i] = _mm512_max_epi32(maxRow[i], aux4);
										maxCol[jj] =  _mm512_max_epi32(maxCol[jj], aux4);	
										// update row buffer
										row[jj-1] = previous;
										previous = current;
										// update max score
										score = _mm512_max_epi32(score,current);
									}
									// update lastCol
									lastCol[i] = auxLastCol;
									auxLastCol = current;
								}
							}
							// store max value
							_mm512_store_epi32(ptr_scores, score);
						}
					}

				// copy scores from auxiliary buffer to final buffer
				for (i=0; i<query_sequences_count ; i++)
					memcpy(scores+(i*vect_sequences_db_count+scores_offset)*MIC_KNC_INT32_VECTOR_LENGTH,mic_scores+i*offload_vect_sequences_db_count*MIC_KNC_INT32_VECTOR_LENGTH,offload_vect_sequences_db_count*MIC_KNC_INT32_VECTOR_LENGTH*sizeof(int));
				mic_chunks++;

			} else {

				cpu_vect_sequences_db_count = chunk_vect_sequences_db_count[c];

				// process database chunk in CPU
				#pragma omp parallel num_threads(cpu_threads-num_mics)
				{

					__m128i  *row, *maxCol, *maxRow, *lastCol, * ptr_scores, *tmp, *ptr_scoreProfile, *ptr_scoreProfile2;
					char * ptr_a, * ptr_b, * scoreProfile;

					__declspec(align(32)) __m128i score, previous, current, auxBlosum[2], auxLastCol, b_values;
					__declspec(align(32)) __m128i aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
					__declspec(align(32)) __m128i vextend_gap_epi8 = _mm_set1_epi8(extend_gap), vopen_extend_gap_epi8 = _mm_set1_epi8(open_gap+extend_gap), vzero_epi8 = _mm_set1_epi8(0);
					__declspec(align(32)) __m128i vextend_gap_epi16 = _mm_set1_epi16(extend_gap), vopen_extend_gap_epi16 = _mm_set1_epi16(open_gap+extend_gap), vzero_epi16 = _mm_set1_epi16(0);
					__declspec(align(32)) __m128i vextend_gap_epi32 = _mm_set1_epi32(extend_gap), vopen_extend_gap_epi32 = _mm_set1_epi32(open_gap+extend_gap), vzero_epi32 = _mm_set1_epi32(0);
					__declspec(align(32)) __m128i v127 = _mm_set1_epi8(127), v32767 = _mm_set1_epi16(32767);
					__declspec(align(32)) __m128i v15 = _mm_set1_epi8(15), v16 = _mm_set1_epi8(16), vneg32 = _mm_set1_epi8(-32);

					unsigned int j, ii, jj, k, disp_1, disp_2, disp_3, disp_4, dim, nbb;
					unsigned long int t, s, q, ss; 
					int tid, overflow_flag, bb1, bb2, bb1_start, bb2_start, bb1_end, bb2_end;

					tid = omp_get_thread_num();

					if (cpu_chunks == 0) {
						// allocate buffers for CPU thread
						cpu_row_ptrs[tid] = (__m128i *) _mm_malloc((cpu_block_size+1)*sizeof(__m128i), 32);
						cpu_maxCol_ptrs[tid] = (__m128i *) _mm_malloc((cpu_block_size+1)*sizeof(__m128i), 32);
						cpu_maxRow_ptrs[tid] = (__m128i *) _mm_malloc((query_sequences_max_length)*sizeof(__m128i), 32);
						cpu_lastCol_ptrs[tid] = (__m128i *) _mm_malloc((query_sequences_max_length)*sizeof(__m128i), 32);
						cpu_scoreProfile_ptrs[tid] = (char *) _mm_malloc((BLOSUM_ROWS_x_CPU_SSE_INT8_VECTOR_LENGTH*cpu_block_size)*sizeof(char), 32);
					}

					row = cpu_row_ptrs[tid];
					maxCol = cpu_maxCol_ptrs[tid];
					maxRow = cpu_maxRow_ptrs[tid];
					lastCol = cpu_lastCol_ptrs[tid];
					scoreProfile = cpu_scoreProfile_ptrs[tid];

					// calculate chunk alignments using score profile
					#pragma omp for schedule(dynamic) nowait
					for (t=0; t< query_sequences_count*cpu_vect_sequences_db_count; t++) {

						q = (query_sequences_count-1) - (t % query_sequences_count);
						s = (cpu_vect_sequences_db_count-1) - (t / query_sequences_count);

						ptr_a = a + a_disp[q];
						ptr_b = ptr_chunk_b + ptr_chunk_b_disp[s];
						ptr_scores = (__m128i *) (scores + (q*vect_sequences_db_count+scores_offset+s)*CPU_SSE_INT8_VECTOR_LENGTH);

						// calculate number of blocks
						nbb = ceil( (double) ptr_chunk_n[s] / (double) cpu_block_size);

						// init buffers
						#pragma unroll(CPU_SSE_UNROLL_COUNT)
						for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm_set1_epi8(0);
						#pragma unroll(CPU_SSE_UNROLL_COUNT)
						for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm_set1_epi8(0);
							
						// set score to 0
						score = _mm_set1_epi8(0);

						for (k=0; k < nbb; k++){

							// calculate dim
							disp_4 = k*cpu_block_size;
							dim = ptr_chunk_n[s]-disp_4;
							dim = (cpu_block_size < dim ? cpu_block_size : dim);

							// init buffers
							#pragma unroll(CPU_SSE_UNROLL_COUNT)
							for (i=0; i<dim+1 ; i++ ) maxCol[i] = _mm_set1_epi8(0);
							#pragma unroll(CPU_SSE_UNROLL_COUNT)
							for (i=0; i<dim+1 ; i++ ) row[i] = _mm_set1_epi8(0);
							auxLastCol = _mm_set1_epi8(0);

							// calculate a[i] displacement
							disp_1 = dim*CPU_SSE_INT8_VECTOR_LENGTH;

							// build score profile
							for (i=0; i< dim ;i++ ) {
								// indexes
								b_values = _mm_loadu_si128((__m128i *) (ptr_b + (disp_4+i)*CPU_SSE_INT8_VECTOR_LENGTH));
								// indexes >= 16
								aux1 = _mm_sub_epi8(b_values, v16);
								// indexes < 16
								aux2 = _mm_cmpgt_epi8(b_values,v15);
								aux3 = _mm_and_si128(aux2,vneg32);
								aux4 = _mm_add_epi8(b_values,aux3);
								ptr_scoreProfile = (__m128i *)(scoreProfile) + i;
								for (j=0; j< BLOSUM_ROWS-1; j++) {
									tmp = (__m128i *) (submat + j*BLOSUM_COLS);
									auxBlosum[0] = _mm_load_si128(tmp);
									auxBlosum[1] = _mm_load_si128(tmp+1);
									aux5  = _mm_shuffle_epi8(auxBlosum[0], aux4);			
									aux6  = _mm_shuffle_epi8(auxBlosum[1], aux1);			
									aux7 = _mm_add_epi8(aux5,  aux6);
									_mm_store_si128(ptr_scoreProfile+j*dim,   aux7);
								}
								_mm_store_si128(ptr_scoreProfile+(BLOSUM_ROWS-1)*dim,  vzero_epi8);
							}

							for( i = 0; i < m[q]; i++){
							
								// previous must start in 0
								previous = _mm_set1_epi8(0);
								// update row[0] with lastCol[i-1]
								row[0] = lastCol[i];
								// calculate i displacement
								ptr_scoreProfile = (__m128i *)(scoreProfile+((int)(ptr_a[i]))*disp_1);

								#pragma unroll(CPU_SSE_UNROLL_COUNT)
								for( jj=1; jj < dim+1; jj++) {
									//calcuate the diagonal value
									current = _mm_adds_epi8(row[jj-1], _mm_load_si128(ptr_scoreProfile+(jj-1)));
									// calculate current max value
									current = _mm_max_epi8(current, maxRow[i]);
									current = _mm_max_epi8(current, maxCol[jj]);
									current = _mm_max_epi8(current, vzero_epi8);
									// update max score
									score = _mm_max_epi8(score,current);
									// update maxRow and maxCol
									maxRow[i] = _mm_subs_epi8(maxRow[i], vextend_gap_epi8);
									maxCol[jj] = _mm_subs_epi8(maxCol[jj], vextend_gap_epi8);
									aux0 = _mm_subs_epi8(current, vopen_extend_gap_epi8);
									maxRow[i] = _mm_max_epi8(maxRow[i], aux0);
									maxCol[jj] =  _mm_max_epi8(maxCol[jj], aux0);	
									// update row buffer
									row[jj-1] = previous;
									previous = current;
								}
								// update lastCol
								lastCol[i] = auxLastCol;
								auxLastCol = current;
							}
						}

						// store max value
						_mm_store_si128 (ptr_scores,_mm_cvtepi8_epi32(score));
						_mm_store_si128 (ptr_scores+1,_mm_cvtepi8_epi32(_mm_srli_si128(score,4)));
						_mm_store_si128 (ptr_scores+2,_mm_cvtepi8_epi32(_mm_srli_si128(score,8)));
						_mm_store_si128 (ptr_scores+3,_mm_cvtepi8_epi32(_mm_srli_si128(score,12)));

						// overflow detection
						aux1 = _mm_cmpeq_epi8(score,v127);
						overflow_flag = _mm_test_all_zeros(aux1,v127); 

						// if overflow
						if (overflow_flag == 0){

							// check overflow in lower 8-bits
							aux1 = _mm_cmpeq_epi8(_mm_slli_si128(score,8),v127);
							bb1_start = _mm_test_all_zeros(aux1,v127);
							// check overflow in upper 8-bits
							aux1 = _mm_cmpeq_epi8(_mm_srli_si128(score,8),v127);
							bb1_end = 2 - _mm_test_all_zeros(aux1,v127);

							// recalculate using 16-bit signed integer precision
							for (bb1=bb1_start; bb1<bb1_end ; bb1++){

								// init buffers
								#pragma unroll(CPU_SSE_UNROLL_COUNT)
								for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm_set1_epi16(0);
								#pragma unroll(CPU_SSE_UNROLL_COUNT)
								for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm_set1_epi16(0);
								
								// set score to 0
								score = _mm_set1_epi16(0);

								disp_2 = bb1*CPU_SSE_INT16_VECTOR_LENGTH;

								for (k=0; k < nbb; k++){

									// calculate dim
									disp_4 = k*cpu_block_size;
									dim = ptr_chunk_n[s]-disp_4;
									dim = (cpu_block_size < dim ? cpu_block_size : dim);

									// init buffers
									#pragma unroll(CPU_SSE_UNROLL_COUNT)
									for (i=0; i<dim+1 ; i++ ) maxCol[i] = _mm_set1_epi16(0);
									#pragma unroll(CPU_SSE_UNROLL_COUNT)
									for (i=0; i<dim+1 ; i++ ) row[i] = _mm_set1_epi16(0);
									auxLastCol = _mm_set1_epi16(0);

									// calculate a[i] displacement
									disp_1 = dim*CPU_SSE_INT8_VECTOR_LENGTH;

									// build score profile
									for (i=0; i< dim ;i++ ) {
										// indexes
										b_values = _mm_loadu_si128((__m128i *) (ptr_b + (disp_4+i)*CPU_SSE_INT8_VECTOR_LENGTH));
										// indexes >= 16
										aux1 = _mm_sub_epi8(b_values, v16);
										// indexes < 16
										aux2 = _mm_cmpgt_epi8(b_values,v15);
										aux3 = _mm_and_si128(aux2,vneg32);
										aux4 = _mm_add_epi8(b_values,aux3);
										ptr_scoreProfile = (__m128i *)(scoreProfile) + i;
										for (j=0; j< BLOSUM_ROWS-1; j++) {
											tmp = (__m128i *) (submat + j*BLOSUM_COLS);
											auxBlosum[0] = _mm_load_si128(tmp);
											auxBlosum[1] = _mm_load_si128(tmp+1);
											aux5  = _mm_shuffle_epi8(auxBlosum[0], aux4);			
											aux6  = _mm_shuffle_epi8(auxBlosum[1], aux1);			
											aux7 = _mm_add_epi8(aux5,  aux6);
											_mm_store_si128(ptr_scoreProfile+j*dim,   aux7);
										}
										_mm_store_si128(ptr_scoreProfile+(BLOSUM_ROWS-1)*dim,  vzero_epi8);
									}

									for( i = 0; i < m[q]; i++){
									
										// previous must start in 0
										previous = _mm_set1_epi16(0);
										// update row[0] with lastCol[i-1]
										row[0] = lastCol[i];
										// calculate i displacement
										ptr_scoreProfile = (__m128i *)(scoreProfile+((int)(ptr_a[i]))*disp_1+disp_2);

										#pragma unroll(CPU_SSE_UNROLL_COUNT)
										for(  jj=1; jj < dim+1; jj++) {
											//calcuate the diagonal value
											current = _mm_adds_epi16(row[jj-1], _mm_cvtepi8_epi16(_mm_loadu_si128(ptr_scoreProfile+(jj-1))));
											// calculate current max value
											current = _mm_max_epi16(current, maxRow[i]);
											current = _mm_max_epi16(current, maxCol[jj]);
											current = _mm_max_epi16(current, vzero_epi16);
											// update maxRow and maxCol
											maxRow[i] = _mm_subs_epi16(maxRow[i], vextend_gap_epi16);
											maxCol[jj] = _mm_subs_epi16(maxCol[jj], vextend_gap_epi16);
											aux0 = _mm_subs_epi16(current, vopen_extend_gap_epi16);
											maxRow[i] = _mm_max_epi16(maxRow[i], aux0);
											maxCol[jj] =  _mm_max_epi16(maxCol[jj], aux0);	
											// update row buffer
											row[jj-1] = previous;
											previous = current;
											// update max score
											score = _mm_max_epi16(score,current);
										}
										// update lastCol
										lastCol[i] = auxLastCol;
										auxLastCol = current;
									}
								}
								// store max value
								_mm_store_si128 (ptr_scores+bb1*2,_mm_cvtepi16_epi32(score));
								_mm_store_si128 (ptr_scores+bb1*2+1,_mm_cvtepi16_epi32(_mm_srli_si128(score,8)));

								// overflow detection
								aux1 = _mm_cmpeq_epi16(score,v32767);
								overflow_flag = _mm_test_all_zeros(aux1,v32767); 

								// if overflow
								if (overflow_flag == 0){

									// overflow detection in lower 16-bits
									aux1 = _mm_cmpeq_epi16(_mm_slli_si128(score,8),v32767);
									bb2_start = _mm_test_all_zeros(aux1,v32767);
									// overflow detection in upper 16-bits
									aux1 = _mm_cmpeq_epi16(_mm_srli_si128(score,8),v32767);
									bb2_end = 2 - _mm_test_all_zeros(aux1,v32767);

									// recalculate using 32-bit signed integer precision
									for (bb2=bb2_start; bb2<bb2_end ; bb2++){

										// init buffers
										#pragma unroll(CPU_SSE_UNROLL_COUNT)
										for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm_set1_epi32(0);
										#pragma unroll(CPU_SSE_UNROLL_COUNT)
										for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm_set1_epi32(0);
											
										// set score to 0
										score = _mm_set1_epi32(0);

										disp_3 = disp_2 + bb2*CPU_SSE_INT32_VECTOR_LENGTH;

										for (k=0; k < nbb; k++){

											// calculate dim
											disp_4 = k*cpu_block_size;
											dim = ptr_chunk_n[s]-disp_4;
											dim = (cpu_block_size < dim ? cpu_block_size : dim);

											// init buffers
											#pragma unroll(CPU_SSE_UNROLL_COUNT)
											for (i=0; i<dim+1 ; i++ ) maxCol[i] = _mm_set1_epi32(0);
											#pragma unroll(CPU_SSE_UNROLL_COUNT)
											for (i=0; i<dim+1 ; i++ ) row[i] = _mm_set1_epi32(0);
											auxLastCol = _mm_set1_epi32(0);

											// calculate a[i] displacement
											disp_1 = dim*CPU_SSE_INT8_VECTOR_LENGTH;

											// build score profile
											for (i=0; i< dim ;i++ ) {
												// indexes
												b_values = _mm_loadu_si128((__m128i *) (ptr_b + (disp_4+i)*CPU_SSE_INT8_VECTOR_LENGTH));
												// indexes >= 16
												aux1 = _mm_sub_epi8(b_values, v16);
												// indexes < 16
												aux2 = _mm_cmpgt_epi8(b_values,v15);
												aux3 = _mm_and_si128(aux2,vneg32);
												aux4 = _mm_add_epi8(b_values,aux3);
												ptr_scoreProfile = (__m128i *)(scoreProfile) + i;
												for (j=0; j< BLOSUM_ROWS-1; j++) {
													tmp = (__m128i *) (submat + j*BLOSUM_COLS);
													auxBlosum[0] = _mm_load_si128(tmp);
													auxBlosum[1] = _mm_load_si128(tmp+1);
													aux5  = _mm_shuffle_epi8(auxBlosum[0], aux4);			
													aux6  = _mm_shuffle_epi8(auxBlosum[1], aux1);			
													aux7 = _mm_add_epi8(aux5,  aux6);
													_mm_store_si128(ptr_scoreProfile+j*dim,   aux7);
												}
												_mm_store_si128(ptr_scoreProfile+(BLOSUM_ROWS-1)*dim,  vzero_epi8);
											}

											for( i = 0; i < m[q]; i++){
											
												// previous must start in 0
												previous = _mm_set1_epi32(0);
												// update row[0] with lastCol[i-1]
												row[0] = lastCol[i];
												// calculate i displacement
												ptr_scoreProfile = (__m128i *)(scoreProfile+((int)(ptr_a[i]))*disp_1+disp_3);

												#pragma unroll(CPU_SSE_UNROLL_COUNT)
												for( jj=1; jj < dim+1;  jj++) {
													//calcuate the diagonal value
													current = _mm_add_epi32(row[jj-1], _mm_cvtepi8_epi32(_mm_loadu_si128(ptr_scoreProfile+(jj-1))));
													// calculate current max value
													current = _mm_max_epi32(current, maxRow[i]);
													current = _mm_max_epi32(current, maxCol[jj]);
													current = _mm_max_epi32(current, vzero_epi32);
													// update maxRow and maxCol
													maxRow[i] = _mm_sub_epi32(maxRow[i], vextend_gap_epi32);
													maxCol[jj] = _mm_sub_epi32(maxCol[jj], vextend_gap_epi32);
													aux0 = _mm_sub_epi32(current, vopen_extend_gap_epi32);
													maxRow[i] = _mm_max_epi32(maxRow[i], aux0);
													maxCol[jj] =  _mm_max_epi32(maxCol[jj], aux0);	
													// update row buffer
													row[jj-1] = previous;
													previous = current;
													// update max score
													score = _mm_max_epi32(score,current);
												}
												// update lastCol
												lastCol[i] = auxLastCol;
												auxLastCol = current;
											}

										}
										// store max value
										_mm_store_si128 (ptr_scores+bb1*2+bb2,score);
									}
								}
							}
						}
					}
				}
				cpu_chunks++;
			}

		}


		if (tid < num_mics){
			
			// de-allocate buffers in corresponding MIC
			#pragma offload_transfer target(mic:tid) nocopy(submat: length(0) FREE) nocopy(queryProfiles: length(0) FREE) \
				nocopy(a: length(0) FREE) nocopy(m:length(0) FREE) nocopy(a_disp: length(0) FREE) \
				nocopy(b:length(0) FREE)  nocopy(n:length(0) FREE) nocopy(b_disp: length(0) FREE) \
				nocopy(mic_scores:length(0) FREE) \
				nocopy(mic_row_ptrs, mic_maxCol_ptrs, mic_maxRow_ptrs, mic_lastCol_ptrs, mic_scoreProfile_ptrs: FREE)

			_mm_free(mic_scores); 

		} else {
			// de-allocate CPU buffers
			if (cpu_chunks > 0){
				for (i=0; i<cpu_threads-num_mics ; i++){
					 _mm_free(cpu_row_ptrs[i]);
					 _mm_free(cpu_maxCol_ptrs[i]);
					 _mm_free(cpu_maxRow_ptrs[i]);
					 _mm_free(cpu_lastCol_ptrs[i]);
					 _mm_free(cpu_scoreProfile_ptrs[i]);
				}
			}
		}


	}

	*workTime = dwalltime()-tick;

	_mm_free(queryProfiles);	
}


// Heterogeneus search with: (1) AVX2 instructions and Score Profile in CPU (2) KNC instructions and Adaptive Profile in MIC
void het_search_avx2_sp_knc_ap (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned int query_sequences_count, unsigned long int Q,
	unsigned int * query_disp, unsigned long int vect_sequences_db_count, char ** chunk_b, unsigned int chunk_count, unsigned int * chunk_vect_sequences_db_count,
	unsigned short int ** chunk_n, unsigned int ** chunk_b_disp, unsigned long int * chunk_vD, char * submat, int open_gap, int extend_gap,
	int cpu_threads, int cpu_block_size, int num_mics, int mic_threads, int * scores, double * workTime, unsigned short int query_length_threshold) {

	unsigned long int offload_max_vD=0, * chunk_accum_vect_sequences_db_count;
	long int i=0, ii, j=0, k=0, jj;
	double tick;
	unsigned short int * m, *n, sequences_db_max_length, query_sequences_max_length; 
	unsigned int * a_disp, * b_disp = NULL, offload_max_vect_sequences_db_count=0, qp_count, sp_count;
	unsigned int offload_vect_sequences_db_count;
	char *a, * b, *queryProfiles;

	a = query_sequences;			b = chunk_b[0];				
	m = query_sequences_lengths;	n = chunk_n[0];				
	a_disp = query_disp;			b_disp = chunk_b_disp[0];	

	query_sequences_max_length = query_sequences_lengths[query_sequences_count-1];
	sequences_db_max_length = chunk_n[chunk_count-1][chunk_vect_sequences_db_count[chunk_count-1]-1];

	// calculate maximum chunk size
	for (i=0; i<chunk_count ; i++) 
		offload_max_vD = (offload_max_vD > chunk_vD[i] ? offload_max_vD : chunk_vD[i]);

	// calculate maximum chunk sequences count
	for (i=0; i<chunk_count ; i++) 
		offload_max_vect_sequences_db_count = (offload_max_vect_sequences_db_count > chunk_vect_sequences_db_count[i] ? offload_max_vect_sequences_db_count : chunk_vect_sequences_db_count[i]);

	// calculate number of query sequences that are processed with query and score profile
	i = 0;
	while ((i < query_sequences_count) && (query_sequences_lengths[i] < query_length_threshold))
		i++;
	qp_count = i;
	sp_count = query_sequences_count-qp_count;

	// build query profile's
	queryProfiles = (char *)_mm_malloc(Q*BLOSUM_COLS*sizeof(char), 64);
	for (i=0; i<Q ; i++)
		memcpy(queryProfiles+i*BLOSUM_COLS,submat+a[i]*BLOSUM_COLS,BLOSUM_COLS*sizeof(char));

	// Allocate memory for CPU buffers
	chunk_accum_vect_sequences_db_count = (unsigned long int *)_mm_malloc(chunk_count*sizeof(unsigned long int), 32);

	chunk_accum_vect_sequences_db_count[0] = 0;
	for (i=1; i<chunk_count ; i++)
		chunk_accum_vect_sequences_db_count[i] = chunk_accum_vect_sequences_db_count[i-1] + chunk_vect_sequences_db_count[i-1];

	// allow nested parallelism
	omp_set_nested(1);

	tick = dwalltime();

	#pragma omp parallel default(none) shared(queryProfiles,submat, a,m,a_disp,query_sequences_count,b,n,b_disp,vect_sequences_db_count,scores, cpu_block_size, num_mics, mic_threads, open_gap, extend_gap, cpu_threads, qp_count,sp_count, chunk_b, chunk_n, chunk_b_disp, chunk_vD, query_sequences_max_length, sequences_db_max_length, Q, chunk_vect_sequences_db_count, chunk_accum_vect_sequences_db_count, chunk_count, offload_max_vD, offload_max_vect_sequences_db_count, query_length_threshold) num_threads(num_mics+1)
	{

		// data for MIC thread
		__declspec(align(64)) __m512i  *mic_row_ptrs[MIC_MAX_NUM_THREADS]={NULL}, *mic_maxCol_ptrs[MIC_MAX_NUM_THREADS]={NULL}, *mic_maxRow_ptrs[MIC_MAX_NUM_THREADS]={NULL}, *mic_lastCol_ptrs[MIC_MAX_NUM_THREADS]={NULL};
		__declspec(align(64)) char * mic_scoreProfile_ptrs[MIC_MAX_NUM_THREADS]={NULL};
		unsigned long int offload_vD, scores_offset;
		unsigned int mic_chunks=0, offload_vect_sequences_db_count; 
		int * mic_scores;
		// data for CPU thread
		__m256i  *cpu_row_ptrs[cpu_threads],  *cpu_row2_ptrs[cpu_threads],  *cpu_row3_ptrs[cpu_threads];
		__m256i	*cpu_maxCol_ptrs[cpu_threads], *cpu_maxRow_ptrs[cpu_threads], *cpu_lastCol_ptrs[cpu_threads];
		char * cpu_scoreProfile_ptrs[cpu_threads];
		int cpu_chunks=0;
		unsigned int cpu_vect_sequences_db_count;
		// common data
		unsigned long int i, c;
		unsigned int * ptr_chunk_b_disp;
		int tid;
		unsigned short int * ptr_chunk_n;
		char * ptr_chunk_b;

		tid = omp_get_thread_num();

		if (tid < num_mics){

			// allocate buffers for MIC thread
			mic_scores = (int*) _mm_malloc(query_sequences_count*(offload_max_vect_sequences_db_count*CPU_AVX2_INT8_VECTOR_LENGTH)*sizeof(int), 64);

			// pre-allocate buffers and transfer common thread data to corresponding MIC
			#pragma offload_transfer target(mic:tid) in(submat: length(BLOSUM_ELEMS) ALLOC) in(queryProfiles: length(Q*BLOSUM_COLS) ALLOC) \
				in(a: length(Q) ALLOC) in(m:length(query_sequences_count) ALLOC) in(a_disp: length(query_sequences_count) ALLOC) \
				nocopy(b:length(offload_max_vD) ALLOC)  nocopy(n:length(offload_max_vect_sequences_db_count) ALLOC) nocopy(b_disp: length(offload_max_vect_sequences_db_count) ALLOC) \
				nocopy(mic_scores: length(query_sequences_count*offload_max_vect_sequences_db_count*CPU_AVX2_INT8_VECTOR_LENGTH) ALLOC) \
				nocopy(mic_row_ptrs, mic_maxCol_ptrs, mic_maxRow_ptrs, mic_lastCol_ptrs, mic_scoreProfile_ptrs: ALLOC)

		} 

		// distribute database chunk between MICs and CPU using dynamic scheduling
		#pragma omp for schedule(dynamic) nowait
		for (c=0; c < chunk_count ; c++) {

			ptr_chunk_b = chunk_b[c];
			ptr_chunk_n = chunk_n[c];
			ptr_chunk_b_disp = chunk_b_disp[c];
			scores_offset = chunk_accum_vect_sequences_db_count[c];
			
			if (tid < num_mics){ // MIC thread

				offload_vD = chunk_vD[c];
				offload_vect_sequences_db_count = chunk_vect_sequences_db_count[c];

				// process database chunk in MIC 
				#pragma offload target(mic:tid) in(ptr_chunk_b[0:offload_vD] : into(b[0:offload_vD]) REUSE) \ 
					in(ptr_chunk_n[0:offload_vect_sequences_db_count] : into(n[0:offload_vect_sequences_db_count]) REUSE) \
					in(ptr_chunk_b_disp[0:offload_vect_sequences_db_count] : into(b_disp[0:offload_vect_sequences_db_count]) REUSE) \
					out(mic_scores: length(query_sequences_count*offload_vect_sequences_db_count*CPU_AVX2_INT8_VECTOR_LENGTH) REUSE) \
					in(a: length(0) REUSE) in(m: length(0) REUSE) in(a_disp: length(0) REUSE) in(submat: length(0) REUSE) in(queryProfiles: length(0) REUSE) \
					nocopy(mic_row_ptrs, mic_maxCol_ptrs, mic_maxRow_ptrs, mic_lastCol_ptrs, mic_scoreProfile_ptrs: REUSE) 
					#pragma omp parallel shared(c, mic_chunks, offload_vect_sequences_db_count, query_sequences_count, open_gap, extend_gap, query_sequences_max_length, sequences_db_max_length, query_length_threshold) num_threads(mic_threads)
					{
						__m512i  *row, *maxCol, *maxRow, *lastCol;
						int  * ptr_scores;
						char * ptr_a, * ptr_b, *ptr_b_block, *ptr_scoreProfile, * scoreProfile, *queryProfile;

						__declspec(align(64)) __m512i vzero = _mm512_setzero_epi32(), score, previous, current, aux1, aux2, aux3, aux4, auxLastCol;
						__declspec(align(64)) __m512i vextend_gap = _mm512_set1_epi32(extend_gap), vopen_extend_gap = _mm512_set1_epi32(open_gap+extend_gap);
						__declspec(align(64)) __m512i v16 = _mm512_set1_epi32(16), submat_hi, submat_lo, b_values;
						__mmask16 mask;

						unsigned int tid, i, j, ii, jj, k, disp_1, disp_2, disp_3, dim, nbb, offset;
						unsigned long int t, tt, s, q; 

						tid = omp_get_thread_num();

						// if this is the first offload, allocate auxiliary buffers
						if (mic_chunks == 0)	{
							mic_row_ptrs[tid] = (__m512i *) _mm_malloc((MIC_KNC_BLOCK_SIZE+1)*sizeof(__m512i), 64);
							mic_maxCol_ptrs[tid] = (__m512i *) _mm_malloc((MIC_KNC_BLOCK_SIZE+1)*sizeof(__m512i), 64);
							mic_maxRow_ptrs[tid] = (__m512i *) _mm_malloc((query_sequences_max_length)*sizeof(__m512i), 64);
							mic_lastCol_ptrs[tid] = (__m512i *) _mm_malloc((query_sequences_max_length)*sizeof(__m512i), 64);
							if (query_sequences_max_length >= query_length_threshold)
								mic_scoreProfile_ptrs[tid] = (char *) _mm_malloc(BLOSUM_ROWS_x_MIC_KNC_INT32_VECTOR_LENGTH*MIC_KNC_BLOCK_SIZE*sizeof(char),16);
						}
						
						row = mic_row_ptrs[tid];
						maxCol = mic_maxCol_ptrs[tid];
						maxRow = mic_maxRow_ptrs[tid];
						lastCol = mic_lastCol_ptrs[tid];
						scoreProfile = mic_scoreProfile_ptrs[tid];

						// calculate chunk alignments using query profile technique
						#pragma omp for schedule(dynamic) nowait
						for (t=0; t< qp_count*offload_vect_sequences_db_count; t++) {

							q = (qp_count-1) - (t % qp_count);
							s = (offload_vect_sequences_db_count-1) - (t / qp_count);
							queryProfile = queryProfiles + a_disp[q]*BLOSUM_COLS;

							for (tt = 0; tt < MIC_KNC_INT32_TO_CPU_AVX2_INT8_ADAPT_FACTOR; tt++) {

								ptr_b = b + b_disp[s] + tt* MIC_KNC_INT32_VECTOR_LENGTH;
								ptr_scores = mic_scores + (q*offload_vect_sequences_db_count+s)*CPU_AVX2_INT8_VECTOR_LENGTH + tt* MIC_KNC_INT32_VECTOR_LENGTH;

								// init buffers
								#pragma unroll(MIC_KNC_UNROLL_COUNT)
								for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm512_setzero_epi32(); // index 0 is not used
								#pragma unroll(MIC_KNC_UNROLL_COUNT)
								for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm512_setzero_epi32();
								
								// set score to 0
								score = _mm512_setzero_epi32();

								// calculate number of blocks
								nbb = ceil( (double) n[s] / (double) MIC_KNC_BLOCK_SIZE);

								for (k=0; k < nbb; k++){

									// calculate dim
									disp_1 = k*MIC_KNC_BLOCK_SIZE;
									dim = (MIC_KNC_BLOCK_SIZE < n[s]-disp_1 ? MIC_KNC_BLOCK_SIZE : n[s]-disp_1);

									// get b block
									ptr_b_block = ptr_b + disp_1*CPU_AVX2_INT8_VECTOR_LENGTH;

									// init buffers
									#pragma unroll(MIC_KNC_UNROLL_COUNT)
									for (i=1; i<dim+1 ; i++ ) maxCol[i] = _mm512_setzero_epi32(); //index 0 is not used
									#pragma unroll(MIC_KNC_UNROLL_COUNT)
									for (i=0; i<dim ; i++ ) row[i] = _mm512_setzero_epi32();
									auxLastCol = _mm512_setzero_epi32();

									for( i =0; i < m[q]; i++){
								
										// previous must start in 0
										previous = _mm512_setzero_epi32();
										// update row[0] with lastCol elements
										row[0] = lastCol[i];
										// load submat values corresponding to current a residue
										disp_1 = i*BLOSUM_COLS;
										#if __MIC__
										submat_lo = _mm512_extload_epi32(queryProfile+disp_1, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0);
										submat_hi = _mm512_extload_epi32(queryProfile+disp_1+16, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0);
										#endif

										#pragma unroll(MIC_KNC_UNROLL_COUNT)
										for( j = 1; j < dim+1; j++) {
											//calcuate the diagonal value
											#if __MIC__
											b_values = _mm512_extload_epi32(ptr_b_block+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0);
											#endif
											mask = _mm512_cmpge_epi32_mask(b_values,v16);
											aux1 = _mm512_permutevar_epi32(b_values, submat_lo);
											aux1 = _mm512_mask_permutevar_epi32(aux1, mask, b_values, submat_hi);
											current = _mm512_add_epi32(row[j-1], aux1);								
											// calculate current max value
											current = _mm512_max_epi32(current, maxRow[i]);
											current = _mm512_max_epi32(current, maxCol[j]);
											current = _mm512_max_epi32(current, vzero);
											// update maxRow and maxCol
											maxRow[i] = _mm512_sub_epi32(maxRow[i], vextend_gap);
											maxCol[j] = _mm512_sub_epi32(maxCol[j], vextend_gap);
											aux1 = _mm512_sub_epi32(current, vopen_extend_gap);
											maxRow[i] = _mm512_max_epi32(maxRow[i], aux1);
											maxCol[j] =  _mm512_max_epi32(maxCol[j], aux1);	
											// update row buffer
											row[j-1] = previous;
											previous = current;
											// update max score
											score = _mm512_max_epi32(score,current);
										}

										// update lastCol
										lastCol[i] = auxLastCol;
										auxLastCol = current;
									}
								}
								// store max value
								_mm512_store_epi32(ptr_scores, score);
							}
						}

						// calculate chunk alignments using score profile technique
						#pragma omp for schedule(dynamic) nowait
						for (t=0; t< sp_count*offload_vect_sequences_db_count; t++) {

							q = qp_count + (sp_count-1) - (t % sp_count);
							s = (offload_vect_sequences_db_count-1) - (t / sp_count);
							ptr_a = a + a_disp[q];

							for (tt=0; tt<MIC_KNC_INT32_TO_CPU_AVX2_INT8_ADAPT_FACTOR; tt++){

								ptr_b = b + b_disp[s] + tt* MIC_KNC_INT32_VECTOR_LENGTH;
								ptr_scores = mic_scores + (q*offload_vect_sequences_db_count+s)*CPU_AVX2_INT8_VECTOR_LENGTH + tt* MIC_KNC_INT32_VECTOR_LENGTH;

								// init buffers
								#pragma unroll(MIC_KNC_UNROLL_COUNT)
								for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm512_setzero_epi32(); // index 0 is not used
								#pragma unroll(MIC_KNC_UNROLL_COUNT)
								for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm512_setzero_epi32();
								
								// set score to 0
								score = _mm512_setzero_epi32();

								// calculate number of blocks
								nbb = ceil( (double) n[s] / (double) MIC_KNC_BLOCK_SIZE);

								for (k=0; k < nbb; k++){

									// calculate dim
									disp_2 = k*MIC_KNC_BLOCK_SIZE;
									dim = (MIC_KNC_BLOCK_SIZE < n[s]-disp_2 ? MIC_KNC_BLOCK_SIZE : n[s]-disp_2);

									// init buffers
									#pragma unroll(MIC_KNC_UNROLL_COUNT)
									for (i=1; i<dim+1 ; i++ ) maxCol[i] = _mm512_setzero_epi32(); //index 0 is not used
									#pragma unroll(MIC_KNC_UNROLL_COUNT)
									for (i=0; i<dim ; i++ ) row[i] = _mm512_setzero_epi32();
									auxLastCol = _mm512_setzero_epi32();

									// build score profile
									disp_1 = dim*MIC_KNC_INT32_VECTOR_LENGTH;
									for (i=0; i<dim ;i++ ) {
										#if __MIC__
										aux1 = _mm512_extload_epi32(ptr_b+(disp_2+i)*CPU_AVX2_INT8_VECTOR_LENGTH, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0);
										#endif
										disp_3 = i*MIC_KNC_INT32_VECTOR_LENGTH;
										#pragma unroll(MIC_KNC_UNROLL_COUNT)
										for (j=0; j< BLOSUM_ROWS-1; j++) {
											#if __MIC__
											aux2 = _mm512_i32extgather_epi32(aux1, submat + j*BLOSUM_COLS, _MM_UPCONV_EPI32_SINT8  , 1, 0);
											_mm512_extstore_epi32(scoreProfile+disp_3+j*disp_1, aux2, _MM_DOWNCONV_EPI32_SINT8 , _MM_HINT_NONE );
											#endif
										}
										#if __MIC__
										_mm512_extstore_epi32(scoreProfile+disp_3+(BLOSUM_ROWS-1)*disp_1, vzero, _MM_DOWNCONV_EPI32_SINT8 , _MM_HINT_NONE );
										#endif
									}


									for( i = 0; i < m[q]; i++){
								
										// previous must start in 0
										previous = _mm512_setzero_epi32();
										// update row[0] with lastCol elements
										row[0] = lastCol[i];
										// calculate i displacement
										ptr_scoreProfile = scoreProfile + ((int)(ptr_a[i]))*disp_1;

										#pragma unroll(MIC_KNC_UNROLL_COUNT)
										for( j=(k*MIC_KNC_BLOCK_SIZE)+1, jj=1; jj < dim+1; j++, jj++) {
											//calcuate the diagonal value
											#if __MIC__
											current = _mm512_add_epi32(row[jj-1], _mm512_extload_epi32(ptr_scoreProfile+(jj-1)*MIC_KNC_INT32_VECTOR_LENGTH, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0));
											#endif							
											// calculate current max value
											current = _mm512_max_epi32(current, maxRow[i]);
											current = _mm512_max_epi32(current, maxCol[jj]);
											current = _mm512_max_epi32(current, vzero);
											// update maxRow and maxCol
											maxRow[i] = _mm512_sub_epi32(maxRow[i], vextend_gap);
											maxCol[jj] = _mm512_sub_epi32(maxCol[jj], vextend_gap);
											aux1 = _mm512_sub_epi32(current, vopen_extend_gap);
											maxRow[i] = _mm512_max_epi32(maxRow[i], aux1);
											maxCol[jj] =  _mm512_max_epi32(maxCol[jj], aux1);	
											// update row buffer
											row[jj-1] = previous;
											previous = current;
											// update max score
											score = _mm512_max_epi32(score,current);
										}
										// update lastCol
										lastCol[i] = auxLastCol;
										auxLastCol = current;
									}
								}
								// store max value
								_mm512_store_epi32(ptr_scores, score);
							}
						}
					}

				// copy scores from auxiliary buffer to final buffer
				for (i=0; i<query_sequences_count ; i++)
					memcpy(scores+(i*vect_sequences_db_count+scores_offset)*CPU_AVX2_INT8_VECTOR_LENGTH,mic_scores+i*offload_vect_sequences_db_count*CPU_AVX2_INT8_VECTOR_LENGTH,offload_vect_sequences_db_count*CPU_AVX2_INT8_VECTOR_LENGTH*sizeof(int));
				mic_chunks++;

			} else {

				cpu_vect_sequences_db_count = chunk_vect_sequences_db_count[c];

				// process database chunk in CPU
				#pragma omp parallel num_threads(cpu_threads-num_mics)
				{

					__m256i *row, *maxCol, *maxRow, *lastCol, * ptr_scores, *ptr_scoreProfile;
					__m128i *tmp;
					char * ptr_a, * ptr_b, * scoreProfile;


					__declspec(align(32)) __m256i score, previous, current, auxLastCol, b_values, blosum_lo, blosum_hi;
					__declspec(align(32)) __m256i aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
					__declspec(align(32)) __m256i vextend_gap_epi8 = _mm256_set1_epi8(extend_gap), vopen_extend_gap_epi8 = _mm256_set1_epi8(open_gap+extend_gap);
					__declspec(align(32)) __m256i vextend_gap_epi16 = _mm256_set1_epi16(extend_gap), vopen_extend_gap_epi16 = _mm256_set1_epi16(open_gap+extend_gap);
					__declspec(align(32)) __m256i vextend_gap_epi32 = _mm256_set1_epi32(extend_gap), vopen_extend_gap_epi32 = _mm256_set1_epi32(open_gap+extend_gap);
					__declspec(align(32)) __m256i vzero_epi8 = _mm256_set1_epi8(0), vzero_epi16 = _mm256_set1_epi16(0), vzero_epi32 = _mm256_set1_epi32(0);
					__declspec(align(32)) __m256i v15 = _mm256_set1_epi8(15), vneg32 = _mm256_set1_epi8(-32), v16 = _mm256_set1_epi8(16);
					__declspec(align(32)) __m256i v127 = _mm256_set1_epi8(127), v32767 = _mm256_set1_epi16(32767);
					__declspec(align(32)) __m128i aux, auxBlosum[2];

					unsigned int i, j, ii, jj, k, disp_1, disp_2, disp_3, disp_4, dim, nbb;
					unsigned long int t, s, q; 
					int tid, overflow_flag, bb1, bb2, bb1_start, bb1_end, bb2_start, bb2_end;

					tid = omp_get_thread_num();

					// allocate memory for auxiliary buffers
					if (cpu_chunks == 0){
						cpu_row_ptrs[tid] = (__m256i *) _mm_malloc((cpu_block_size+1)*sizeof(__m256i), 32);
						cpu_maxCol_ptrs[tid] = (__m256i *) _mm_malloc((cpu_block_size+1)*sizeof(__m256i), 32);
						cpu_maxRow_ptrs[tid] = (__m256i *) _mm_malloc((query_sequences_max_length)*sizeof(__m256i), 32);
						cpu_lastCol_ptrs[tid] = (__m256i *) _mm_malloc((query_sequences_max_length)*sizeof(__m256i), 32);
						cpu_scoreProfile_ptrs[tid] = (char *) _mm_malloc((BLOSUM_ROWS_x_CPU_AVX2_INT8_VECTOR_LENGTH*cpu_block_size)*sizeof(char), 32);
					}

					row = cpu_row_ptrs[tid];
					maxCol = cpu_maxCol_ptrs[tid];
					maxRow = cpu_maxRow_ptrs[tid];
					lastCol = cpu_lastCol_ptrs[tid];
					scoreProfile = cpu_scoreProfile_ptrs[tid];
						
					// calculate alignment score
					#pragma omp for schedule(dynamic) nowait
					for (t=0; t< query_sequences_count*cpu_vect_sequences_db_count; t++) {

						q = (query_sequences_count-1) - (t % query_sequences_count);
						s = (cpu_vect_sequences_db_count-1) - (t / query_sequences_count);

						ptr_a = a + a_disp[q];
						ptr_b = ptr_chunk_b + ptr_chunk_b_disp[s];
						ptr_scores = (__m256i *) (scores + (q*vect_sequences_db_count+scores_offset+s)*CPU_AVX2_INT8_VECTOR_LENGTH);

						// caluclate number of blocks
						nbb = ceil( (double) ptr_chunk_n[s] / (double) cpu_block_size);

						// init buffers
						#pragma unroll(CPU_AVX2_UNROLL_COUNT)
						for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm256_set1_epi8(0);
						#pragma unroll(CPU_AVX2_UNROLL_COUNT)
						for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm256_set1_epi8(0);
							
						// set score to 0
						score = _mm256_set1_epi8(0);

						for (k=0; k < nbb; k++){

							// calculate dim
							disp_4 = k*cpu_block_size;
							dim = ptr_chunk_n[s]-disp_4;
							dim = (cpu_block_size < dim ? cpu_block_size : dim);

							// init buffers
							#pragma unroll(CPU_AVX2_UNROLL_COUNT)
							for (i=0; i<dim+1 ; i++ ) maxCol[i] = _mm256_set1_epi8(0);
							#pragma unroll(CPU_AVX2_UNROLL_COUNT)
							for (i=0; i<dim+1 ; i++ ) row[i] = _mm256_set1_epi8(0);
							auxLastCol = _mm256_set1_epi8(0);

							// calculate a[i] displacement
							disp_1 = dim*CPU_AVX2_INT8_VECTOR_LENGTH;

							// build score profile
							for (i=0; i< dim ;i++ ) {
								// indexes
								b_values =  _mm256_loadu_si256((__m256i *) (ptr_b + (disp_4+i)*CPU_AVX2_INT8_VECTOR_LENGTH));
								// indexes >= 16
								aux1 = _mm256_sub_epi8(b_values, v16);
								// indexes < 16
								aux2 = _mm256_cmpgt_epi8(b_values,v15);
								aux3 = _mm256_and_si256(aux2,vneg32);
								aux4 = _mm256_add_epi8(b_values,aux3);
								ptr_scoreProfile = (__m256i *)(scoreProfile) + i;
								for (j=0; j< BLOSUM_ROWS-1; j++) {
									tmp = (__m128i*) (submat +  j*BLOSUM_COLS);
									auxBlosum[0] = _mm_load_si128(tmp);
									auxBlosum[1] = _mm_load_si128(tmp+1);
									blosum_lo = _mm256_loadu2_m128i(&auxBlosum[0], &auxBlosum[0]);
									blosum_hi = _mm256_loadu2_m128i(&auxBlosum[1], &auxBlosum[1]);
									aux5 = _mm256_shuffle_epi8(blosum_lo,aux4);
									aux6 = _mm256_shuffle_epi8(blosum_hi,aux1);
									_mm256_store_si256(ptr_scoreProfile+j*dim,_mm256_or_si256(aux5,aux6));
								}
								_mm256_store_si256(ptr_scoreProfile+(BLOSUM_ROWS-1)*dim,vzero_epi8);
							}

							for( i = 0; i < m[q]; i++){
							
								// previous must start in 0
								previous = _mm256_set1_epi8(0);
								// update row[0] with lastCol[i-1]
								row[0] = lastCol[i];
								// calculate score profile displacement
								ptr_scoreProfile = (__m256i *)(scoreProfile+((unsigned int)(ptr_a[i]))*disp_1);

								#pragma unroll(CPU_AVX2_UNROLL_COUNT)
								for( jj=1; jj < dim+1; jj++) {
									//calcuate the diagonal value
									current =  _mm256_adds_epi8(row[jj-1], _mm256_load_si256(ptr_scoreProfile+(jj-1)));
									// calculate current max value
									current = _mm256_max_epi8(current, maxRow[i]);
									current = _mm256_max_epi8(current, maxCol[jj]);
									current = _mm256_max_epi8(current, vzero_epi8);
									// update maxRow and maxCol
									maxRow[i] =  _mm256_subs_epi8(maxRow[i], vextend_gap_epi8);
									maxCol[jj] = _mm256_subs_epi8(maxCol[jj], vextend_gap_epi8);
									aux0 =  _mm256_subs_epi8(current, vopen_extend_gap_epi8);
									maxRow[i] = _mm256_max_epi8(maxRow[i], aux0);
									maxCol[jj] =  _mm256_max_epi8(maxCol[jj], aux0);	
									// update row buffer
									row[jj-1] = previous;
									previous = current;
									// update max score
									score = _mm256_max_epi8(score,current);
								}
								// update lastCol
								lastCol[i] = auxLastCol;
								auxLastCol = current;
							}
						}

						// store max value
						aux = _mm256_extracti128_si256 (score,0);
						_mm256_store_si256 (ptr_scores,_mm256_cvtepi8_epi32(aux));
						_mm256_store_si256 (ptr_scores+1,_mm256_cvtepi8_epi32(_mm_srli_si128(aux,8)));
						aux = _mm256_extracti128_si256 (score,1);
						_mm256_store_si256 (ptr_scores+2,_mm256_cvtepi8_epi32(aux));
						_mm256_store_si256 (ptr_scores+3,_mm256_cvtepi8_epi32(_mm_srli_si128(aux,8)));

						// overflow detection
						aux1 = _mm256_cmpeq_epi8(score,v127);
						overflow_flag =  _mm256_testz_si256(aux1,v127); 

						// if overflow
						if (overflow_flag == 0){

							// check overflow in low 16 bits
							aux1 = _mm256_cmpeq_epi8(_mm256_inserti128_si256(vzero_epi8,_mm256_extracti128_si256(score,0),0),v127);
							bb1_start = _mm256_testz_si256(aux1,v127);
							// check overflow in high 16 bits
							aux1 = _mm256_cmpeq_epi8(_mm256_inserti128_si256(vzero_epi8,_mm256_extracti128_si256(score,1),0),v127);
							bb1_end = 2 - _mm256_testz_si256(aux1,v127);

							// recalculate using 16-bit signed integer precision
							for (bb1=bb1_start; bb1<bb1_end ; bb1++){

								// init buffers
								#pragma unroll(CPU_AVX2_UNROLL_COUNT)
								for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm256_set1_epi16(0);
								#pragma unroll(CPU_AVX2_UNROLL_COUNT)
								for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm256_set1_epi16(0);
								
								// set score to 0
								score = _mm256_set1_epi16(0);

								disp_2 = bb1*CPU_AVX2_INT16_VECTOR_LENGTH;

								for (k=0; k < nbb; k++){

									// calculate dim
									disp_4 = k*cpu_block_size;
									dim = ptr_chunk_n[s]-disp_4;
									dim = (cpu_block_size < dim ? cpu_block_size : dim);

									// init buffers
									#pragma unroll(CPU_AVX2_UNROLL_COUNT)
									for (i=0; i<dim+1 ; i++ ) maxCol[i] = _mm256_set1_epi16(0);
									#pragma unroll(CPU_AVX2_UNROLL_COUNT)
									for (i=0; i<dim+1 ; i++ ) row[i] = _mm256_set1_epi16(0);
									auxLastCol = _mm256_set1_epi16(0);

									// calculate a[i] displacement
									disp_1 = dim*CPU_AVX2_INT8_VECTOR_LENGTH;

									// build score profile
									for (i=0; i< dim ;i++ ) {
										// indexes
										b_values =  _mm256_loadu_si256((__m256i *) (ptr_b + (disp_4+i)*CPU_AVX2_INT8_VECTOR_LENGTH));
										// indexes >= 16
										aux1 = _mm256_sub_epi8(b_values, v16);
										// indexes < 16
										aux2 = _mm256_cmpgt_epi8(b_values,v15);
										aux3 = _mm256_and_si256(aux2,vneg32);
										aux4 = _mm256_add_epi8(b_values,aux3);
										ptr_scoreProfile = (__m256i *)(scoreProfile) + i;
										for (j=0; j< BLOSUM_ROWS-1; j++) {
											tmp = (__m128i*) (submat +  j*BLOSUM_COLS);
											auxBlosum[0] = _mm_load_si128(tmp);
											auxBlosum[1] = _mm_load_si128(tmp+1);
											blosum_lo = _mm256_loadu2_m128i(&auxBlosum[0], &auxBlosum[0]);
											blosum_hi = _mm256_loadu2_m128i(&auxBlosum[1], &auxBlosum[1]);
											aux5 = _mm256_shuffle_epi8(blosum_lo,aux4);
											aux6 = _mm256_shuffle_epi8(blosum_hi,aux1);
											_mm256_store_si256(ptr_scoreProfile+j*dim,_mm256_or_si256(aux5,aux6));
										}
										_mm256_store_si256(ptr_scoreProfile+(BLOSUM_ROWS-1)*dim,vzero_epi8);
									}

									for( i = 0; i < m[q]; i++){
									
										// previous must start in 0
										previous = _mm256_set1_epi16(0);
										// update row[0] with lastCol[i-1]
										row[0] = lastCol[i];
										// calculate score profile displacement
										ptr_scoreProfile = (__m256i*)(scoreProfile+((int)(ptr_a[i]))*disp_1+disp_2);

										#pragma unroll(CPU_AVX2_UNROLL_COUNT)
										for(  jj=1; jj < dim+1; jj++) {
											//calcuate the diagonal value
											current = _mm256_adds_epi16(row[jj-1], _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(jj-1)))));
											// calculate current max value
											current = _mm256_max_epi16(current, maxRow[i]);
											current = _mm256_max_epi16(current, maxCol[jj]);
											current = _mm256_max_epi16(current, vzero_epi16);
											// update maxRow and maxCol
											maxRow[i] = _mm256_subs_epi16(maxRow[i], vextend_gap_epi16);
											maxCol[jj] = _mm256_subs_epi16(maxCol[jj], vextend_gap_epi16);
											aux0 = _mm256_subs_epi16(current, vopen_extend_gap_epi16);
											maxRow[i] = _mm256_max_epi16(maxRow[i], aux0);
											maxCol[jj] =  _mm256_max_epi16(maxCol[jj], aux0);	
											// update row buffer
											row[jj-1] = previous;
											previous = current;
											// update max score
											score = _mm256_max_epi16(score,current);
										}
										// update lastCol
										lastCol[i] = auxLastCol;
										auxLastCol = current;

									}
								}
								// store max value
								aux = _mm256_extracti128_si256 (score,0);
								_mm256_store_si256 (ptr_scores+bb1*2,_mm256_cvtepi16_epi32(aux));
								aux = _mm256_extracti128_si256 (score,1);
								_mm256_store_si256 (ptr_scores+bb1*2+1,_mm256_cvtepi16_epi32(aux));

								// overflow detection
								aux1 = _mm256_cmpeq_epi16(score,v32767);
								overflow_flag =  _mm256_testz_si256(aux1,v32767); 

								// if overflow
								if (overflow_flag == 0){

									// check overflow in low 16 bits
									aux1 = _mm256_cmpeq_epi16(_mm256_inserti128_si256(vzero_epi16,_mm256_extracti128_si256(score,0),0),v32767);
									bb2_start = _mm256_testz_si256(aux1,v32767);
									// check overflow in high 16 bits
									aux1 = _mm256_cmpeq_epi16(_mm256_inserti128_si256(vzero_epi16,_mm256_extracti128_si256(score,1),0),v32767);
									bb2_end = 2 - _mm256_testz_si256(aux1,v32767);

									// recalculate using 32-bit signed integer precision
									for (bb2=bb2_start; bb2<bb2_end ; bb2++){

										// init buffers
										#pragma unroll(CPU_AVX2_UNROLL_COUNT)
										for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm256_set1_epi32(0);
										#pragma unroll(CPU_AVX2_UNROLL_COUNT)
										for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm256_set1_epi32(0);
											
										// set score to 0
										score = _mm256_set1_epi32(0);

										disp_3 = disp_2 + bb2*CPU_AVX2_INT32_VECTOR_LENGTH;

										for (k=0; k < nbb; k++){

											// calculate dim
											disp_4 = k*cpu_block_size;
											dim = ptr_chunk_n[s]-disp_4;
											dim = (cpu_block_size < dim ? cpu_block_size : dim);

											// init buffers
											#pragma unroll(CPU_AVX2_UNROLL_COUNT)
											for (i=0; i<dim+1 ; i++ ) maxCol[i] = _mm256_set1_epi32(0);
											#pragma unroll(CPU_AVX2_UNROLL_COUNT)
											for (i=0; i<dim+1 ; i++ ) row[i] = _mm256_set1_epi32(0);
											auxLastCol = _mm256_set1_epi32(0);

											// calculate a[i] displacement
											disp_1 = dim*CPU_AVX2_INT8_VECTOR_LENGTH;

											// build score profile
											for (i=0; i< dim ;i++ ) {
												// indexes
												b_values =  _mm256_loadu_si256((__m256i *) (ptr_b + (disp_4+i)*CPU_AVX2_INT8_VECTOR_LENGTH));
												// indexes >= 16
												aux1 = _mm256_sub_epi8(b_values, v16);
												// indexes < 16
												aux2 = _mm256_cmpgt_epi8(b_values,v15);
												aux3 = _mm256_and_si256(aux2,vneg32);
												aux4 = _mm256_add_epi8(b_values,aux3);
												ptr_scoreProfile = (__m256i *)(scoreProfile) + i;
												for (j=0; j< BLOSUM_ROWS-1; j++) {
													tmp = (__m128i*) (submat +  j*BLOSUM_COLS);
													auxBlosum[0] = _mm_load_si128(tmp);
													auxBlosum[1] = _mm_load_si128(tmp+1);
													blosum_lo = _mm256_loadu2_m128i(&auxBlosum[0], &auxBlosum[0]);
													blosum_hi = _mm256_loadu2_m128i(&auxBlosum[1], &auxBlosum[1]);
													aux5 = _mm256_shuffle_epi8(blosum_lo,aux4);
													aux6 = _mm256_shuffle_epi8(blosum_hi,aux1);
													_mm256_store_si256(ptr_scoreProfile+j*dim,_mm256_or_si256(aux5,aux6));
												}
												_mm256_store_si256(ptr_scoreProfile+(BLOSUM_ROWS-1)*dim,vzero_epi8);
											}

											for( i = 0; i < m[q]; i+=2){
											
												// previous must start in 0
												previous = _mm256_set1_epi32(0);
												// update row[0] with lastCol[i-1]
												row[0] = lastCol[i];
												// calculate score profile displacement
												ptr_scoreProfile = (__m256i *)(scoreProfile+((unsigned int)(ptr_a[i]))*disp_1+disp_3);

												#pragma unroll(CPU_AVX2_UNROLL_COUNT)
												for(  jj=1; jj < dim+1; jj++) {
													//calcuate the diagonal value
													current = _mm256_add_epi32(row[jj-1], _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(jj-1)))));
													// calculate current max value
													current = _mm256_max_epi32(current, maxRow[i]);
													current = _mm256_max_epi32(current, maxCol[jj]);
													current = _mm256_max_epi32(current, vzero_epi32);
													// update maxRow and maxCol
													maxRow[i] = _mm256_sub_epi32(maxRow[i], vextend_gap_epi32);
													maxCol[jj] = _mm256_sub_epi32(maxCol[jj], vextend_gap_epi32);
													aux0 = _mm256_sub_epi32(current, vopen_extend_gap_epi32);
													maxRow[i] = _mm256_max_epi32(maxRow[i], aux0);
													maxCol[jj] =  _mm256_max_epi32(maxCol[jj], aux0);	
													// update row buffer
													row[jj-1] = previous;
													previous = current;
													// update max score
													score = _mm256_max_epi32(score,current);
												}
												// update lastCol
												lastCol[i] = auxLastCol;
												auxLastCol = current;
											}
											// store max value
											_mm256_store_si256 (ptr_scores+bb1*2+bb2,score);
										}
									}

								}
							}

						}
					}
				}
				cpu_chunks++;
			}

		}

		if (tid < num_mics){
			
			// de-allocate buffers in corresponding MIC
			#pragma offload_transfer target(mic:tid) nocopy(submat: length(0) FREE) nocopy(queryProfiles: length(0) FREE) \
				nocopy(a: length(0) FREE) nocopy(m:length(0) FREE) nocopy(a_disp: length(0) FREE) \
				nocopy(b:length(0) FREE)  nocopy(n:length(0) FREE) nocopy(b_disp: length(0) FREE) \
				nocopy(mic_scores:length(0) FREE) \
				nocopy(mic_row_ptrs, mic_maxCol_ptrs, mic_maxRow_ptrs, mic_lastCol_ptrs, mic_scoreProfile_ptrs: FREE)

			_mm_free(mic_scores); 

		} else {
			// de-allocate CPU buffers
			if (cpu_chunks > 0){
				for (i=0; i<cpu_threads-num_mics ; i++){
					 _mm_free(cpu_row_ptrs[i]);
					 _mm_free(cpu_maxCol_ptrs[i]);
					 _mm_free(cpu_maxRow_ptrs[i]);
					 _mm_free(cpu_lastCol_ptrs[i]);
					 _mm_free(cpu_scoreProfile_ptrs[i]);
				}
			}
		}


	}

	*workTime = dwalltime()-tick;

	_mm_free(queryProfiles);
	
}