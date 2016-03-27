#include "MICsearch.h"

// MIC search with KNC instructions and Adaptive Profile technique
void mic_search_knc_ap_multiple_chunks (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned int query_sequences_count, unsigned long int Q,
	unsigned int * query_disp, unsigned long int vect_sequences_db_count, char ** chunk_b, unsigned int chunk_count, unsigned int * chunk_vect_sequences_db_count,
	unsigned short int ** chunk_n, unsigned int ** chunk_b_disp, unsigned long int * chunk_vD, char * submat, int open_gap, int extend_gap,
	int num_mics, int mic_threads, int * scores, double * workTime, unsigned short int query_length_threshold){

	long int i=0, ii, j=0, k=0, jj;
	double tick;

	char *queryProfiles, *a, * b;
	unsigned short int * m, *n, sequences_db_max_length, query_sequences_max_length; 
	unsigned int * a_disp, * b_disp = NULL, offload_max_vect_sequences_db_count=0, qp_count, sp_count;
	unsigned int c, offload_vect_sequences_db_count;
	unsigned long int offload_max_vD=0, * chunk_accum_vect_sequences_db_count;

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

	// build query profile's
	queryProfiles = (char *)_mm_malloc(Q*BLOSUM_COLS*sizeof(char), 64);
	for (i=0; i<Q ; i++)
		memcpy(queryProfiles+i*BLOSUM_COLS,submat+a[i]*BLOSUM_COLS,BLOSUM_COLS*sizeof(char));

	// calculate number of query sequences that are processed with query and score profile
	i = 0;
	while ((i < query_sequences_count) && (query_sequences_lengths[i] < query_length_threshold))
		i++;
	qp_count = i;
	sp_count = query_sequences_count-qp_count;

	// Allocate memory for CPU buffers
	chunk_accum_vect_sequences_db_count = (unsigned long int *)_mm_malloc(chunk_count*sizeof(unsigned long int), 32);
	chunk_accum_vect_sequences_db_count[0] = 0;
	for (i=1; i<chunk_count ; i++)
		chunk_accum_vect_sequences_db_count[i] = chunk_accum_vect_sequences_db_count[i-1] + chunk_vect_sequences_db_count[i-1];

	tick = dwalltime();

	#pragma omp parallel default(none) shared(submat, a,queryProfiles,m,a_disp,query_sequences_count,b,n,b_disp,vect_sequences_db_count,scores, open_gap, extend_gap, qp_count,sp_count, chunk_b, chunk_n, chunk_b_disp, chunk_vD, query_sequences_max_length, sequences_db_max_length, Q, chunk_vect_sequences_db_count, chunk_count, offload_max_vD, offload_max_vect_sequences_db_count, query_length_threshold, chunk_accum_vect_sequences_db_count, mic_threads) num_threads(num_mics)
	{
		__declspec(align(64)) __m512i  *row_ptrs[MIC_MAX_NUM_THREADS]={NULL}, *maxCol_ptrs[MIC_MAX_NUM_THREADS]={NULL}, *maxRow_ptrs[MIC_MAX_NUM_THREADS]={NULL}, *lastCol_ptrs[MIC_MAX_NUM_THREADS]={NULL};
		__declspec(align(64)) char * scoreProfile_ptrs[MIC_MAX_NUM_THREADS]={NULL};
		unsigned long int offload_vD, scores_offset;
		unsigned int i, mic_chunks=0, * ptr_chunk_b_disp, offload_vect_sequences_db_count; 
		int mic_no, * mic_scores;
		unsigned short int * ptr_chunk_n;
		char * ptr_chunk_b;

		mic_no = omp_get_thread_num();
		mic_scores = (int*) _mm_malloc(query_sequences_count*(offload_max_vect_sequences_db_count*MIC_KNC_INT32_VECTOR_LENGTH)*sizeof(int), 64);

		// pre-allocate buffers and transfer common thread data to corresponding MIC
		#pragma offload_transfer target(mic:mic_no) in(submat: length(BLOSUM_ELEMS) ALLOC) in(queryProfiles: length(Q*BLOSUM_COLS) ALLOC) \
		in(a: length(Q) ALLOC) in(m:length(query_sequences_count) ALLOC) in(a_disp: length(query_sequences_count) ALLOC) \
		nocopy(b:length(offload_max_vD) ALLOC)  nocopy(n:length(offload_max_vect_sequences_db_count) ALLOC) nocopy(b_disp: length(offload_max_vect_sequences_db_count) ALLOC) \
		nocopy(mic_scores: length(query_sequences_count*offload_max_vect_sequences_db_count*MIC_KNC_INT32_VECTOR_LENGTH) ALLOC) \
		nocopy(row_ptrs, maxCol_ptrs, maxRow_ptrs, lastCol_ptrs, scoreProfile_ptrs: ALLOC)

		// distribute database chunk between MICs using dynamic scheduling
		#pragma omp for schedule(dynamic) nowait
		for (c=0; c < chunk_count ; c++) {

			offload_vD = chunk_vD[c];
			offload_vect_sequences_db_count = chunk_vect_sequences_db_count[c];
			ptr_chunk_b = chunk_b[c];
			ptr_chunk_n = chunk_n[c];
			ptr_chunk_b_disp = chunk_b_disp[c];
			scores_offset = chunk_accum_vect_sequences_db_count[c];

			// process database chunk in MIC 
			#pragma offload target(mic:mic_no) in(ptr_chunk_b[0:offload_vD] : into(b[0:offload_vD]) REUSE) \ 
				in(ptr_chunk_n[0:offload_vect_sequences_db_count] : into(n[0:offload_vect_sequences_db_count]) REUSE) \
				in(ptr_chunk_b_disp[0:offload_vect_sequences_db_count] : into(b_disp[0:offload_vect_sequences_db_count]) REUSE) \
				out(mic_scores: length(query_sequences_count*offload_vect_sequences_db_count*MIC_KNC_INT32_VECTOR_LENGTH) REUSE) \
				in(a: length(0) REUSE) in(m: length(0) REUSE) in(a_disp: length(0) REUSE) in(submat: length(0) REUSE) in(queryProfiles: length(0) REUSE) \
				nocopy(row_ptrs, maxCol_ptrs, maxRow_ptrs, lastCol_ptrs, scoreProfile_ptrs: REUSE) 
				#pragma omp parallel shared(c, mic_chunks, offload_vect_sequences_db_count, query_sequences_count, open_gap, extend_gap, query_sequences_max_length, sequences_db_max_length) num_threads(mic_threads) 
				{

					__m512i  *row, *maxCol, *maxRow, *lastCol;
					int  * ptr_scores;
					char * ptr_a, * ptr_b, *ptr_b_block, * scoreProfile, * queryProfile, * ptr_scoreProfile;

					__declspec(align(64)) __m512i vzero = _mm512_setzero_epi32(), score, previous, current, aux1, aux2, aux3, aux4, auxLastCol;
					__declspec(align(64)) __m512i vextend_gap = _mm512_set1_epi32(extend_gap), vopen_extend_gap = _mm512_set1_epi32(open_gap+extend_gap);
					__declspec(align(64)) __m512i v16 = _mm512_set1_epi32(16), submat_hi, submat_lo, b_values;
					__mmask16 mask;

					unsigned int tid, i, j, ii, jj, k, disp_1, disp_2, disp_3, dim, nbb;
					unsigned long int t, s, q; 

					tid = omp_get_thread_num();

					// if this is the first offload, allocate auxiliary buffers
					if (mic_chunks == 0)	{
						row_ptrs[tid] = (__m512i *) _mm_malloc((MIC_KNC_BLOCK_SIZE+1)*sizeof(__m512i), 64);
						maxCol_ptrs[tid] = (__m512i *) _mm_malloc((MIC_KNC_BLOCK_SIZE+1)*sizeof(__m512i), 64);
						maxRow_ptrs[tid] = (__m512i *) _mm_malloc((query_sequences_max_length)*sizeof(__m512i), 64);
						lastCol_ptrs[tid] = (__m512i *) _mm_malloc((query_sequences_max_length)*sizeof(__m512i), 64);
						if (query_sequences_max_length >= query_length_threshold)
							scoreProfile_ptrs[tid] = (char *) _mm_malloc(BLOSUM_ROWS_x_MIC_KNC_INT32_VECTOR_LENGTH*sequences_db_max_length*sizeof(char),16);
					}
					
					row = row_ptrs[tid];
					maxCol = maxCol_ptrs[tid];
					maxRow = maxRow_ptrs[tid];
					lastCol = lastCol_ptrs[tid];
					scoreProfile = scoreProfile_ptrs[tid];

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
								// store maxRow in auxiliar var
								aux2 = maxRow[i];

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
									current = _mm512_max_epi32(current, aux2);
									current = _mm512_max_epi32(current, maxCol[jj]);
									current = _mm512_max_epi32(current, vzero);
									// update maxRow and maxCol
									aux2 = _mm512_sub_epi32(aux2, vextend_gap);
									maxCol[jj] = _mm512_sub_epi32(maxCol[jj], vextend_gap);
									aux1 = _mm512_sub_epi32(current, vopen_extend_gap);
									aux2 = _mm512_max_epi32(aux2, aux1);
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
								// update maxRow
								maxRow[i] = aux2;

							}

						}

						// store max value
						
						_mm512_store_epi32(ptr_scores, score);
						

					}

					// calculate database chunk alignments with score profile technique
					#pragma omp for schedule(dynamic) nowait
					for (t=0; t< sp_count*offload_vect_sequences_db_count; t++) {

						q = qp_count + (sp_count-1) - (t % sp_count);
						s = (offload_vect_sequences_db_count-1) - (t / sp_count);

						ptr_a = a + a_disp[q];
						ptr_b = b + b_disp[s];
						ptr_scores = mic_scores + (q*offload_vect_sequences_db_count+s)*MIC_KNC_INT32_VECTOR_LENGTH;

						// build score profile
						disp_1 = n[s]*MIC_KNC_INT32_VECTOR_LENGTH;
						for (i=0; i<n[s] ;i++ ) {
							#if __MIC__
							aux1 = _mm512_extload_epi32(ptr_b+i*MIC_KNC_INT32_VECTOR_LENGTH, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0);
							#endif
							disp_2 = i*MIC_KNC_INT32_VECTOR_LENGTH;
							#pragma unroll(MIC_KNC_UNROLL_COUNT)
							for (j=0; j< BLOSUM_ROWS-1; j++) {
								#if __MIC__
								aux2 = _mm512_i32extgather_epi32(aux1, submat + j*BLOSUM_COLS, _MM_UPCONV_EPI32_SINT8  , 1, 0);
								_mm512_extstore_epi32(scoreProfile+disp_2+j*disp_1, aux2, _MM_DOWNCONV_EPI32_SINT8 , _MM_HINT_NONE );
								#endif

							}
						}

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

							for( i = 0; i < m[q]; i++){
						
								// previous must start in 0
								previous = _mm512_setzero_epi32();
								// update row[0] with lastCol elements
								row[0] = lastCol[i];
								// calculate i displacement
								ptr_scoreProfile = scoreProfile + ((int)(ptr_a[i]))*disp_1 + disp_2*MIC_KNC_INT32_VECTOR_LENGTH;
								// store maxRow in auxiliar var
								aux2 = maxRow[i];

								#pragma unroll(MIC_KNC_UNROLL_COUNT)
								for( jj=1; jj < dim+1; jj++) {
									//calcuate the diagonal value
									#if __MIC__
									current = _mm512_add_epi32(row[jj-1], _mm512_extload_epi32(ptr_scoreProfile+(jj-1)*MIC_KNC_INT32_VECTOR_LENGTH, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0));
									#endif							
									// calculate current max value
									current = _mm512_max_epi32(current, aux2);
									current = _mm512_max_epi32(current, maxCol[jj]);
									current = _mm512_max_epi32(current, vzero);
									// update maxRow and maxCol
									aux2 = _mm512_sub_epi32(aux2, vextend_gap);
									maxCol[jj] = _mm512_sub_epi32(maxCol[jj], vextend_gap);
									aux1 = _mm512_sub_epi32(current, vopen_extend_gap);
									aux2 = _mm512_max_epi32(aux2, aux1);
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
								// update maxRow
								maxRow[i] = aux2;

							}

						}

						// store max value
						
						_mm512_store_epi32(ptr_scores, score);
						

					}

					if (c == chunk_count-1)	{
						 _mm_free(row); _mm_free(maxCol); _mm_free(maxRow); _mm_free(lastCol);
 						if (query_sequences_max_length > query_length_threshold) _mm_free(scoreProfile);
					}
			
				}

				
				// copy scores from auxiliary buffer to final buffer
				for (i=0; i<query_sequences_count ; i++)
					memcpy(scores+(i*vect_sequences_db_count+scores_offset)*MIC_KNC_INT32_VECTOR_LENGTH,mic_scores+i*offload_vect_sequences_db_count*MIC_KNC_INT32_VECTOR_LENGTH,offload_vect_sequences_db_count*MIC_KNC_INT32_VECTOR_LENGTH*sizeof(int));
				mic_chunks++;

		}

		// de-allocate MIC buffers
		#pragma offload_transfer target(mic:mic_no) nocopy(submat: length(0) FREE) nocopy(queryProfiles: length(0) FREE) \
			nocopy(a: length(0) FREE) nocopy(m:length(0) FREE) nocopy(a_disp: length(0) FREE) \
			nocopy(b:length(0) FREE)  nocopy(n:length(0) FREE) nocopy(b_disp: length(0) FREE) \
			nocopy(mic_scores:length(0) FREE) \
			nocopy(row_ptrs, maxCol_ptrs, maxRow_ptrs, lastCol_ptrs, scoreProfile_ptrs: FREE)

		_mm_free(mic_scores); 

	}

	*workTime = dwalltime()-tick;

	_mm_free(chunk_accum_vect_sequences_db_count); _mm_free(queryProfiles);
	
}

// MIC search with KNC instructions and Adaptive Profile technique
void mic_search_knc_ap_single_chunk (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned int query_sequences_count, unsigned long int Q,
	unsigned int * query_disp, unsigned long int vect_sequences_db_count, char ** chunk_b, unsigned int chunk_count, unsigned int * chunk_vect_sequences_db_count,
	unsigned short int ** chunk_n, unsigned int ** chunk_b_disp, unsigned long int * chunk_vD, char * submat, int open_gap, int extend_gap,
	int num_mics, int mic_threads, int * scores, double * workTime, unsigned short int query_length_threshold){

	long int i=0, ii, j=0, k=0, jj;
	double tick;

	char *a, * b, *queryProfiles;
	unsigned short int * m, *n, sequences_db_max_length, query_sequences_max_length; 
	unsigned int * a_disp, * b_disp = NULL, offload_max_vect_sequences_db_count=0, qp_count, sp_count;
	unsigned int c, offload_vect_sequences_db_count;
	unsigned long int offload_vD;

	a = query_sequences;			b = chunk_b[0];				
	m = query_sequences_lengths;	n = chunk_n[0];				
	a_disp = query_disp;			b_disp = chunk_b_disp[0];	

	query_sequences_max_length = query_sequences_lengths[query_sequences_count-1];
	sequences_db_max_length = chunk_n[chunk_count-1][chunk_vect_sequences_db_count[chunk_count-1]-1];

	// build query profile's
	queryProfiles = (char *)_mm_malloc(Q*BLOSUM_COLS*sizeof(char), 64);
	for (i=0; i<Q ; i++)
		memcpy(queryProfiles+i*BLOSUM_COLS,submat+a[i]*BLOSUM_COLS,BLOSUM_COLS*sizeof(char));

	// calculate number of query sequences that are processed with query and score profile
	i = 0;
	while ((i < query_sequences_count) && (query_sequences_lengths[i] < query_length_threshold))
		i++;
	qp_count = i;
	sp_count = query_sequences_count-qp_count;

	tick = dwalltime();

			offload_vD = chunk_vD[0];
			offload_vect_sequences_db_count = chunk_vect_sequences_db_count[0];
			b = chunk_b[0];
			n = chunk_n[0];
			b_disp = chunk_b_disp[0];

			#pragma offload target(mic:0) in(submat: length(BLOSUM_ELEMS)) in(queryProfiles: length(Q*BLOSUM_COLS)) in(b: length(offload_vD)) \
				in(n: length(offload_vect_sequences_db_count)) in(b_disp: length(offload_vect_sequences_db_count)) \
				in(a: length(Q)) in(m: length(query_sequences_count)) in(a_disp: length(query_sequences_count)) \
				out(scores: length(query_sequences_count*offload_vect_sequences_db_count*MIC_KNC_INT32_VECTOR_LENGTH)) 
				#pragma omp parallel shared(offload_vect_sequences_db_count, query_sequences_count, open_gap, extend_gap, query_sequences_max_length, sequences_db_max_length) num_threads(mic_threads) 
				{

					__m512i  *row, *maxCol, *maxRow, *lastCol;
					int  * ptr_scores;
					char * ptr_a, * ptr_b, *ptr_b_block, * scoreProfile, *queryProfile, *ptr_scoreProfile;

					__declspec(align(64)) __m512i vzero = _mm512_setzero_epi32(), score, previous, current, aux1, aux2, aux3, aux4, auxLastCol;
					__declspec(align(64)) __m512i vextend_gap = _mm512_set1_epi32(extend_gap), vopen_extend_gap = _mm512_set1_epi32(open_gap+extend_gap);
					__declspec(align(64)) __m512i v16 = _mm512_set1_epi32(16), submat_hi, submat_lo, b_values;
					__mmask16 mask;

					unsigned int tid, i, j, ii, jj, k, disp_1, disp_2, disp_3, dim, nbb;
					unsigned long int t, s, q; 

					// allocate memory for auxiliary buffers
					row = (__m512i *) _mm_malloc((MIC_KNC_BLOCK_SIZE+1)*sizeof(__m512i), 64);
					maxCol = (__m512i *) _mm_malloc((MIC_KNC_BLOCK_SIZE+1)*sizeof(__m512i), 64);
					maxRow = (__m512i *) _mm_malloc((query_sequences_max_length)*sizeof(__m512i), 64);
					lastCol = (__m512i *) _mm_malloc((query_sequences_max_length)*sizeof(__m512i), 64);
					if (query_sequences_max_length >= query_length_threshold)
						scoreProfile = (char *) _mm_malloc(BLOSUM_ROWS_x_MIC_KNC_INT32_VECTOR_LENGTH*sequences_db_max_length*sizeof(char),16);

					// calculate chunk alignments using query profile technique
					#pragma omp for schedule(dynamic) nowait
					for (t=0; t< qp_count*offload_vect_sequences_db_count; t++) {

						q = (qp_count-1) - (t % qp_count);
						s = (offload_vect_sequences_db_count-1) - (t / qp_count);

						queryProfile = queryProfiles + a_disp[q]*BLOSUM_COLS;
						ptr_b = b + b_disp[s];
						ptr_scores = scores + (q*offload_vect_sequences_db_count+s)*MIC_KNC_INT32_VECTOR_LENGTH;

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
								// store maxRow in auxiliar var
								aux2 = maxRow[i];

								#pragma unroll(MIC_KNC_UNROLL_COUNT)
								for( jj=1; jj < dim+1; jj++) {
									//calcuate the diagonal value
									#if __MIC__
									b_values = _mm512_extload_epi32(ptr_b_block+(jj-1)*MIC_KNC_INT32_VECTOR_LENGTH, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0);
									#endif
									mask = _mm512_cmpge_epi32_mask(b_values,v16);
									aux1 = _mm512_permutevar_epi32(b_values, submat_lo);
									aux1 = _mm512_mask_permutevar_epi32(aux1, mask, b_values, submat_hi);
									current = _mm512_add_epi32(row[jj-1], aux1);								
									// calculate current max value
									current = _mm512_max_epi32(current, aux2);
									current = _mm512_max_epi32(current, maxCol[jj]);
									current = _mm512_max_epi32(current, vzero);
									// update maxRow and maxCol
									aux2 = _mm512_sub_epi32(aux2, vextend_gap);
									maxCol[jj] = _mm512_sub_epi32(maxCol[jj], vextend_gap);
									aux1 = _mm512_sub_epi32(current, vopen_extend_gap);
									aux2 = _mm512_max_epi32(aux2, aux1);
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
								// update maxRow
								maxRow[i] = aux2;

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
						ptr_scores = scores + (q*offload_vect_sequences_db_count+s)*MIC_KNC_INT32_VECTOR_LENGTH;

						// build score profile
						disp_1 = n[s]*MIC_KNC_INT32_VECTOR_LENGTH;
						for (i=0; i<n[s] ;i++ ) {
							#if __MIC__
							aux1 = _mm512_extload_epi32(ptr_b+i*MIC_KNC_INT32_VECTOR_LENGTH, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0);
							#endif
							disp_2 = i*MIC_KNC_INT32_VECTOR_LENGTH;
							#pragma unroll(MIC_KNC_UNROLL_COUNT)
							for (j=0; j< BLOSUM_ROWS-1; j++) {
								#if __MIC__
								aux2 = _mm512_i32extgather_epi32(aux1, submat + j*BLOSUM_COLS, _MM_UPCONV_EPI32_SINT8  , 1, 0);
								_mm512_extstore_epi32(scoreProfile+disp_2+j*disp_1, aux2, _MM_DOWNCONV_EPI32_SINT8 , _MM_HINT_NONE );
								#endif

							}
						}

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

							for( i = 0; i < m[q]; i++){
						
								// previous must start in 0
								previous = _mm512_setzero_epi32();
								// update row[0] with lastCol elements
								row[0] = lastCol[i];
								// calculate i displacement
								ptr_scoreProfile = scoreProfile + ((int)(ptr_a[i]))*disp_1 + disp_2*MIC_KNC_INT32_VECTOR_LENGTH;
								// store maxRow in auxiliar var
								aux2 = maxRow[i];

								#pragma unroll(MIC_KNC_UNROLL_COUNT)
								for( jj=1; jj < dim+1; jj++) {
									//calcuate the diagonal value
									#if __MIC__
									current = _mm512_add_epi32(row[jj-1], _mm512_extload_epi32(ptr_scoreProfile+(jj-1)*MIC_KNC_INT32_VECTOR_LENGTH, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0));
									#endif							
									// calculate current max value
									current = _mm512_max_epi32(current, aux2);
									current = _mm512_max_epi32(current, maxCol[jj]);
									current = _mm512_max_epi32(current, vzero);
									// update maxRow and maxCol
									aux2 = _mm512_sub_epi32(aux2, vextend_gap);
									maxCol[jj] = _mm512_sub_epi32(maxCol[jj], vextend_gap);
									aux1 = _mm512_sub_epi32(current, vopen_extend_gap);
									aux2 = _mm512_max_epi32(aux2, aux1);
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
								// update maxRow
								maxRow[i] = aux2;

							}

						}

						// store max value
						
						_mm512_store_epi32(ptr_scores, score);
						

					}

					_mm_free(row); _mm_free(maxCol); _mm_free(maxRow); _mm_free(lastCol);
 					if (query_sequences_max_length > query_length_threshold) _mm_free(scoreProfile);
				}

	*workTime = dwalltime()-tick;

	_mm_free(queryProfiles);
	
}