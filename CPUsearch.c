#include "submat.h"
#include "CPUsearch.h"
#include "utils.h"

// CPU search using SSE instrucions and Score Profile technique
void cpu_search_sse_sp (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned long int query_sequences_count, unsigned int * query_disp,
	char * vect_sequences_db, unsigned short int * vect_sequences_db_lengths, unsigned long int vect_sequences_db_count, unsigned long int * vect_sequences_db_disp,
	char * submat, int open_gap, int extend_gap, int n_threads, int cpu_block_size, int * scores, double * workTime){

	long int i, j, k;
	double tick;

	char *a, * b;
	unsigned int * a_disp;
	unsigned long int * b_disp = NULL;
	unsigned short int * m, *n, sequences_db_max_length, query_sequences_max_length; 

	a = query_sequences;
	m = query_sequences_lengths;
	a_disp = query_disp;

	query_sequences_max_length = query_sequences_lengths[query_sequences_count-1];
	sequences_db_max_length = vect_sequences_db_lengths[vect_sequences_db_count-1];

	b =  vect_sequences_db;
	n = vect_sequences_db_lengths;
	b_disp = vect_sequences_db_disp;

	tick = dwalltime();
	
	#pragma omp parallel default(none) shared(cpu_block_size, a, b, n, m, a_disp, b_disp, submat, scores, query_sequences_count, vect_sequences_db_count, open_gap, extend_gap, sequences_db_max_length, query_sequences_max_length) num_threads(n_threads) 
	{

		__m128i  *row, *maxCol, *maxRow, *lastCol, * ptr_scores, *tmp;
		char * ptr_a, * ptr_b, * scoreProfile, *ptr_scoreProfile;

		__declspec(align(32)) __m128i score, previous, current, auxBlosum[2], auxLastCol, b_values;
		__declspec(align(32)) __m128i aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
		__declspec(align(32)) __m128i vextend_gap_epi8 = _mm_set1_epi8(extend_gap), vopen_extend_gap_epi8 = _mm_set1_epi8(open_gap+extend_gap), vzero_epi8 = _mm_set1_epi8(0);
		__declspec(align(32)) __m128i vextend_gap_epi16 = _mm_set1_epi16(extend_gap), vopen_extend_gap_epi16 = _mm_set1_epi16(open_gap+extend_gap), vzero_epi16 = _mm_set1_epi16(0);
		__declspec(align(32)) __m128i vextend_gap_epi32 = _mm_set1_epi32(extend_gap), vopen_extend_gap_epi32 = _mm_set1_epi32(open_gap+extend_gap), vzero_epi32 = _mm_set1_epi32(0);
		__declspec(align(32)) __m128i v127 = _mm_set1_epi8(127), v32767 = _mm_set1_epi16(32767);
		__declspec(align(32)) __m128i v15 = _mm_set1_epi8(15), v16 = _mm_set1_epi8(16), vneg32 = _mm_set1_epi8(-32);

		unsigned int i, j, ii, jj, k, disp_1, disp_2, disp_3, disp_4, dim1, dim2, nbb;
		unsigned long int t, s, q; 
		int overflow_flag, bb1, bb1_start, bb1_end, bb2, bb2_start, bb2_end;

		// allocate memory for auxiliary buffers
		row = (__m128i *) _mm_malloc((cpu_block_size+1)*sizeof(__m128i), 32);
		maxCol = (__m128i *) _mm_malloc((cpu_block_size+1)*sizeof(__m128i), 32);
		maxRow = (__m128i *) _mm_malloc((query_sequences_max_length)*sizeof(__m128i), 32);
		lastCol = (__m128i *) _mm_malloc((query_sequences_max_length)*sizeof(__m128i), 32);
		scoreProfile = (char *) _mm_malloc((BLOSUM_ROWS_x_CPU_SSE_INT8_VECTOR_LENGTH*sequences_db_max_length)*sizeof(char), 32);
			
		// calculate alignment score
		#pragma omp for schedule(dynamic) nowait
		for (t=0; t< vect_sequences_db_count; t++) {

			s = (vect_sequences_db_count-1) - t;
			ptr_b = b + b_disp[s];

			// build score profile
			disp_1 = n[s]*CPU_SSE_INT8_VECTOR_LENGTH;
			for (i=0; i< n[s] ;i++ ) {
				disp_2 = i*CPU_SSE_INT8_VECTOR_LENGTH;
				// indexes
				b_values = _mm_loadu_si128((__m128i *) (ptr_b + disp_2));
				// indexes >= 16
				aux1 = _mm_sub_epi8(b_values, v16);
				// indexes < 16
				aux2 = _mm_cmpgt_epi8(b_values,v15);
				aux3 = _mm_and_si128(aux2,vneg32);
				aux4 = _mm_add_epi8(b_values,aux3);
				ptr_scoreProfile = scoreProfile + disp_2;
				for (j=0; j< BLOSUM_ROWS; j++) {
					disp_3 = j*disp_1;
					tmp = (__m128i *) (submat + j*BLOSUM_COLS);
					auxBlosum[0] = _mm_load_si128(tmp);
					auxBlosum[1] = _mm_load_si128(tmp+1);
					aux5  = _mm_shuffle_epi8(auxBlosum[0], aux4);			
					aux6  = _mm_shuffle_epi8(auxBlosum[1], aux1);			
					aux7 = _mm_add_epi8(aux5,  aux6);
					_mm_store_si128((__m128i*)(ptr_scoreProfile+disp_3),   aux7);
				}
			}

			// caluclate number of blocks
			nbb = ceil( (double) n[s] / (double) cpu_block_size);

			for (q=0; q<query_sequences_count; q++){

				ptr_a = a + a_disp[q];
				ptr_scores = (__m128i *) (scores + (q*vect_sequences_db_count+s)*CPU_SSE_INT8_VECTOR_LENGTH);

				// init buffers
				#pragma unroll(CPU_SSE_UNROLL_COUNT)
				for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm_set1_epi8(0);
				#pragma unroll(CPU_SSE_UNROLL_COUNT)
				for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm_set1_epi8(0);
				
				// set score to 0
				score = _mm_set1_epi8(0);
				// calculate a[i] displacement
				disp_1 = n[s]*CPU_SSE_INT8_VECTOR_LENGTH;

				for (k=0; k < nbb; k++){

					// calculate dim1
					disp_4 = k*cpu_block_size;
					dim1 = n[s]-disp_4;
					dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

					// init buffers
					#pragma unroll(CPU_SSE_UNROLL_COUNT)
					for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm_set1_epi8(0);
					#pragma unroll(CPU_SSE_UNROLL_COUNT)
					for (i=0; i<dim1+1 ; i++ ) row[i] = _mm_set1_epi8(0);
					auxLastCol = _mm_set1_epi8(0);

					for( i = 0; i < m[q]; i++){
				
						// previous must start in 0
						previous = _mm_set1_epi8(0);
						// update row[0] with lastCol[i-1]
						row[0] = lastCol[i];
						// calculate score profile displacement
						ptr_scoreProfile = scoreProfile+((int)(ptr_a[i]))*disp_1 + disp_4*CPU_SSE_INT8_VECTOR_LENGTH;
						// calculate dim2
						dim2 = dim1 / CPU_SSE_UNROLL_COUNT;

						for (ii=0; ii<dim2 ; ii++) {

							#pragma unroll(CPU_SSE_UNROLL_COUNT)
							for( j=ii*CPU_SSE_UNROLL_COUNT+1, jj=0; jj < CPU_SSE_UNROLL_COUNT; jj++, j++) {
								//calcuate the diagonal value
								current = _mm_adds_epi8(row[j-1], _mm_load_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_SSE_INT8_VECTOR_LENGTH)));
								// calculate current max value
								current = _mm_max_epi8(current, maxRow[i]);
								current = _mm_max_epi8(current, maxCol[j]);
								current = _mm_max_epi8(current, vzero_epi8);
								// update maxRow and maxCol
								maxRow[i] = _mm_subs_epi8(maxRow[i], vextend_gap_epi8);
								maxCol[j] = _mm_subs_epi8(maxCol[j], vextend_gap_epi8);
								aux0 = _mm_subs_epi8(current, vopen_extend_gap_epi8);
								maxRow[i] = _mm_max_epi8(maxRow[i], aux0);
								maxCol[j] =  _mm_max_epi8(maxCol[j], aux0);	
								// update max score
								score = _mm_max_epi8(score,current);
								// update row buffer
								row[j-1] = previous;
								previous = current;
							}
						}

						#pragma unroll
						for( j = dim2*CPU_SSE_UNROLL_COUNT+1; j < dim1+1; j++) {
							//calcuate the diagonal value
							current = _mm_adds_epi8(row[j-1], _mm_load_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_SSE_INT8_VECTOR_LENGTH)));
							// calculate current max value
							current = _mm_max_epi8(current, maxRow[i]);
							current = _mm_max_epi8(current, maxCol[j]);
							current = _mm_max_epi8(current, vzero_epi8);
							// update maxRow and maxCol
							maxRow[i] = _mm_subs_epi8(maxRow[i], vextend_gap_epi8);
							maxCol[j] = _mm_subs_epi8(maxCol[j], vextend_gap_epi8);
							aux0 = _mm_subs_epi8(current, vopen_extend_gap_epi8);
							maxRow[i] = _mm_max_epi8(maxRow[i], aux0);
							maxCol[j] =  _mm_max_epi8(maxCol[j], aux0);	
							// update max score
							score = _mm_max_epi8(score,current);
							// update row buffer
							row[j-1] = previous;
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

					// detect if overflow occurred in low-half, high-half or both halves
					aux1 = _mm_cmpeq_epi8(_mm_slli_si128(score,8),v127);
					bb1_start = _mm_test_all_zeros(aux1,v127);
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

							// calculate dim1
							disp_4 = k*cpu_block_size;
							dim1 = n[s]-disp_4;
							dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

							// init buffers
							#pragma unroll(CPU_SSE_UNROLL_COUNT)
							for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm_set1_epi16(0);
							#pragma unroll(CPU_SSE_UNROLL_COUNT)
							for (i=0; i<dim1+1 ; i++ ) row[i] = _mm_set1_epi16(0);
							auxLastCol = _mm_set1_epi16(0);

							for( i = 0; i < m[q]; i++){
						
								// previous must start in 0
								previous = _mm_set1_epi16(0);
								// update row[0] with lastCol[i-1]
								row[0] = lastCol[i];
								// calculate score profile displacement
								ptr_scoreProfile = scoreProfile+((int)(ptr_a[i]))*disp_1+disp_2+disp_4*CPU_SSE_INT8_VECTOR_LENGTH;
								// calculate dim2
								dim2 = dim1 / CPU_SSE_UNROLL_COUNT;

								for (ii=0; ii<dim2 ; ii++) {
									
									#pragma unroll(CPU_SSE_UNROLL_COUNT)
									for( j=ii*CPU_SSE_UNROLL_COUNT+1, jj=0; jj < CPU_SSE_UNROLL_COUNT;  jj++, j++) {
										//calcuate the diagonal value
										current = _mm_adds_epi16(row[j-1], _mm_cvtepi8_epi16(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_SSE_INT8_VECTOR_LENGTH))));
										// calculate current max value
										current = _mm_max_epi16(current, maxRow[i]);
										current = _mm_max_epi16(current, maxCol[j]);
										current = _mm_max_epi16(current, vzero_epi16);
										// update maxRow and maxCol
										maxRow[i] = _mm_subs_epi16(maxRow[i], vextend_gap_epi16);
										maxCol[j] = _mm_subs_epi16(maxCol[j], vextend_gap_epi16);
										aux0 = _mm_subs_epi16(current, vopen_extend_gap_epi16);
										maxRow[i] = _mm_max_epi16(maxRow[i], aux0);
										maxCol[j] =  _mm_max_epi16(maxCol[j], aux0);	
										// update row buffer
										row[j-1] = previous;
										previous = current;
										// update max score
										score = _mm_max_epi16(score,current);
									}
								}
								#pragma unroll
								for( j = dim2*CPU_SSE_UNROLL_COUNT+1; j < dim1+1; j++) {
									//calcuate the diagonal value
									current = _mm_adds_epi16(row[j-1], _mm_cvtepi8_epi16(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_SSE_INT8_VECTOR_LENGTH))));
									// calculate current max value
									current = _mm_max_epi16(current, maxRow[i]);
									current = _mm_max_epi16(current, maxCol[j]);
									current = _mm_max_epi16(current, vzero_epi16);
									// update maxRow and maxCol
									maxRow[i] = _mm_subs_epi16(maxRow[i], vextend_gap_epi16);
									maxCol[j] = _mm_subs_epi16(maxCol[j], vextend_gap_epi16);
									aux0 = _mm_subs_epi16(current, vopen_extend_gap_epi16);
									maxRow[i] = _mm_max_epi16(maxRow[i], aux0);
									maxCol[j] =  _mm_max_epi16(maxCol[j], aux0);	
									// update row buffer
									row[j-1] = previous;
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

						// detect if overflow occurred in low-half, high-half or both halves
							aux1 = _mm_cmpeq_epi16(_mm_slli_si128(score,8),v32767);
							bb2_start = _mm_test_all_zeros(aux1,v32767);
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

									// calculate dim1
									disp_4 = k*cpu_block_size;
									dim1 = n[s]-disp_4;
									dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

									// init buffers
									#pragma unroll(CPU_SSE_UNROLL_COUNT)
									for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm_set1_epi32(0);
									#pragma unroll(CPU_SSE_UNROLL_COUNT)
									for (i=0; i<dim1+1 ; i++ ) row[i] = _mm_set1_epi32(0);
									auxLastCol = _mm_set1_epi32(0);

									for( i = 0; i < m[q]; i++){
								
										// previous must start in 0
										previous = _mm_set1_epi32(0);
										// update row[0] with lastCol[i-1]
										row[0] = lastCol[i];
										// calculate score profile displacement
										ptr_scoreProfile = scoreProfile+((int)(ptr_a[i]))*disp_1+disp_3+disp_4*CPU_SSE_INT8_VECTOR_LENGTH;
										// calculate dim2
										dim2 = dim1 / CPU_SSE_UNROLL_COUNT;

										for (ii=0; ii<dim2 ; ii++) {

											#pragma unroll(CPU_SSE_UNROLL_COUNT)
											for( j=ii*CPU_SSE_UNROLL_COUNT+1, jj=0; jj < CPU_SSE_UNROLL_COUNT;  jj++, j++) {
												//calcuate the diagonal value
												current = _mm_add_epi32(row[j-1], _mm_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_SSE_INT8_VECTOR_LENGTH))));
												// calculate current max value
												current = _mm_max_epi32(current, maxRow[i]);
												current = _mm_max_epi32(current, maxCol[j]);
												current = _mm_max_epi32(current, vzero_epi32);
												// update maxRow and maxCol
												maxRow[i] = _mm_sub_epi32(maxRow[i], vextend_gap_epi32);
												maxCol[j] = _mm_sub_epi32(maxCol[j], vextend_gap_epi32);
												aux0 = _mm_sub_epi32(current, vopen_extend_gap_epi32);
												maxRow[i] = _mm_max_epi32(maxRow[i], aux0);
												maxCol[j] =  _mm_max_epi32(maxCol[j], aux0);	
												// update row buffer
												row[j-1] = previous;
												previous = current;
												// update max score
												score = _mm_max_epi32(score,current);
											}
										}
										#pragma unroll
										for( j = dim2*CPU_SSE_UNROLL_COUNT+1; j < dim1+1; j++) {
											//calcuate the diagonal value
											current = _mm_add_epi32(row[j-1], _mm_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_SSE_INT8_VECTOR_LENGTH))));
											// calculate current max value
											current = _mm_max_epi32(current, maxRow[i]);
											current = _mm_max_epi32(current, maxCol[j]);
											current = _mm_max_epi32(current, vzero_epi32);
											// update maxRow and maxCol
											maxRow[i] = _mm_sub_epi32(maxRow[i], vextend_gap_epi32);
											maxCol[j] = _mm_sub_epi32(maxCol[j], vextend_gap_epi32);
											aux0 = _mm_sub_epi32(current, vopen_extend_gap_epi32);
											maxRow[i] = _mm_max_epi32(maxRow[i], aux0);
											maxCol[j] =  _mm_max_epi32(maxCol[j], aux0);	
											// update row buffer
											row[j-1] = previous;
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

		 _mm_free(row); _mm_free(maxCol); _mm_free(maxRow); _mm_free(lastCol); _mm_free(scoreProfile);
	}

	*workTime = dwalltime()-tick;	
}

// CPU search using AVX2 instructions and Score Profile technique
void cpu_search_avx2_sp (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned long int query_sequences_count, unsigned int * query_disp,
	char * vect_sequences_db, unsigned short int * vect_sequences_db_lengths, unsigned long int vect_sequences_db_count, unsigned long int * vect_sequences_db_disp,
	char * submat, int open_gap, int extend_gap, int n_threads, int cpu_block_size, int * scores, double * workTime){

	long int i, j, k;
	double tick;

	char *a, * b;
	unsigned int  * a_disp;
	unsigned long int * b_disp = NULL;
	unsigned short int * m, *n, sequences_db_max_length, query_sequences_max_length; 

	a = query_sequences;
	m = query_sequences_lengths;
	a_disp = query_disp;

	query_sequences_max_length = query_sequences_lengths[query_sequences_count-1];
	sequences_db_max_length = vect_sequences_db_lengths[vect_sequences_db_count-1];

	b =  vect_sequences_db;
	n = vect_sequences_db_lengths;
	b_disp = vect_sequences_db_disp;

	tick = dwalltime();

	#pragma omp parallel default(none) shared(cpu_block_size, a, b, n, m, a_disp, b_disp, submat, scores, query_sequences_count, vect_sequences_db_count, open_gap, extend_gap, sequences_db_max_length, query_sequences_max_length) num_threads(n_threads) 
	{

		__m256i *row, *maxCol, *maxRow, *lastCol, * ptr_scores;
		__m128i *tmp;
		char * ptr_a, * ptr_b, * scoreProfile, *ptr_scoreProfile;


		__declspec(align(32)) __m256i score, previous, current, auxLastCol, b_values, blosum_lo, blosum_hi;
		__declspec(align(32)) __m256i aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
		__declspec(align(32)) __m256i vextend_gap_epi8 = _mm256_set1_epi8(extend_gap), vopen_extend_gap_epi8 = _mm256_set1_epi8(open_gap+extend_gap);
		__declspec(align(32)) __m256i vextend_gap_epi16 = _mm256_set1_epi16(extend_gap), vopen_extend_gap_epi16 = _mm256_set1_epi16(open_gap+extend_gap);
		__declspec(align(32)) __m256i vextend_gap_epi32 = _mm256_set1_epi32(extend_gap), vopen_extend_gap_epi32 = _mm256_set1_epi32(open_gap+extend_gap);
		__declspec(align(32)) __m256i vzero_epi8 = _mm256_set1_epi8(0), vzero_epi16 = _mm256_set1_epi16(0), vzero_epi32 = _mm256_set1_epi32(0);
		__declspec(align(32)) __m256i v15 = _mm256_set1_epi8(15), vneg32 = _mm256_set1_epi8(-32), v16 = _mm256_set1_epi8(16);
		__declspec(align(32)) __m256i v127 = _mm256_set1_epi8(127), v32767 = _mm256_set1_epi16(32767);
		__declspec(align(32)) __m128i aux, auxBlosum[2];

		unsigned int i, j, ii, jj, k, disp_1, disp_2, disp_3, disp_4, dim1, dim2, nbb;
		unsigned long int t, s, q; 
		int overflow_flag, bb1, bb2, bb1_start, bb1_end, bb2_start, bb2_end;

		// allocate memory for auxiliary buffers
		row = (__m256i *) _mm_malloc((cpu_block_size+1)*sizeof(__m256i), 32);
		maxCol = (__m256i *) _mm_malloc((cpu_block_size+1)*sizeof(__m256i), 32);
		maxRow = (__m256i *) _mm_malloc((query_sequences_max_length)*sizeof(__m256i), 32);
		lastCol = (__m256i *) _mm_malloc((query_sequences_max_length)*sizeof(__m256i), 32);
		scoreProfile = (char *) _mm_malloc((BLOSUM_ROWS_x_CPU_AVX2_INT8_VECTOR_LENGTH*sequences_db_max_length)*sizeof(char), 32);
			
		// calculate alignment score
		#pragma omp for schedule(dynamic) nowait
		for (t=0; t< vect_sequences_db_count; t++) {

			s = (vect_sequences_db_count-1) - t;
			ptr_b = b + b_disp[s];

			// build score profile
			disp_1 = n[s]*CPU_AVX2_INT8_VECTOR_LENGTH;
			for (i=0; i< n[s] ;i++ ) {
				disp_2 = i*CPU_AVX2_INT8_VECTOR_LENGTH;
				// indexes
				b_values =  _mm256_loadu_si256((__m256i *) (ptr_b + disp_2));
				// indexes >= 16
				aux1 = _mm256_sub_epi8(b_values, v16);
				// indexes < 16
				aux2 = _mm256_cmpgt_epi8(b_values,v15);
				aux3 = _mm256_and_si256(aux2,vneg32);
				aux4 = _mm256_add_epi8(b_values,aux3);
				ptr_scoreProfile = scoreProfile + disp_2;
				for (j=0; j< BLOSUM_ROWS; j++) {
					disp_2 = j*disp_1;
					disp_3 = j*BLOSUM_COLS;
					tmp = (__m128i*) (submat + disp_3);
					auxBlosum[0] = _mm_load_si128(tmp);
					auxBlosum[1] = _mm_load_si128(tmp+1);
					blosum_lo = _mm256_loadu2_m128i(&auxBlosum[0], &auxBlosum[0]);
					blosum_hi = _mm256_loadu2_m128i(&auxBlosum[1], &auxBlosum[1]);
					aux5 = _mm256_shuffle_epi8(blosum_lo,aux4);
					aux6 = _mm256_shuffle_epi8(blosum_hi,aux1);
					_mm256_store_si256((__m256i *)(ptr_scoreProfile+disp_2),_mm256_or_si256(aux5,aux6));
				}
			}

			// caluclate number of blocks
			nbb = ceil( (double) n[s] / (double) cpu_block_size);

			for (q=0; q<query_sequences_count; q++){

				ptr_a = a + a_disp[q];
				ptr_scores = (__m256i *) (scores + (q*vect_sequences_db_count+s)*CPU_AVX2_INT8_VECTOR_LENGTH);

				// init buffers
				#pragma unroll(CPU_AVX2_UNROLL_COUNT)
				for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm256_set1_epi8(0);
				#pragma unroll(CPU_AVX2_UNROLL_COUNT)
				for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm256_set1_epi8(0);
				
				// set score to 0
				score = _mm256_set1_epi8(0);
				// calculate a[i] displacement
				disp_1 = n[s]*CPU_AVX2_INT8_VECTOR_LENGTH;

				for (k=0; k < nbb; k++){

					// calculate dim1
					disp_4 = k*cpu_block_size;
					dim1 = n[s]-disp_4;
					dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

					// init buffers
					#pragma unroll(CPU_AVX2_UNROLL_COUNT)
					for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm256_set1_epi8(0);
					#pragma unroll(CPU_AVX2_UNROLL_COUNT)
					for (i=0; i<dim1+1 ; i++ ) row[i] = _mm256_set1_epi8(0);
					auxLastCol = _mm256_set1_epi8(0);

					for( i = 0; i < m[q]; i++){
				
						// previous must start in 0
						previous = _mm256_set1_epi8(0);
						// update row[0] with lastCol[i-1]
						row[0] = lastCol[i];
						// calculate score profile displacement
						ptr_scoreProfile = scoreProfile+((unsigned int)(ptr_a[i]))*disp_1 + disp_4*CPU_AVX2_INT8_VECTOR_LENGTH;
						// calculate dim2
						dim2 = dim1 / CPU_AVX2_UNROLL_COUNT;

						for (ii=0; ii<dim2 ; ii++) {

							#pragma unroll(CPU_AVX2_UNROLL_COUNT)
							for( j=ii*CPU_AVX2_UNROLL_COUNT+1, jj=0; jj < CPU_AVX2_UNROLL_COUNT;  jj++, j++) {
								//calcuate the diagonal value
								current =  _mm256_adds_epi8(row[j-1], _mm256_load_si256((__m256i *) (ptr_scoreProfile+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH)));
								// calculate current max value
								current = _mm256_max_epi8(current, maxRow[i]);
								current = _mm256_max_epi8(current, maxCol[j]);
								current = _mm256_max_epi8(current, vzero_epi8);
								// update maxRow and maxCol
								maxRow[i] =  _mm256_subs_epi8(maxRow[i], vextend_gap_epi8);
								maxCol[j] = _mm256_subs_epi8(maxCol[j], vextend_gap_epi8);
								aux0 =  _mm256_subs_epi8(current, vopen_extend_gap_epi8);
								maxRow[i] = _mm256_max_epi8(maxRow[i], aux0);
								maxCol[j] =  _mm256_max_epi8(maxCol[j], aux0);	
								// update row buffer
								row[j-1] = previous;
								previous = current;
								// update max score
								score = _mm256_max_epi8(score,current);
							}
						}
						#pragma unroll
						for( j = dim2*CPU_AVX2_UNROLL_COUNT+1; j < dim1+1; j++) {
							//calcuate the diagonal value
							current =  _mm256_adds_epi8(row[j-1], _mm256_load_si256((__m256i *) (ptr_scoreProfile+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH)));
							// calculate current max value
							current = _mm256_max_epi8(current, maxRow[i]);
							current = _mm256_max_epi8(current, maxCol[j]);
							current = _mm256_max_epi8(current, vzero_epi8);
							// update maxRow and maxCol
							maxRow[i] =  _mm256_subs_epi8(maxRow[i], vextend_gap_epi8);
							maxCol[j] = _mm256_subs_epi8(maxCol[j], vextend_gap_epi8);
							aux0 =  _mm256_subs_epi8(current, vopen_extend_gap_epi8);
							maxRow[i] = _mm256_max_epi8(maxRow[i], aux0);
							maxCol[j] =  _mm256_max_epi8(maxCol[j], aux0);	
							// update row buffer
							row[j-1] = previous;
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

							// calculate dim1
							disp_4 = k*cpu_block_size;
							dim1 = n[s]-disp_4;
							dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

							// init buffers
							#pragma unroll(CPU_AVX2_UNROLL_COUNT)
							for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm256_set1_epi16(0);
							#pragma unroll(CPU_AVX2_UNROLL_COUNT)
							for (i=0; i<dim1+1 ; i++ ) row[i] = _mm256_set1_epi16(0);
							auxLastCol = _mm256_set1_epi16(0);

							for( i = 0; i < m[q]; i++){
						
								// previous must start in 0
								previous = _mm256_set1_epi16(0);
								// update row[0] with lastCol[i-1]
								row[0] = lastCol[i];
								// calculate score profile displacement
								ptr_scoreProfile = scoreProfile+((int)(ptr_a[i]))*disp_1+disp_2+disp_4*CPU_AVX2_INT8_VECTOR_LENGTH;
								// calculate dim2
								dim2 = dim1 / CPU_AVX2_UNROLL_COUNT;

								for (ii=0; ii<dim2 ; ii++) {

									#pragma unroll(CPU_AVX2_UNROLL_COUNT)
									for( j=ii*CPU_AVX2_UNROLL_COUNT+1, jj=0; jj < CPU_AVX2_UNROLL_COUNT;  jj++, j++) {
										//calcuate the diagonal value
										current = _mm256_adds_epi16(row[j-1], _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH))));
										// calculate current max value
										current = _mm256_max_epi16(current, maxRow[i]);
										current = _mm256_max_epi16(current, maxCol[j]);
										current = _mm256_max_epi16(current, vzero_epi16);
										// update maxRow and maxCol
										maxRow[i] = _mm256_subs_epi16(maxRow[i], vextend_gap_epi16);
										maxCol[j] = _mm256_subs_epi16(maxCol[j], vextend_gap_epi16);
										aux0 = _mm256_subs_epi16(current, vopen_extend_gap_epi16);
										maxRow[i] = _mm256_max_epi16(maxRow[i], aux0);
										maxCol[j] =  _mm256_max_epi16(maxCol[j], aux0);	
										// update row buffer
										row[j-1] = previous;
										previous = current;
										// update max score
										score = _mm256_max_epi16(score,current);
									}
								}
								#pragma unroll
								for( j = dim2*CPU_AVX2_UNROLL_COUNT+1; j < dim1+1; j++) {
									//calcuate the diagonal value
									current = _mm256_adds_epi16(row[j-1], _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH))));
									// calculate current max value
									current = _mm256_max_epi16(current, maxRow[i]);
									current = _mm256_max_epi16(current, maxCol[j]);
									current = _mm256_max_epi16(current, vzero_epi16);
									// update maxRow and maxCol
									maxRow[i] = _mm256_subs_epi16(maxRow[i], vextend_gap_epi16);
									maxCol[j] = _mm256_subs_epi16(maxCol[j], vextend_gap_epi16);
									aux0 = _mm256_subs_epi16(current, vopen_extend_gap_epi16);
									maxRow[i] = _mm256_max_epi16(maxRow[i], aux0);
									maxCol[j] =  _mm256_max_epi16(maxCol[j], aux0);	
									// update row buffer
									row[j-1] = previous;
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

									// calculate dim1
									disp_4 = k*cpu_block_size;
									dim1 = n[s]-disp_4;
									dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

									// init buffers
									#pragma unroll(CPU_AVX2_UNROLL_COUNT)
									for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm256_set1_epi32(0);
									#pragma unroll(CPU_AVX2_UNROLL_COUNT)
									for (i=0; i<dim1+1 ; i++ ) row[i] = _mm256_set1_epi32(0);
									auxLastCol = _mm256_set1_epi32(0);

									for( i = 0; i < m[q]; i++){
								
										// previous must start in 0
										previous = _mm256_set1_epi32(0);
										// update row[0] with lastCol[i-1]
										row[0] = lastCol[i];
										// calculate score profile displacement
										ptr_scoreProfile = scoreProfile+((unsigned int)(ptr_a[i]))*disp_1+disp_3+disp_4*CPU_AVX2_INT8_VECTOR_LENGTH;
										// calculate dim2
										dim2 = dim1 / CPU_AVX2_UNROLL_COUNT;

										for (ii=0; ii<dim2 ; ii++) {

											#pragma unroll(CPU_AVX2_UNROLL_COUNT)
											for( j=ii*CPU_AVX2_UNROLL_COUNT+1, jj=0; jj < CPU_AVX2_UNROLL_COUNT;  jj++, j++) {
												//calcuate the diagonal value
												current = _mm256_add_epi32(row[j-1], _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH))));
												// calculate current max value
												current = _mm256_max_epi32(current, maxRow[i]);
												current = _mm256_max_epi32(current, maxCol[j]);
												current = _mm256_max_epi32(current, vzero_epi32);
												// update maxRow and maxCol
												maxRow[i] = _mm256_sub_epi32(maxRow[i], vextend_gap_epi32);
												maxCol[j] = _mm256_sub_epi32(maxCol[j], vextend_gap_epi32);
												aux0 = _mm256_sub_epi32(current, vopen_extend_gap_epi32);
												maxRow[i] = _mm256_max_epi32(maxRow[i], aux0);
												maxCol[j] =  _mm256_max_epi32(maxCol[j], aux0);	
												// update row buffer
												row[j-1] = previous;
												previous = current;
												// update max score
												score = _mm256_max_epi32(score,current);
											}
										}
										#pragma unroll
										for( j = dim2*CPU_AVX2_UNROLL_COUNT+1; j < dim1+1; j++) {
											//calcuate the diagonal value
											current = _mm256_add_epi32(row[j-1], _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH))));
											// calculate current max value
											current = _mm256_max_epi32(current, maxRow[i]);
											current = _mm256_max_epi32(current, maxCol[j]);
											current = _mm256_max_epi32(current, vzero_epi32);
											// update maxRow and maxCol
											maxRow[i] = _mm256_sub_epi32(maxRow[i], vextend_gap_epi32);
											maxCol[j] = _mm256_sub_epi32(maxCol[j], vextend_gap_epi32);
											aux0 = _mm256_sub_epi32(current, vopen_extend_gap_epi32);
											maxRow[i] = _mm256_max_epi32(maxRow[i], aux0);
											maxCol[j] =  _mm256_max_epi32(maxCol[j], aux0);	
											// update row buffer
											row[j-1] = previous;
											previous = current;
											// update max score
											score = _mm256_max_epi32(score,current);
										}

										// update lastCol
										lastCol[i] = auxLastCol;
										auxLastCol = current;
									}
								}
								// store max value
								_mm256_store_si256 (ptr_scores+bb1*2+bb2,score);
							}
						}

					}
				}

			}
		}

		 _mm_free(row); _mm_free(maxCol); _mm_free(maxRow); _mm_free(lastCol); _mm_free(scoreProfile);
	}

	*workTime = dwalltime()-tick;

}
