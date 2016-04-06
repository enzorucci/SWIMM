#include "sequences.h"

/* preprocess_db function preprocess the database sequences named input_filename. The preprocessed database filenames start with out_filename. */
void preprocess_db (char * input_filename, char * out_filename, int n_procs) {

	unsigned long int sequences_count=0, D=0, disp, accum, chunk_size, i, j, k;
	unsigned short int *sequences_lengths=NULL, * title_lengths=NULL, length=0, tmp_length, ok;
	char ** sequences=NULL, **titles=NULL, buffer[BUFFER_SIZE], filename[BUFFER_SIZE], * bin_filename, * res, *tmp_seq, *b=NULL, diff, new_line='\n';
	FILE * sequences_file, *titles_file, *info_file, * bin_file;
	int max_title_length;
	double tick= dwalltime();

	// open dabatase sequence filename 
	sequences_file = fopen(input_filename,"r");

	if (sequences_file == NULL)	{
		printf("SWIMM: An error occurred while opening input sequence file.\n");
		exit(2);
	}

	// Allocate memory for sequences_lengths array 
	sequences_lengths = (unsigned short int *) malloc (ALLOCATION_CHUNK*sizeof(unsigned short int));
	title_lengths = (unsigned short int *) malloc (ALLOCATION_CHUNK*sizeof(unsigned short int));

	// Calculate number of sequences in database and its lengths 
	sequences_count=0;

	res = fgets(buffer,BUFFER_SIZE,sequences_file);
	while (res != NULL) {
		length = 0;
		// read title
		while (strrchr(buffer,new_line) == NULL) {
			length += strlen(buffer);
			res = fgets(buffer,BUFFER_SIZE,sequences_file);
		}
		title_lengths[sequences_count] = length + strlen(buffer) + 1;
		// read sequence
		length = 0;
		res = fgets(buffer,BUFFER_SIZE,sequences_file);
		while ((res != NULL) && (buffer[0] != '>')) {
			length += strlen(buffer)-1;
			res = fgets(buffer,BUFFER_SIZE,sequences_file);
		}
		sequences_lengths[sequences_count] = length;
		(sequences_count)++;
		if ((sequences_count) % ALLOCATION_CHUNK == 0) {
			sequences_lengths = (unsigned short int *) realloc(sequences_lengths,((sequences_count)+ALLOCATION_CHUNK)*sizeof(unsigned short int));
			title_lengths = (unsigned short int *) realloc(title_lengths,((sequences_count)+ALLOCATION_CHUNK)*sizeof(unsigned short int));
		}
	}

	// Allocate memory for sequences array 
	sequences = (char **) malloc(sequences_count*sizeof(char *));
	if (sequences == NULL) { printf("SWIMM: An error occurred while allocating memory for sequences.\n"); exit(1); }
	for (i=0; i<sequences_count; i++ ) {
		sequences[i] = (char *) malloc(sequences_lengths[i]*sizeof(char));
		if (sequences[i] == NULL) { printf("SWIMM: An error occurred while allocating memory.\n"); exit(1); }
	}

	// Rewind sequences database file 
	rewind(sequences_file);

	// Read sequences from the database file and load them in sequences array 
	i = 0;
	res = fgets(buffer,BUFFER_SIZE,sequences_file);
	while (res != NULL) {
		// read title
		while (strrchr(buffer,new_line) == NULL)
			res = fgets(buffer,BUFFER_SIZE,sequences_file);
		// read sequence
		length = 1;
		res = fgets(buffer,BUFFER_SIZE,sequences_file);
		while ((res != NULL) && (buffer[0] != '>')) {
			//printf("%s %d\n",buffer,strlen(buffer));
			strncpy(sequences[i]+(length-1),buffer,strlen(buffer)-1);
			length += strlen(buffer)-1;
			res = fgets(buffer,BUFFER_SIZE,sequences_file);
		}
		i++;
	}

	// Rewind sequences database file 
	rewind(sequences_file);

	// Allocate memory for titles array 
	titles = (char **) malloc(sequences_count*sizeof(char *));
	if (titles == NULL) { printf("SWIMM: An error occurred while allocating memory for sequence titles.\n"); exit(1); }
	for (i=0; i<sequences_count; i++ ) {
		titles[i] = (char *) malloc(title_lengths[i]*sizeof(char));
		if (titles[i] == NULL) { printf("SWIMM: An error occurred while allocating memory for sequence titles.\n"); exit(1); }
	}

	// calculate max title length
	max_title_length = 0;
	for (i=0; i<sequences_count ; i++)
		max_title_length = (max_title_length > title_lengths[i] ? max_title_length : title_lengths[i]);

	// free memory
	free(title_lengths);

	// read sequence headers
	i = 0;
	res = fgets(buffer,BUFFER_SIZE,sequences_file);
	while (res != NULL) {
		// discard sequences
		while ((res != NULL) && (buffer[0] != '>')) 
			res = fgets(buffer,BUFFER_SIZE,sequences_file);
		if (res != NULL){
			// read header
			length = 1;
			do{
				strncpy(titles[i]+(length-1),buffer,strlen(buffer)-1);
				length += strlen(buffer)-1;
				res = fgets(buffer,BUFFER_SIZE,sequences_file);
			} while (strrchr(buffer,new_line) == NULL);
			titles[i][length] = '\0';
			i++;
		}
	}

	// Close sequences database file 
	fclose(sequences_file);

	// Sort sequence array by length 
	sort_sequences(sequences,titles,sequences_lengths, sequences_count, n_procs);

	// Create titles file: this text file contains the sequences description
	sprintf(filename,"%s.desc",out_filename);
	titles_file = fopen(filename,"w");

	if (titles_file == NULL)	{
		printf("SWIMM: An error occurred while opening sequence header file.\n");
		exit(2);
	}

	// write titles
	for (i=0; i<sequences_count ; i++)
		fprintf(titles_file,"%s\n",titles[i]);
	
	// close titles file
	fclose(titles_file);

	// calculate total number of residues
	#pragma omp parallel for reduction(+:D) num_threads(n_procs)
	for (i=0; i< sequences_count; i++ )
		D = D +  sequences_lengths[i];

	// transform bidimensional sequence array to a unidimensional one
	b = (char *) malloc(D*sizeof(char));	
	if (b == NULL) { printf("SWIMM: An error occurred while allocating memory for sequences.\n"); exit(1); }

	disp = 0;
	for (i=0; i< sequences_count; i++ ) {
		memcpy(b+disp,sequences[i],sequences_lengths[i]);
		disp += sequences_lengths[i];
	}

	// Free memory
	for (i=0; i< sequences_count; i++ ) 
		free(sequences[i]);
	free(sequences);

	// preprocess vect sequences DB
	// original alphabet: 'A'..'Z' => preprocessed alphabet: 0..24 (J, O and U are replaced with dummy symbol)
	#pragma omp parallel for private(diff) num_threads(n_procs) schedule(dynamic)	
	for (i=0; i< D; i++) {
		b[i] = ((b[i] == 'J') ? DUMMY_ELEMENT : b[i]);
		b[i] = ((b[i] == 'O') ? DUMMY_ELEMENT : b[i]);
		b[i] = ((b[i] == 'U') ? DUMMY_ELEMENT : b[i]);
		diff = 'A';
		diff = (b[i] > 'J' ? diff+1 : diff);
		diff = (b[i] > 'O' ? diff+1 : diff);
		diff = (b[i] > 'U' ? diff+1 : diff);
		b[i] -= diff;
	}

	// Create info file: this file contains sequences count, number of residues and the maximum title length
	sprintf(filename,"%s.info",out_filename);
	info_file = fopen(filename,"w");

	if (info_file == NULL)	{
		printf("SWIMM: An error occurred while opening info file.\n");
		exit(2);
	}

	// Write info
	fprintf(info_file,"%ld %ld %d",sequences_count,D,max_title_length);

	// close info file
	fclose(info_file);

	// Create sequences binary file: this file contains first the sequences lengths and then the preprocessed sequences residues
	sprintf(filename,"%s.seq",out_filename);
	bin_file = fopen(filename,"wb");

	if (bin_file == NULL)	{
		printf("SWIMM: An error occurred while opening sequence file.\n");
		exit(2);
	}

	// Write vectorized sequences lengths
	fwrite(sequences_lengths,sizeof(unsigned short int),sequences_count,bin_file);

	//Write sequences
	fwrite(b,sizeof(char),D,bin_file);

	// Close bin file
	fclose(bin_file);

	// free memory
	free(sequences_lengths);
	free(b);

	printf("\nSWIMM v1.1.1\n\n");
	printf("Database file:\t\t\t %s\n",input_filename); 
	printf("Database size:\t\t\t%ld sequences (%ld residues) \n",sequences_count,D);
	printf("Preprocessed database name:\t%s\n",out_filename); 
	printf("Preprocessing time:\t\t%lf seconds\n\n",dwalltime()-tick);

}

// Load query sequence from file in a
void load_query_sequences(char * queries_filename, int execution_mode, char ** ptr_query_sequences, char *** ptr_query_headers, unsigned short int **ptr_query_sequences_lengths,
						unsigned short int **ptr_m, unsigned long int * query_sequences_count, unsigned long int * ptr_Q, unsigned int ** ptr_query_sequences_disp, int n_procs) {

	long int i, j, k;
	unsigned long int sequences_count=0, Q=0, disp, accum, chunk_size;
	unsigned int * sequences_disp;
	unsigned short int *sequences_lengths, *m, * title_lengths, *tmp, length=0, tmp_length, ok;
	char ** sequences=NULL, **titles, buffer[BUFFER_SIZE], filename[BUFFER_SIZE], * bin_filename, * res, *tmp_seq, *a, diff, new_line='\n';
	FILE * sequences_file;

	// open query sequence filename 
	sequences_file = fopen(queries_filename,"r");

	if (sequences_file == NULL)	{
		printf("SWIMM: An error occurred while opening input sequence file.\n");
		exit(2);
	}

	// Allocate memory for sequences_lengths array 
	sequences_lengths = (unsigned short int *) malloc (ALLOCATION_CHUNK*sizeof(unsigned short int));
	title_lengths = (unsigned short int *) malloc (ALLOCATION_CHUNK*sizeof(unsigned short int));

	// Calculate number of sequences in database and its lengths 
	sequences_count=0;

	res = fgets(buffer,BUFFER_SIZE,sequences_file);
	while (res != NULL) {
		length = 0;
		// read title
		while (strrchr(buffer,new_line) == NULL) {
			length += strlen(buffer);
			res = fgets(buffer,BUFFER_SIZE,sequences_file);
		}
		title_lengths[sequences_count] = length + strlen(buffer) + 1;
		// read sequence
		length = 0;
		res = fgets(buffer,BUFFER_SIZE,sequences_file);
		while ((res != NULL) && (buffer[0] != '>')) {
			length += strlen(buffer)-1;
			res = fgets(buffer,BUFFER_SIZE,sequences_file);
		}
		sequences_lengths[sequences_count] = length;
		(sequences_count)++;
		if ((sequences_count) % ALLOCATION_CHUNK == 0) {
			sequences_lengths = (unsigned short int *) realloc(sequences_lengths,((sequences_count)+ALLOCATION_CHUNK)*sizeof(unsigned short int));
			title_lengths = (unsigned short int *) realloc(title_lengths,((sequences_count)+ALLOCATION_CHUNK)*sizeof(unsigned short int));
		}
	}

	// copy lengths to aligned buffer
	tmp = sequences_lengths;
	m = (unsigned short int *) _mm_malloc (sequences_count*sizeof(unsigned short int), (execution_mode == CPU_ONLY ? 32 : 64));
	sequences_lengths = (unsigned short int *) _mm_malloc (sequences_count*sizeof(unsigned short int), (execution_mode == CPU_ONLY ? 32 : 64));
	memcpy(m,tmp,sequences_count*sizeof(unsigned short int));
	memcpy(sequences_lengths,tmp,sequences_count*sizeof(unsigned short int));
	free(tmp);

	// Allocate memory for sequences array 
	sequences = (char **) malloc(sequences_count*sizeof(char *));
	if (sequences == NULL) { printf("SWIMM: An error occurred while allocating memory for query sequences.\n"); exit(1); }
	for (i=0; i<sequences_count; i++ ) {
		sequences[i] = (char *) malloc(sequences_lengths[i]*sizeof(char));
		if (sequences[i] == NULL) { printf("SWIMM: An error occurred while allocating memory.\n"); exit(1); }
	}

	// Rewind sequences database file 
	rewind(sequences_file);

	// Read sequences from the database file and load them in sequences array 
	i = 0;
	res = fgets(buffer,BUFFER_SIZE,sequences_file);
	while (res != NULL) {
		// read title
		while (strrchr(buffer,new_line) == NULL)
			res = fgets(buffer,BUFFER_SIZE,sequences_file);
		// read sequence
		length = 1;
		res = fgets(buffer,BUFFER_SIZE,sequences_file);
		while ((res != NULL) && (buffer[0] != '>')) {
			//printf("%s %d\n",buffer,strlen(buffer));
			strncpy(sequences[i]+(length-1),buffer,strlen(buffer)-1);
			length += strlen(buffer)-1;
			res = fgets(buffer,BUFFER_SIZE,sequences_file);
		}
		i++;
	}

	// Rewind sequences database file 
	rewind(sequences_file);

	// Allocate memory for titles array 
	titles = (char **) malloc(sequences_count*sizeof(char *));
	if (titles == NULL) { printf("SWIMM: An error occurred while allocating memory for sequence titles.\n"); exit(1); }
	for (i=0; i<sequences_count; i++ ) {
		titles[i] = (char *) malloc(title_lengths[i]*sizeof(char));
		if (titles[i] == NULL) { printf("SWIMM: An error occurred while allocating memory for sequence titles.\n"); exit(1); }
	}

	i = 0;
	res = fgets(buffer,BUFFER_SIZE,sequences_file);
	while (res != NULL) {
		// discard sequences
		while ((res != NULL) && (buffer[0] != '>')) 
			res = fgets(buffer,BUFFER_SIZE,sequences_file);
		if (res != NULL){
			// read header
			length = 1;
			do{
				strncpy(titles[i]+(length-1),buffer,strlen(buffer)-1);
				length += strlen(buffer)-1;
				res = fgets(buffer,BUFFER_SIZE,sequences_file);
			} while (strrchr(buffer,new_line) == NULL);
			titles[i][length] = '\0';
			i++;
		}
	}

	// Close sequences database file 
	fclose(sequences_file);

	// Sort sequence array by length 
	sort_sequences(sequences,titles,sequences_lengths, sequences_count, n_procs);
	
	// make sequences length even for CPU and Hybrid computing
	if (execution_mode == MIC_ONLY){

		// calculate total number of residues
		#pragma omp parallel for reduction(+:Q) num_threads(n_procs)
		for (i=0; i< sequences_count; i++ )
			Q = Q +  sequences_lengths[i];

		*ptr_Q = Q;

		a = (char *) _mm_malloc(Q*sizeof(char),  64);	
		if (a == NULL) { printf("SWIMM: An error occurred while allocating memory for sequences.\n"); exit(1); }

		disp = 0;
		for (i=0; i< sequences_count; i++ ) {
			// copy query sequence
			memcpy(a+disp,sequences[i],sequences_lengths[i]);
			disp += sequences_lengths[i];
		}
	} else {

		// calculate total number of residues
		#pragma omp parallel for reduction(+:Q) num_threads(n_procs)
		for (i=0; i< sequences_count; i++ )
			Q = Q +  sequences_lengths[i] + (sequences_lengths[i]%2);

		*ptr_Q = Q;

		a = (char *) _mm_malloc(Q*sizeof(char),  (execution_mode == CPU_ONLY ? 32 : 64));	
		if (a == NULL) { printf("SWIMM: An error occurred while allocating memory for sequences.\n"); exit(1); }

		disp = 0;
		for (i=0; i< sequences_count; i++ ) {
			// copy query sequence
			memcpy(a+disp,sequences[i],sequences_lengths[i]);
			// if length is odd then make it even and copy dummy element at last position
			if (sequences_lengths[i]%2==1){
				a[disp+sequences_lengths[i]]=DUMMY_ELEMENT;
				m[i]++;
			}
			disp += m[i];
		}
	}


	// process vect sequences DB
	#pragma omp parallel for private(diff) num_threads(n_procs) schedule(dynamic)	
	for (i=0; i< Q; i++) {
		a[i] = ((a[i] == 'J') ? DUMMY_ELEMENT : a[i]);
		a[i] = ((a[i] == 'O') ? DUMMY_ELEMENT : a[i]);
		a[i] = ((a[i] == 'U') ? DUMMY_ELEMENT : a[i]);
		diff = 'A';
		diff = (a[i] > 'J' ? diff+1 : diff);
		diff = (a[i] > 'O' ? diff+1 : diff);
		diff = (a[i] > 'U' ? diff+1 : diff);
		a[i] -= diff;
	}

	// Calculate displacement for current sequences db 
	sequences_disp = (unsigned int *) _mm_malloc((sequences_count+1)*sizeof(unsigned int),  (execution_mode == CPU_ONLY ? 32 : 64));

	sequences_disp[0] = 0;
	for (i=1; i < sequences_count+1; i++) 
		sequences_disp[i] = sequences_disp[i-1] + m[i-1];

	*ptr_query_sequences = a;
	*ptr_query_sequences_lengths = sequences_lengths;
	*ptr_m = m;
	*ptr_query_sequences_disp = sequences_disp;
	*ptr_query_headers = titles;
	*query_sequences_count = sequences_count;

	// Free memory
	for (i=0; i< sequences_count; i++ ) 
		free(sequences[i]);
	free(sequences);
	free(title_lengths);
}

void assemble_multiple_chunks_db (char * sequences_filename, int vector_length, unsigned long int max_chunk_size, unsigned long int * sequences_count,
				unsigned long int * D, unsigned short int * sequences_db_max_length, int * max_title_length, unsigned long int * vect_sequences_count, 
				unsigned long int * vD, char ***ptr_chunk_vect_sequences_db, unsigned int * chunk_count, unsigned int ** ptr_chunk_vect_sequences_db_count, 
				unsigned long int ** ptr_chunk_vD, unsigned short int *** ptr_chunk_vect_sequences_db_lengths, unsigned int *** ptr_chunk_vect_sequences_db_disp,
				int n_procs) {

	char ** sequences, *s, **chunk_vect_sequences_db, filename[200], * header, *b;
	unsigned short int ** chunk_vect_sequences_db_lengths, ** chunk_n, * sequences_lengths, * vect_sequences_lengths;
	unsigned long int i, ii, j, jj, k, * chunk_vD, accum, aux_vD=0, offset, chunk_size, * vect_sequences_disp;
	unsigned int * chunk_vect_sequences_count, **chunk_vect_sequences_disp, * tmp_chunk_vect_sequences_disp, c;
	FILE * sequences_file, * info_file;

	// Open info file
	sprintf(filename,"%s.info",sequences_filename);
	info_file = fopen(filename,"r");

	if (info_file == NULL)	{
		printf("SWIMM: An error occurred while opening info file.\n");
		exit(2);
	}

	fscanf(info_file,"%ld %ld %d",sequences_count,D,max_title_length);

	fclose(info_file);

	// Open sequences file
	sprintf(filename,"%s.seq",sequences_filename);
	sequences_file = fopen(filename,"r");

	if (sequences_file == NULL)	{
		printf("SWIMM: An error occurred while opening info file.\n");
		exit(2);
	}

	// Read sequences lengths
	sequences_lengths = (unsigned short int *) malloc((*sequences_count)*sizeof(unsigned short int));
	fread(sequences_lengths,sizeof(unsigned short int),*sequences_count,sequences_file);

	// Read sequences
	s = (char *) malloc((*D)*sizeof(char));
	fread(s,sizeof(char),*D,sequences_file);	

	fclose(sequences_file);

	sequences = (char **) malloc((*sequences_count)*sizeof(char *));

	sequences[0] = s;
	for (i=1; i<*sequences_count ; i++)
		sequences[i] = sequences[i-1] + sequences_lengths[i-1];

	// calculate vect_sequences_count
	*vect_sequences_count = ceil( (double) (*sequences_count) / (double) vector_length);

	// Allocate memory for vect_sequences_lengths
	vect_sequences_lengths = (unsigned short int *) malloc((*vect_sequences_count)*sizeof(unsigned short int));
	if (vect_sequences_lengths == NULL) { 		printf("SWIMM: An error occurred while allocating memory.\n"); exit(1); }
	vect_sequences_disp = (unsigned long int *) malloc((*vect_sequences_count+1)*sizeof(unsigned long int));
	if (vect_sequences_disp == NULL) { printf("SWIMM: An error occurred while allocating memory.\n"); exit(1); }

	// calculate values for vect_sequences_lengths array
	for (i=0; i< *vect_sequences_count - 1; i++ ) 
		vect_sequences_lengths[i] = sequences_lengths[(i+1)*vector_length-1];
	vect_sequences_lengths[*vect_sequences_count-1] = sequences_lengths[*sequences_count-1];

	// make length multiple of 4 to allow 32/64 bytes aligned data
	for (i=0; i< *vect_sequences_count; i++ ) 
		vect_sequences_lengths[i] = ceil( (double) vect_sequences_lengths[i] / (double) SEQ_LEN_MULT) * SEQ_LEN_MULT;

	// Calculate displacement for current sequences db 
	vect_sequences_disp[0] = 0;
	for (k=1; k < *vect_sequences_count+1; k++) 
		vect_sequences_disp[k] = vect_sequences_disp[k-1] + (vect_sequences_lengths[k-1]*vector_length);

	#pragma omp parallel for reduction(+:aux_vD) num_threads(n_procs)
	for (i=0; i< *vect_sequences_count; i++ )
		aux_vD = aux_vD + vect_sequences_lengths[i]*vector_length;

	*vD = aux_vD;

	b = (char *) _mm_malloc((*vD)*sizeof(char),16);

	// Copy sequences db to host buffers reordering elements to get better locality when computing alignments
	for (i=0; i < *vect_sequences_count-1; i++) {
		for (j=0; j< vect_sequences_lengths[i]; j++ ) {
			for (k=0;k< vector_length; k++)
				if (j < sequences_lengths[i*vector_length+k])
					*(b+vect_sequences_disp[i]+(j*vector_length)+k) = sequences[i*vector_length+k][j];
				else
					*(b+vect_sequences_disp[i]+(j*vector_length)+k) = PREPROCESSED_DUMMY_ELEMENT;
		}
	}
	//rest = sequences_count % vector_length;  
	for (i=*vect_sequences_count-1, j=0; j< vect_sequences_lengths[i]; j++ ) {
		for (k=0;k< vector_length; k++)
			if (i*vector_length+k < *sequences_count){
				if (j < sequences_lengths[i*vector_length+k])
					*(b+vect_sequences_disp[i]+(j*vector_length)+k) = sequences[i*vector_length+k][j];
				else
					*(b+vect_sequences_disp[i]+(j*vector_length)+k) = PREPROCESSED_DUMMY_ELEMENT;
			} else 
				*(b+vect_sequences_disp[i]+(j*vector_length)+k) = PREPROCESSED_DUMMY_ELEMENT;
	}

	// calculate chunks
	*chunk_count = 1;
	chunk_vect_sequences_count = (unsigned int *) malloc((*chunk_count)*sizeof(unsigned int));
	if (chunk_vect_sequences_count == NULL) { printf("SWIMM: An error occurred while allocating memory.\n"); exit(1); }

	i = 0;
	c = 0;
	while (i< *vect_sequences_count) {
		// group sequences till reach max chunk size
		j = 0;
		chunk_size = 0;
		accum = vect_sequences_lengths[i]*vector_length*sizeof(char) + sizeof(unsigned short int) + sizeof(unsigned int); // secuencias + longitud + desplazamiento
		while ((i< *vect_sequences_count) && (chunk_size <= max_chunk_size)) {
			chunk_size += accum;
			j++;
			i++; 
			if (i < *vect_sequences_count)
				accum = vect_sequences_lengths[i]*vector_length*sizeof(char) + sizeof(unsigned short int) + sizeof(unsigned int); // secuencias + longitud + desplazamiento
		}
		// number of sequences in chunk
		chunk_vect_sequences_count[c] = j;
		// increment chunk_count
		(*chunk_count)++;
		c++;
		chunk_vect_sequences_count = (unsigned int *) realloc(chunk_vect_sequences_count,(*chunk_count)*sizeof(unsigned int));
		if (chunk_vect_sequences_count == NULL) { printf("SWIMM: An error occurred while allocating memory.\n"); exit(1); }

	}
	// update chunk count
	(*chunk_count)--;

	// calculate chunk_vect_sequences_db_lengths
	chunk_vect_sequences_db_lengths = (unsigned short int **) _mm_malloc((*chunk_count)*sizeof(unsigned short int *),64);
	if (chunk_vect_sequences_db_lengths == NULL) { printf("SWIMM: An error occurred while allocating memory.\n"); exit(1); }

	offset = 0;
	for (i=0; i< *chunk_count ; i++) {
		chunk_vect_sequences_db_lengths[i] = (unsigned short int *) _mm_malloc((chunk_vect_sequences_count[i])*sizeof(unsigned short int),64);
		memcpy(chunk_vect_sequences_db_lengths[i],vect_sequences_lengths+offset,(chunk_vect_sequences_count[i])*sizeof(unsigned short int));
		offset += chunk_vect_sequences_count[i];
	}

	// calculate chunk_vect_sequences_db_disp
	accum = 0;
	chunk_vect_sequences_disp = (unsigned int **) _mm_malloc((*chunk_count)*sizeof(unsigned int *),64);
	for (i=0; i< *chunk_count ; i++){
		chunk_vect_sequences_disp[i] = (unsigned int *) _mm_malloc(chunk_vect_sequences_count[i]*sizeof(unsigned int),64);
		if (chunk_vect_sequences_disp[i] == NULL) { printf("SWIMM: An error occurred while allocating memory.\n"); exit(1); }
		// adapt sequence displacements to chunk
		offset = vect_sequences_disp[accum];
		for ( j=0, jj=accum; j<chunk_vect_sequences_count[i] ; j++, jj++)
			chunk_vect_sequences_disp[i][j] = (unsigned int)(vect_sequences_disp[jj] - offset);
		accum += chunk_vect_sequences_count[i];
	}

	// calculate chunk_vD
	chunk_vD = (unsigned long int *) malloc((*chunk_count)*sizeof(unsigned long int));
	if (chunk_vD == NULL) { printf("SWIMM: An error occurred while allocating memory.\n"); exit(1); }

	offset = 0;
	for (i=0; i< *chunk_count; i++){
		ii = offset + chunk_vect_sequences_count[i];
		chunk_vD[i] = vect_sequences_disp[ii] - vect_sequences_disp[offset];
		offset = ii;
	}

	// calculate chunk_vect_sequences_db
	chunk_vect_sequences_db = (char **) _mm_malloc((*chunk_count)*sizeof(char *), 64);
	if (chunk_vect_sequences_db == NULL) { printf("SWIMM: An error occurred while allocating memory.\n"); exit(1); }

	offset = 0;
	for (i=0; i< *chunk_count ; i++) {
		chunk_vect_sequences_db[i] = b + offset;
		offset += chunk_vD[i];
	}

	*ptr_chunk_vect_sequences_db = chunk_vect_sequences_db;
	*ptr_chunk_vect_sequences_db_count = chunk_vect_sequences_count;
	*ptr_chunk_vD = chunk_vD;
	*ptr_chunk_vect_sequences_db_lengths = chunk_vect_sequences_db_lengths;
	*ptr_chunk_vect_sequences_db_disp = chunk_vect_sequences_disp;
	*sequences_db_max_length = sequences_lengths[*sequences_count-1];

	free(s);
	free(sequences);
	free(sequences_lengths);
	free(vect_sequences_lengths);
	free(vect_sequences_disp);
}

void assemble_single_chunk_db (char * sequences_filename, int vector_length, unsigned long int * sequences_count,
				unsigned long int * D, unsigned short int * sequences_db_max_length, int * max_title_length, unsigned long int * vect_sequences_db_count,
				unsigned long int * vD, char **ptr_vect_sequences_db, unsigned short int ** ptr_vect_sequences_db_lengths, 
				unsigned long int ** ptr_vect_sequences_db_disp, int n_procs) {

	char ** sequences, *s, filename[200], ** sequences_db_headers, *header, *b;
	unsigned short int * vect_sequences_lengths, * sequences_lengths;
	unsigned long int i, j, k, accum, aux_vD=0, *vect_sequences_disp;
	FILE * sequences_file, * info_file;

	// Open info file
	sprintf(filename,"%s.info",sequences_filename);
	info_file = fopen(filename,"r");

	if (info_file == NULL)	{
		printf("SWIMM: An error occurred while opening info file.\n");
		exit(2);
	}

	fscanf(info_file,"%ld %ld %d",sequences_count,D,max_title_length);

	fclose(info_file);

	// Open sequences file
	sprintf(filename,"%s.seq",sequences_filename);
	sequences_file = fopen(filename,"rb");

	if (sequences_file == NULL)	{
		printf("SWIMM: An error occurred while opening info file.\n");
		exit(2);
	}

	// Read sequences lengths
	sequences_lengths = (unsigned short int *) malloc((*sequences_count)*sizeof(unsigned short int));
	fread(sequences_lengths,sizeof(unsigned short int),*sequences_count,sequences_file);

	// Read sequences
	s = (char *) malloc((*D)*sizeof(char));
	fread(s,sizeof(char),*D,sequences_file);	

	fclose(sequences_file);

	sequences = (char **) malloc((*sequences_count)*sizeof(char *));

	sequences[0] = s;
	for (i=1; i<*sequences_count ; i++)
		sequences[i] = sequences[i-1] + sequences_lengths[i-1];

	// calculate vect_sequences_count
	*vect_sequences_db_count = ceil( (double) (*sequences_count) / (double) vector_length);

	// Allocate memory for vect_sequences_lengths
	vect_sequences_lengths = (unsigned short int *) _mm_malloc((*vect_sequences_db_count)*sizeof(unsigned short int),32);
	if (vect_sequences_lengths == NULL) { printf("SWIMM: An error occurred while allocating memory.\n"); exit(1); }
	vect_sequences_disp = (unsigned long int *) _mm_malloc((*vect_sequences_db_count+1)*sizeof(unsigned long int),32);
	if (vect_sequences_disp == NULL) { printf("SWIMM: An error occurred while allocating memory.\n"); exit(1); }

	// calculate values for vect_sequences_lengths array
	for (i=0; i< *vect_sequences_db_count - 1; i++ ) 
		vect_sequences_lengths[i] = sequences_lengths[(i+1)*vector_length-1];
	vect_sequences_lengths[*vect_sequences_db_count-1] = sequences_lengths[*sequences_count-1];

	// make length multiple of 4 to allow 32/64 bytes aligned data
	for (i=0; i< *vect_sequences_db_count; i++ ) 
		vect_sequences_lengths[i] = ceil( (double) vect_sequences_lengths[i] / (double) SEQ_LEN_MULT) * SEQ_LEN_MULT;

	#pragma omp parallel for reduction(+:aux_vD) num_threads(n_procs)
	for (i=0; i< *vect_sequences_db_count; i++ )
		aux_vD = aux_vD + vect_sequences_lengths[i]*vector_length;

	*vD = aux_vD;

	b = (char *) _mm_malloc((*vD)*sizeof(char),32);

	// Calculate displacement for current sequences db 
	vect_sequences_disp[0] = 0;
	for (k=1; k < *vect_sequences_db_count+1; k++) 
		vect_sequences_disp[k] = vect_sequences_disp[k-1] + (vect_sequences_lengths[k-1]*vector_length);

	// Copy sequences db to host buffers reordering elements to get better locality when computing alignments
	for (i=0; i < *vect_sequences_db_count-1; i++) {
		for (j=0; j< vect_sequences_lengths[i]; j++ ) {
			for (k=0;k< vector_length; k++)
				if (j < sequences_lengths[i*vector_length+k])
					*(b+vect_sequences_disp[i]+(j*vector_length)+k) = sequences[i*vector_length+k][j];
				else
					*(b+vect_sequences_disp[i]+(j*vector_length)+k) = PREPROCESSED_DUMMY_ELEMENT;
		}
	}
	//rest = sequences_count % vector_length;  
	for (i=*vect_sequences_db_count-1, j=0; j< vect_sequences_lengths[i]; j++ ) {
		for (k=0;k< vector_length; k++)
			if (i*vector_length+k < *sequences_count){
				if (j < sequences_lengths[i*vector_length+k])
					*(b+vect_sequences_disp[i]+(j*vector_length)+k) = sequences[i*vector_length+k][j];
				else
					*(b+vect_sequences_disp[i]+(j*vector_length)+k) = PREPROCESSED_DUMMY_ELEMENT;
			} else 
				*(b+vect_sequences_disp[i]+(j*vector_length)+k) = PREPROCESSED_DUMMY_ELEMENT;
	}

	*ptr_vect_sequences_db = b;
	*ptr_vect_sequences_db_lengths = vect_sequences_lengths;
	*ptr_vect_sequences_db_disp = vect_sequences_disp;
	*sequences_db_max_length = sequences_lengths[*sequences_count-1];

	free(s);
	free(sequences);
	free(sequences_lengths);
}

void load_database_headers (char * sequences_filename, unsigned long int sequences_count, int max_title_length, char *** ptr_sequences_db_headers) {

	char ** sequences_db_headers, filename[200], * header;
	FILE * header_file;
	unsigned long int i;

	// Load sequence headers

	// Open header file
	sprintf(filename,"%s.desc",sequences_filename);
	header_file = fopen(filename,"r");

	if (header_file == NULL)	{
		printf("SWIMM: An error occurred while opening sequence description file.\n");
		exit(3);
	}

	// Read sequences lengths
	sequences_db_headers = (char **) malloc(sequences_count*sizeof(char *));
	header = (char *) malloc((max_title_length+1)*sizeof(char));

	for (i=0; i<sequences_count; i++){
		fgets(header,max_title_length,header_file);
		sequences_db_headers[i] = (char *) malloc((strlen(header)+1)*sizeof(char));
		strcpy(sequences_db_headers[i],header);
	}

	fclose(header_file);
	free(header);

	*ptr_sequences_db_headers = sequences_db_headers;
}


void merge_sequences(char ** sequences, char ** titles, unsigned short int * sequences_lengths, unsigned long int size) {
	unsigned long int i1 = 0;
	unsigned long int i2 = size / 2;
	unsigned long int it = 0;
	// allocate memory for temporary buffers
	char ** tmp1 = (char **) malloc(size*sizeof(char *));
	char ** tmp2 = (char **) malloc(size*sizeof(char *));
	unsigned short int * tmp3 = (unsigned short int *) malloc (size*sizeof(unsigned short int));

	while(i1 < size/2 && i2 < size) {
		if (sequences_lengths[i1] <= sequences_lengths[i2]) {
			tmp1[it] = sequences[i1];
			tmp2[it] = titles[i1];
			tmp3[it] = sequences_lengths[i1];
			i1++;
		}
		else {
			tmp1[it] = sequences[i2];
			tmp2[it] = titles[i2];
			tmp3[it] = sequences_lengths[i2];
			i2 ++;
		}
		it ++;
	}

	while (i1 < size/2) {
		tmp1[it] = sequences[i1];
		tmp2[it] = titles[i1];
		tmp3[it] = sequences_lengths[i1];
	    i1++;
	    it++;
	}
	while (i2 < size) {
		tmp1[it] = sequences[i2];
		tmp2[it] = titles[i2];
		tmp3[it] = sequences_lengths[i2];
	    i2++;
	    it++;
	}

	memcpy(sequences, tmp1, size*sizeof(char *));
	memcpy(titles, tmp2, size*sizeof(char *));
	memcpy(sequences_lengths, tmp3, size*sizeof(unsigned short int));

	free(tmp1);
	free(tmp2);
	free(tmp3);

}


void mergesort_sequences_serial (char ** sequences, char ** titles, unsigned short int * sequences_lengths, unsigned long int size) {
	char * tmp_seq;
	unsigned short int tmp_seq_len;

	if (size == 2) { 
		if (sequences_lengths[0] > sequences_lengths[1]) {
			// swap sequences
			tmp_seq = sequences[0];
			sequences[0] = sequences[1];
			sequences[1] = tmp_seq;
			// swap titles
			tmp_seq = titles[0];
			titles[0] = titles[1];
			titles[1] = tmp_seq;
			// swap sequences lengths
			tmp_seq_len = sequences_lengths[0];
			sequences_lengths[0] = sequences_lengths[1];
			sequences_lengths[1] = tmp_seq_len;
			return;
		}
	} else {
		if (size > 2){
			mergesort_sequences_serial(sequences, titles, sequences_lengths, size/2);
			mergesort_sequences_serial(sequences + size/2, titles + size/2, sequences_lengths + size/2, size - size/2);
			merge_sequences(sequences, titles, sequences_lengths, size);
		}
	}
}

void sort_sequences (char ** sequences, char ** titles, unsigned short int * sequences_lengths, unsigned long int size, int threads) {
    if ( threads == 1) {
	      mergesort_sequences_serial(sequences, titles, sequences_lengths, size);
    }
    else if (threads > 1) {
        #pragma omp parallel sections
        {
            #pragma omp section
            sort_sequences(sequences, titles, sequences_lengths, size/2, threads/2);
            #pragma omp section
            sort_sequences(sequences + size/2, titles  + size/2, sequences_lengths + size/2, size-size/2, threads-threads/2);
        }

        merge_sequences(sequences, titles, sequences_lengths, size);
    } // threads > 1
}
