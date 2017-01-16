#include "arguments.h"
#include "sequences.h"

const char *argp_program_bug_address =
  "<erucci@lidi.info.unlp.edu.ar>";

/* Program documentation. */
static char doc[] =
  "\nSWIMM is a software to accelerate Smith-Waterman protein database search on Intel heterogeneous architectures based on Xeon and Xeon Phi processors";

void program_arguments_processing (int argc, char * argv[]) {

	int i, result, arg_count=4;

	struct argp_option options[] =	{
		{ 0, 0, 0, 0, "SWIMM execution", 1},
		{ 0, 'S', "<string>", 0, "'preprocess' for database preprocessing, 'search' for database search. [REQUIRED]", 1},
		{ 0, 0, 0, 0, "preprocess", 2},
		{ "input", 'i', "<string>", 0, "Input sequence filename (must be in FASTA format). [REQUIRED]", 2},
		{ "output", 'o', "<string>", 0, "Output filename. [REQUIRED]", 2},
		{ 0, 0, 0, 0, "search", 3},
		{ "query", 'q', "<string>", 0,  "Input query sequence filename (must be in FASTA format). [REQUIRED]", 3},
		{ "db", 'd', "<string>", 0, "Preprocessed database output filename. [REQUIRED]", 3},
		{ "sm", 's', "<string>", 0, "Substitution matrix. Supported values: blosum45, blosum50, blosum62, blosum80, blosum90, pam30, pam70, pam250 (default: blosum62).", 3},
		{ "gap_open", 'g', "<integer>", 0, "Gap open penalty (default: 10).", 3},
		{ "gap_extend", 'e', "<integer>", 0, "Gap extend penalty (default: 2).", 3},
		{ "execution_mode", 'm', "<integer>", 0, "Execution mode: 0 for Xeon only, 1 for Xeon Phi only, 2 for Xeon+Xeon Phi (default: 2). [REQUIRED]", 3},
		{ "cpu_threads", 'c', "<integer>", 0, "Number of Xeon threads. Valid option only when execution mode is 0 or 2 (default: 4).", 3},
		{ "num_mics", 'x', "<integer>", 0, "Number of Xeon Phis. Valid option only when execution mode is 1 or 2 (default: 1).", 3},
		{ "mic_threads", 't', "<integer>", 0, "Number of Xeon Phi threads. Valid option only when execution mode is 1 or 2 (default: 240).", 3},
		{ "mic_profile", 'p', "<char>", 0, "Profile technique in Xeon Phi: 'Q' for Query Profile, 'S' for Score Profile, 'A' for Adaptive Profile (default: A). Score Profile is always used in Xeon.", 3},
		{ "query_length_threshold", 'u', "<integer>", 0, "Query length threshold. Valid option only when Adaptive Profile is used (default: 567).", 3},
		{ "vector_length", 'v', "<integer>", 0, "Vector length: 16 for Xeon Phi, 16 for Xeon with SSE support, 32 for Xeon with AVX2 support (default: 16).", 3},
		{ "top", 'r', "<integer>", 0, "Number of scores to show (default: 10).", 3},
		{ "max_chunk_size", 'k', "<integer>", 0, "Maximum chunk size in bytes. Valid option only when execution mode is 1 or 2 (default: 100663296).", 3},
		{ "block_size", 'b', "<integer>", 0, "Xeon block size. Valid option when execution mode is 0 or 2 (default: 120).", 3},
		{ 0 }
	};

	struct argp argp = { options, parse_opt, 0, doc};
	result = argp_parse (&argp, argc, argv, 0, 0, &arg_count); 
}

static int parse_opt (int key, char *arg, struct argp_state *state) {

	int *arg_count = state->input;

	switch(key) {
		case 'S': 
			if ((strcmp(arg,"preprocess") != 0) && (strcmp(arg,"search")))
				argp_failure (state, 1, 0, "%s is not a valid option for execution.",arg);
			else
				op = arg;
			break;
		case 'i':
			input_filename = arg;
			break;
		case 'o':
			output_filename = arg;
			break;
		case 'm':
			execution_mode = atoi(arg);
			if ((execution_mode < CPU_ONLY) || (execution_mode > HETEROGENEOUS))
				argp_failure (state, 1, 0, "%d is not a valid option for execution mode.",execution_mode);
			break;
		case 'q':
			queries_filename = arg;
			break;
		case 'd':
			sequences_filename = arg;
			break;
		case 's':
			if ((strcmp(arg,"blosum45") != 0) && (strcmp(arg,"blosum50") != 0) && (strcmp(arg,"blosum62") != 0) && (strcmp(arg,"blosum80") != 0) && (strcmp(arg,"blosum90") != 0) && (strcmp(arg,"pam30") != 0) && (strcmp(arg,"pam70") != 0) && (strcmp(arg,"pam250") != 0))
				argp_failure (state, 1, 0, "%s is not a valid option for substitution matrix.",arg);
			else {
				if (strcmp(arg,"blosum45") == 0) { submat = blosum45; strcpy(submat_name,"BLOSUM45"); }
				if (strcmp(arg,"blosum50") == 0) { submat = blosum50; strcpy(submat_name,"BLOSUM50"); }
				if (strcmp(arg,"blosum62") == 0) { submat = blosum62; strcpy(submat_name,"BLOSUM62"); }
				if (strcmp(arg,"blosum80") == 0) { submat = blosum80;  strcpy(submat_name,"BLOSUM80"); }
				if (strcmp(arg,"blosum90") == 0) { submat = blosum90; strcpy(submat_name,"BLOSUM90"); }
				if (strcmp(arg,"pam30") == 0) { submat = pam30;  strcpy(submat_name,"PAM30"); }
				if (strcmp(arg,"pam70") == 0) {	submat = pam70; strcpy(submat_name,"PAM70"); }
				if (strcmp(arg,"pam250") == 0) { submat = pam250;  strcpy(submat_name,"PAM250"); }
			}
			break;
		case 'g':
			open_gap = atoi(arg);
			if ((open_gap < 0) || (open_gap > 127))
				argp_failure (state, 1, 0, "%s is not a valid option for gap open penalty.",open_gap);
			break;
		case 'e':
			extend_gap = atoi(arg);
			if ((extend_gap < 0) || (extend_gap > 127))
				argp_failure (state, 1, 0, "%s is not a valid option for gap extend penalty.",extend_gap);
			break;
		case 'c':
			cpu_threads = atoi(arg);
			if (cpu_threads < 0)
				argp_failure (state, 1, 0, "The number of Xeon threads must be greater than 0.");
			break;
		case 'x':
			num_mics = atoi(arg);
			if (num_mics < 0)
				argp_failure (state, 1, 0, "The number of Xeon Phis must be greater than 0.");
			break;
		case 't':
			mic_threads = atoi(arg);
			if (mic_threads < 0)
				argp_failure (state, 1, 0, "The number of Xeon Phi threads must be greater than 0.");
			break;
		case 'p':
			if ((strcmp(arg,"Q") != 0) && (strcmp(arg,"S") != 0) && (strcmp(arg,"A") != 0))
				argp_failure (state, 1, 0, "%s is not a valid option for profile technique.",arg);
			else
				profile = arg[0];
			break;
		case 'u':
			query_length_threshold = atoi(arg);
			if ((query_length_threshold < 0) || (query_length_threshold > 65535))
				argp_failure (state, 1, 0, "%s is not a valid option for query length threshold.",query_length_threshold);
			break;
		case 'v':
			vector_length = atoi(arg);
			if ((vector_length != 16) && (vector_length != 32))
				argp_failure (state, 1, 0, "%d is not a valid option for vector length.",vector_length);
			break;
		case 'r':
			top = atoi(arg);
			if (top < 0)
				argp_failure (state, 1, 0, "The number of scores to show must be greater than 0.");
			break;
		case 'k':
			max_chunk_size = atol(arg);
			if (max_chunk_size < 0)
				argp_failure (state, 1, 0, "The maximum chuk size must be greater than 0.");
			break;
		case 'b':
			cpu_block_size = atoi(arg);
			cpu_block_size = ((cpu_block_size % SEQ_LEN_MULT != 0) ? (cpu_block_size/SEQ_LEN_MULT)*SEQ_LEN_MULT : cpu_block_size);
			if (cpu_block_size < 0)
				argp_failure (state, 1, 0, "The Xeon block size must be greater than 0.");
			break;
		case ARGP_KEY_END:
			if (*arg_count == 1)
				argp_failure (state, 1, 0, "Missing options");
			if (op == NULL)
				argp_failure (state, 1, 0, "SWIMM execution option is required");
			else 
				if (strcmp(op,"preprocess") == 0){
					if (input_filename == NULL)
						argp_failure (state, 1, 0, "Input sequence filename is required");
					if (output_filename == NULL)
						argp_failure (state, 1, 0, "Output filename is required");
				} else {
					if (sequences_filename == NULL)
						argp_failure (state, 1, 0, "Database filename is required");
					if (queries_filename == NULL)
						argp_failure (state, 1, 0, "Query sequences filename is required");
					if (((profile == 'A') || (profile == 'Q')) && (execution_mode == CPU_ONLY))
						argp_failure (state, 1, 0, "Adaptive Profile and Query Profile are not valid for Xeon.");
					if ((execution_mode == MIC_ONLY) && (vector_length == 32))
						argp_failure (state, 1, 0, "Xeon Phi only supports vector length equal to 16.");
					if (profile == 0)
						profile = ADAPTIVE_PROFILE;
					
				}
/*	    default:
			return ARGP_ERR_UNKNOWN;
*/	}

	return 0;
} 

