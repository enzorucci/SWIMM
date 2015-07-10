# SWIMM
Smith-Waterman implementation for Intel Multicore and Manycore architectures

## Description
SWIMM is a software to accelerate Smith-Waterman protein database search on Intel heterogeneous architectures based on Xeon and Xeon Phi processors. SWIMM takes advantage of SIMD computing through the use of SSE and AVX2 extensions on the Xeon and the KNC instruction set on the Xeon Phi. In addition, it offers three execution modes: (1) Xeon, (2) Xeon Phi and (3) concurrent Xeon and Xeon Phi. On a heterogeneous platform based on two Xeon E5-2695 v3 and a single Xeon Phi 3120P, SWIMM reaches up to 380 GCUPS on heterogeneous mode (AVX2+KNC), 350 GCUPS for the isolated Xeon (AVX2) and 50 GCUPS on Xeon Phi (KNC), while searching Environmental NR database.

## Usage
Databases must be preprocessed before searching it.

### Parameters
SWIMM execution

      -S <string> 'preprocess' for database preprocessing, 'search' for database search. [REQUIRED]

* preprocess
```
  -i,   --input=<string> Input sequence filename (must be in FASTA format). [REQUIRED]
  
  -o,   --output=<string> Output filename. [REQUIRED]
  
  -c,   --cpu_threads=<integer> Number of Xeon threads.
```

* search
```
  -q,   --query=<string> Input query sequence filename (must be in FASTA format). [REQUIRED]
  
  -d,   --db=<string> Preprocessed database output filename. [REQUIRED]
  
  -m,   --execution_mode=<integer> Execution mode: 0 for Xeon only, 1 for Xeon Phi only, 2 for concurrent Xeon and Xeon Phi (default: 2). [REQUIRED]
  
  -c,   --cpu_threads=<integer> Number of Xeon threads. Valid option only when execution mode is 0 or 2 (default: 4).
  
  -e,   --gap_extend=<integer> Gap extend penalty (default: 2).
  
  -g,   --gap_open=<integer> Gap open penalty (default: 10).

  -b,   --block_size=<integer>  Xeon block size. Valid option when execution mode is 0 or 2 (default: 256).

  -k,   --max_chunk_size=<integer> Maximum chunk size in bytes. Valid option only when execution mode is 1 or 2 (default: 134217728).
  
  -p,   --mic_profile=<char> Profile technique in Xeon Phi: ’Q’ for Query Profile, ’S’ for Score Profile, ’A’ for Adaptive Profile (default: A). Score Profile is always used in Xeon.
  
  -r,   --top=<integer> Number of scores to show (default: 10). 
  
  -s,   --sm=<string> Substitution matrix. Supported values: blosum45, blosum50, blosum62, blosum80, blosum90, pam30, pam70, pam250 (default: blosum62).
  
  -t,   --mic_threads=<integer> Number of Xeon Phi threads. Valid option only when execution mode is 1 or 2 (default: 240). 
  
  -u,   --query_length_threshold=<integer> Query length threshold. Valid option only when Adaptive Profile is used (default: 567).
  
  -v,   --vector_length=<integer> Vector length: 16 for Xeon Phi, 16 for Xeon with SSE support, 32 for Xeon with AVX2 support (default: 16).
  
  -x,   --num_mics=<integer> Number of Xeon Phis. Valid option only when execution mode is 1 or 2 (default: 1).
  
  -?,   --help Give this help list
        --usage Give a short usage message
```

### Examples

* Database preprocessing

  `./swimm -S preprocess -i db.fasta -o out `
  
  Preprocess *db.fasta* database using 4 CPU threads. The preprocessed database name will be *out*.
  
  `./swimm -S preprocess -i db.fasta -o out -c 8`
  
  Preprocess *db.fasta* database using 8 CPU threads. The preprocessed database name will be *out*.

* Database search


  `./swimm -S search -q query.fasta -d out -m 0 `
  
  Search query sequence *query.fasta* against *out* preprocessed database in Xeon mode with 4 CPU threads using SSE instruction set.
  
  `./swimm -S search -q query.fasta -d out -m 0 -c 16`
  
  Search query sequence *query.fasta* against *out* preprocessed database in Xeon mode with 16 CPU threads using SSE instruction set.
  
  `./swimm -S search -q query.fasta -d out -m 0 -c 16 -v 32`
  
  Search query sequence *query.fasta* against *out* preprocessed database in Xeon mode with 16 CPU threads using AVX2 instruction set.
  
    `./swimm -S search -q query.fasta -d out -m 0 -c 16 -b 128`
  
  Search query sequence *query.fasta* against *out* preprocessed database in Xeon mode with 16 CPU threads using SSE instruction set and block size equal to 128.
  
  `./swimm -S search -q query.fasta -d out -m 1`
  
  Search query sequence *query.fasta* against *out* preprocessed database in Xeon Phi mode using one accelerator with 240 threads.
  
  `./swimm -S search -q query.fasta -d out -m 1 -p S`
  
  Search query sequence *query.fasta* against *out* preprocessed database in Xeon Phi mode using one accelerator with 240 threads and score profile technique to get substitution matrix values.
  
  `./swimm -S search -q query.fasta -d out -m 1 -x 2`
  
  Search query sequence *query.fasta* against *out* preprocessed database in Xeon Phi mode using two accelerators with 240 threads each.
  
  `./swimm -S search -q query.fasta -d out -m 1 -x 2 -t 244`
  
  Search query sequence *query.fasta* against *out* preprocessed database in Xeon Phi mode using one accelerator with 244 threads each.
  
  `./swimm -S search -q query.fasta -d out -m 2 `
  
  Search query sequence *query.fasta* against *out* preprocessed database in concurrent Xeon and Xeon Phi mode with 4 CPU threads (SSE) and one single accelerator (KNC).
  
  `./swimm -S search -q query.fasta -d out -m 2 -v 32`
  
  Search query sequence *query.fasta* against *out* preprocessed database in concurrent Xeon and Xeon Phi mode with 4 CPU threads (AVX2) and one single accelerator (KNC).  
  
  `./swimm -S search -q query.fasta -d out -m 2 -k 67108864`
  
  Search query sequence *query.fasta* against *out* preprocessed database in concurrent Xeon and Xeon Phi mode with 4 CPU threads and one single accelerator. Divide database in chunks of 67108864 bytes (default: 134217728).
  
  `./swimm --help`
  
  `./swimm -?`
  
  Print help list.

### Importante notes
* Database and query files must be in FASTA format.
* Supported substitution matrixes: BLOSUM45, BLOSUM50, BLOSUM62, BLOSUM80, BLOSUM90, PAM30, PAM70 and PAM250. User-specific substitution matrix will be supported soon.
* For Xeon Phi, users can choose among query profile, score profile or adaptive profile. Because for shorter queries, query profile is better, we recommend using adaptive profile and tune the query length threshold (default: 567). For Xeon, score profile is always employed.
* Workload balance and data locality exploitation are critical to achieve good performance. Tune the chunk size and the Xeon block size with the *-k* and *-b* options, respectively.

## Reference
An Energy-aware Performance Analysis of SWIMM: Smith-Waterman Implementation on Intel's Multicore and Manycore architectures
Enzo Rucci, Carlos García, Guillermo Botella, Armando De Giusti, Marcelo Naiouf and Manuel Prieto-Matías
Concurrency and Computation: Practice and Experience
*In press*

## Changelog
* July 09, 2015 (v1.0.2)
Source code released
* January 11, 2015 (v1.0)
Binary code released

## Contact
If you have any question or suggestion, please contact Enzo Rucci (erucci [at] lidi.info.unlp.edu.ar)
