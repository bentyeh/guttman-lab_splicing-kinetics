# Repository description

- `data/`
  - `junctionFraction.matrix`: tab-delimited table of spliced fractions
    - Author: Mitchell Guttman
    - Source code: missing
    - Columns
      - `Gene`: gene symbol based on MGI / GENCODE
      - `Junction`: intron coordinates
      - `RPKM`: RPKM for the whole gene at time point 0 min.; this file has been filtered for genes with RPKM > 5
      - `Counts`: total number of reads spanning the splice junction across all time points
      - `0`, ..., `240`: spliced fraction at each time point
          $$\text{Spliced fraction} = \frac{\text{spliced}}{\text{spliced} + \text{unspliced}}$$
          where $\text{unspliced}$ is the number of reads aligning to any part of intron divided by the intron length, and $\text{spliced}$ is the number of reads spanning the splice junction. A value of -1 means that there were 0 reads (spliced or unspliced) at that time point for that intron/junction.
  - `moleculecount.xlsx`: Excel spreadsheet of the estimated number of molecules before and after PCR in the 5eU protocol; shared by Prashant over Slack on 2023-01-27
- `modules/`: core Python library for simulating splicing kinetics and fitting kinetic parameters. Scripts and notebooks import these modules by adding `modules/` to `sys.path`.
  - `simulate.py`: discrete-time stochastic simulation of transcription initiation, elongation, splicing, and decay for labeled transcripts of a single gene isoform
    - Parameters (`params`, in this order): `k_init` (transcripts / time step), `k_decay` (1 / time step), `k_splice` (1 / time step), `k_elong` (nucleotides / time step). All scripts treat 1 time step as 1 second.
    - Labeling stops (no further initiation) at time step `t_wash` (default 600, i.e., a 10 min pulse), followed by the chase.
    - Key assumptions: single transcription start and end site; non-bursty (Poisson) initiation; constant elongation rate along the gene; the same splicing rate for all introns; an intron can be spliced only after its 3' splice site has been transcribed; only fully elongated transcripts decay. Overlapping introns are handled as mutually exclusive splicing events.
    - `simulate_transcripts()` runs one simulation; `parallel_simulations()` runs `n` replicate simulations in a process pool and aggregates them (`mean`, `mean_nan`, `mean_and_var`).
  - `stats_transcripts.py`: summary statistics computed from simulated transcripts at each time point. These are compared against data during fitting. The abbreviations in parentheses are used in `scripts/` filenames.
    - `spliced_fraction()` (`sfs`): spliced / (spliced + unspliced) for each intron
    - `splice_site_counts()` (`bcs`, "bond counts"): for each intron, the number of transcripts containing its donor bond, its acceptor bond, or its spliced junction
    - `junction_counts()` (`jcs`): for each intron, counts of exon, intron, and spliced-junction coverage (assumes no alternative splicing)
    - `multi_stats()`: computes several statistics in one simulation pass
  - `fitting.py`: estimates `params` by minimizing the distance (default: Euclidean, via `scipy.spatial.distance.pdist`) between data and the mean of `n` simulations, searching over log10-transformed parameters
    - `main()` wraps three fitting methods: `gp` (Gaussian-process Bayesian optimization, `skopt.gp_minimize`), `scipy` (L-BFGS-B, `scipy.optimize.minimize`), and `igs` (custom `iterative_grid_search()`)
    - `main()` currently hard-codes the mouse *Actb* isoform ENSMUST00000100497.10, the experimental time points, and the parameter bounds
  - `utils_genomics.py`: gene-structure helpers. `parse_gtf_to_introns()` converts GTF exon annotations to intron coordinates, `pos_intron_to_pos_exon()` converts intron coordinates to exon coordinates, and `mutually_exclusive_splicing()` builds the matrix of introns whose splicing is mutually exclusive.
- `notebooks/archive/`: notebooks kept for the record; not maintained
  - `2022-11_fit_nanog_gp.ipynb` (formerly `notebooks/Untitled.ipynb`): the only attempt so far to fit the model to real data. It predates the `modules/` refactor and defines its own simulation code (`simulate_transcripts(alpha, beta, gamma, v, ...)`, `average_simulations()`), which `modules/simulate.py` and `modules/utils_genomics.py` have since replaced. It will not run against the current modules.
    - Data: intron and junction read counts for the 5 annotated *Nanog* introns, from Mitchell Guttman's `introns.counts` and `junctions.counts` (`data_aux/from_mitch/`), plus `normalized_counts_filtered.csv`
    - Fit: Gaussian-process minimization (`skopt.gp_minimize`, 200 calls) of the sum of squared errors against the mean of 25 simulations. Progress was checkpointed to `data_aux/current_res.pkl` (not tracked).
    - Result: the fit did not produce plausible rates. Decay and splicing rates converged to their lower bounds (1e-4 / s), and the elongation rate converged to about 1 nt/s (far below typical rates of about 25–75 nt/s). This motivated the simulation-based identifiability study (`notebooks/identifiability.ipynb`) before any further real-data fitting.
- `scripts/old/`: superseded or failed scripts from the February 2023 comparison of PCR/optical duplicate marking tools, run on the original aligned BAMs (`aligned.MAPQ255STAR.mouse.bam`). Kept for the record; none produced outputs that are used elsewhere.
  - Context: the current scripts `scripts/dedup_picard.sbatch` (Picard MarkDuplicates) and `scripts/dedup_samtools.sbatch` (samtools markdup), both with optical duplicate distance 2500, write `data/dedup/{timepoint}.MarkDuplicates.bam` and `data/dedup/{timepoint}.markdup.bam`. `scripts/dedup_diff_no_flags_or_tags.sbatch` confirmed that the two outputs are identical apart from flags and tags (all `.no_flags_or_tags.diff` files are empty). `scripts/dedup_diff_flags_and_tags.sbatch` diffs flags and tags read by read in both directions. `notebooks/process_data.ipynb` analyzes these outputs, and Picard MarkDuplicates was adopted in the Snakemake workflow (`filter_align_dedup` rule).
  - `dedup_diff.sh`: early draft of `scripts/dedup_diff_flags_and_tags.sbatch`. It diffs only reads that differ on the Picard side, writes `.sam` rather than `.tsv`, and its SLURM job and log names were mistakenly copied from the no-flags-or-tags script.
  - `dedup_picard_samtools.sbatch`: runs samtools markdup on Picard-deduplicated BAMs, to test whether a second tool finds duplicates the first missed. **Failed**: every job hit the 4 h time limit during `samtools sort`. Its output paths also lack a directory prefix, so outputs would have been written to the working directory.
  - `dedup_samtools_picard.sbatch`: the reverse (Picard MarkDuplicates on samtools markdup output). **Failed**: Picard rejected the samtools markdup BAMs with `SAM validation error: INVALID_FLAG_MATE_UNMAPPED ... Mate unmapped flag should not be set for unpaired read`. Rerunning with `--VALIDATION_STRINGENCY LENIENT` would likely avoid this.

# Other software to consider

[INSPEcT](https://bioconductor.org/packages/INSPEcT/): a Bioconductor R package ([de Pretis et al.](https://doi.org/10.1093/bioinformatics/btv288)) that estimates RNA synthesis, processing (splicing) and degradation rates from metabolic-labeling time courses, using total and nascent RNA-seq