# Status

As of 2026-11, written by Claude Code Opus 5.5; last active work April 2023

**Simulation and identifiability (complete)**
- Stochastic simulation and fitting framework in `modules/`.
- Simulated data for 81 parameter sets on mouse *Actb*, fit with 3 methods (grid search, L-BFGS-B, Gaussian process) for each of 3 statistics (spliced fraction, bond counts, junction counts); results in `data_aux/sim_*` (HPC only, not tracked).
- Finding (`notebooks/identifiability.ipynb`): spliced fraction is *not* identifiable (different `k_init`/`k_elong` combinations give near-identical trajectories); bond counts are. In spot checks, `k_splice` is recovered well while `k_init` and `k_elong` trade off.
- Possibly unanalyzed: junction-count fits and the bond-count L-BFGS-B rerun (`--n 100`) were run after `identifiability.ipynb` was last updated.

**Data processing (complete)**
- Snakemake workflow (`workflow/`) ran to completion on 2023-04-14 for all 12 time points: trimming, two-pass STAR alignment, unique-read filtering, Picard deduplication.
- Read retention (`notebooks/alignment.ipynb`): >99% survive trimming, ~77% map, ~87% of mapped reads are unique, and only ~30% of unique reads survive deduplication (high PCR duplication; cf. `data/moleculecount.xlsx`).

**Fitting real data (not started)**
- The only attempt, an early fit to *Nanog* before the `modules/` refactor (`notebooks/archive/2022-11_fit_nanog_gp.ipynb`), gave implausible rates.

**Next steps**
1. Analyze the junction-count fits; decide which statistic to fit (bond or junction counts).
2. Add a workflow step that computes per-intron junction/bond counts from the deduplicated BAMs (`workflow/workup/align/{timepoint}/aligned_unique_dedup.sorted.bam`).
3. Fit real *Actb* and *Nanog* data (`data/sample_*.annot`), then scale to more genes.
4. Address the "Open questions" below.

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
  - `sample_actb.annot`, `sample_nanog.annot`: example transcript annotations for `scripts/simulate_main.py` (`--file_annot`), in its whitespace-delimited format: `#` comment lines, then the gene length, then one `start end` line per intron, with 1-indexed inclusive coordinates relative to the TSS. *Actb*: ENSMUST00000100497.10 (GENCODE; 5 introns). *Nanog*: NM_001289828.1 (RefSeq, mm10; 3 introns; formerly `nanog.mm10.annot` in the project root).
  - `sample_actb.json`: example simulation parameters for `scripts/simulate_main.py`
  - Although `data/` is gitignored, the `sample_*` files above are tracked (added with `git add -f`).
- `modules/`: core Python library for simulating splicing kinetics and fitting kinetic parameters. Scripts and notebooks import these modules by adding `modules/` to `sys.path`.
  - `simulate.py`: discrete-time stochastic simulation of transcription initiation, elongation, splicing, and decay for labeled transcripts of a single gene isoform
    - Parameters (`params`, in this order): `k_init` (transcripts / time step), `k_decay` (1 / time step), `k_splice` (1 / time step), `k_elong` (nucleotides / time step). All scripts treat 1 time step as 1 second.
    - Labeling stops (no further initiation) at time step `t_wash` (default 600, i.e., a 10 min pulse), followed by the chase.
    - Key assumptions: single transcription start and end site; non-bursty (Poisson) initiation; constant elongation rate along the gene; the same splicing rate for all introns; an intron can be spliced only after its 3' splice site has been transcribed; only fully elongated transcripts decay. Overlapping introns are handled as mutually exclusive splicing events.
    - `simulate_transcripts()` runs one simulation; `parallel_simulations()` runs `n` replicate simulations in a process pool and aggregates them (`mean`, `mean_nan`, `mean_and_var`).
  - `stats_transcripts.py`: summary statistics computed from simulated transcripts at each time point. These are compared against data during fitting. The abbreviations in parentheses are used in `scripts/` filenames.
    - `spliced_fraction()` (`sfs`): spliced / (spliced + unspliced) for each intron
      - "Spliced" counts transcripts in which the intron has been spliced out (state `-1`); "unspliced" counts transcripts in which the intron is fully transcribed and elongation has passed its 3' splice site (state `> 1`). Transcripts still elongating through the intron, and introns excluded by a mutually exclusive splicing event (`NaN`), are counted in neither group.
      - History: originally named `su_ratio()`, which computed spliced / unspliced. Commit `55093ac` (2023-01-03) renamed it and changed the denominator to spliced + unspliced, so the statistic is bounded in [0, 1], is defined when there are no unspliced transcripts, and has the same form as the spliced fraction in `junctionFraction.matrix` (though that file's spliced and unspliced values are read-based, not transcript-based). Code written before then (e.g., `tests/test_simulate.py` and early notebooks) may still refer to `su_ratio`.
    - `splice_site_counts()` (`bcs`, "bond counts"): for each intron, the number of transcripts containing its donor bond, its acceptor bond, or its spliced junction
    - `junction_counts()` (`jcs`): for each intron, counts of exon, intron, and spliced-junction coverage (assumes no alternative splicing)
    - `multi_stats()`: computes several statistics in one simulation pass
  - `fitting.py`: estimates `params` by minimizing the distance (default: Euclidean, via `scipy.spatial.distance.pdist`) between data and the mean of `n` simulations, searching over log10-transformed parameters
    - `main()` wraps three fitting methods: `gp` (Gaussian-process Bayesian optimization, `skopt.gp_minimize`), `scipy` (L-BFGS-B, `scipy.optimize.minimize`), and `igs` (custom `iterative_grid_search()`)
    - `main()` currently hard-codes the mouse *Actb* isoform ENSMUST00000100497.10, the experimental time points, and the parameter bounds
  - `utils_genomics.py`: gene-structure helpers. `parse_gtf_to_introns()` converts GTF exon annotations to intron coordinates, `pos_intron_to_pos_exon()` converts intron coordinates to exon coordinates, and `mutually_exclusive_splicing()` builds the matrix of introns whose splicing is mutually exclusive.
  - `simulate_reads_wip.py` (formerly `modules/todo.py`; December 2022): **work in progress, not functional**, and not imported anywhere. A sketch of simulating sequencing reads from simulated transcripts, so that simulations could be compared to real data at the level of reads rather than transcripts. The current statistics (`stats_transcripts.py`) are computed directly on transcripts, ignoring fragmentation, read length, and sampling depth.
    - `read_to_junction_counts()`: classifies reads as exon, intron, or spliced-junction reads for each intron
    - `count_weighted_reads()`: intended to compute expected read counts per feature by averaging over all possible reads, given fragment length, read length, and sequencing depth; incomplete (contains TODOs)
    - `sample_reads()` and `transcripts_to_bam()`: empty stubs (sample reads instead of averaging; write simulated reads to a BAM file)
    - `main()`: example usage, but does not run (`stats_kwargs` is commented out)
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

# Open questions for exploratory data analysis

Originally outlined in `notebooks/reads_eda.ipynb` (December 2022). Several can now be answered from the Snakemake workflow outputs (`workflow/workup/align/{timepoint}/`).

- Reads
  - What percent of reads map uniquely to the reference genome? *Partially answered in `notebooks/alignment.ipynb`: ~87% of mapped reads (~67% of raw reads) map uniquely (MAPQ 255).*
- Splice sites
  - How many putative introns (≥10 reads spanning an exon-exon boundary) are there, and how many are annotated in GENCODE?
    - Of those not annotated: how many have an annotated donor but unannotated acceptor, an unannotated donor but annotated acceptor, or neither?
    - Of those with neither site annotated: how many are intronic (within an annotated GENCODE gene) vs. intergenic?
    - For comparison, TT-seq in K562 (hg38, GENCODE v24): "About one half of the putative introns (177,322) mapped to splice sites that had been annotated in the database of transcribed regions GENCODE ..., whereas the other half did not (164,533)." ([Wachutka et al. 2019](https://doi.org/10.7554/eLife.45056))
- Isoforms
  - What proportion of introns, and what proportion of reads, correspond to each gene's major isoform? Plot per gene.
- Splicing over time
  - What is the ratio of non-split to split reads over time, per gene? (cf. [Wachutka et al. 2019](https://doi.org/10.7554/eLife.45056), Figure 1—figure supplement 1)

# Notes

Formerly `notes.md` (December 2022).

- GENCODE annotations
  - Each exon entry is associated with a particular transcript. If the same exon is shared by multiple transcripts, the GTF/GFF3 annotation file has a separate exon entry (row) for each transcript. Example: exon chr17:35820000-35820120 (GRCm39) in the *Pou5f1* gene (ENSMUSG00000024406.17) is shared by transcripts ENSMUST00000025271.17, ENSMUST00000173934.8, ENSMUST00000172651.8, ENSMUST00000173805.8, and ENSMUST00000173256.8.
  - The "Transcript sequences" FASTA files include only exons, i.e., only sequences of fully spliced mRNAs.
- Potential modeling considerations
  - "splicing decisions [may be] connected to differential rates of Pol II elongation through splice junctions" ([PRO-seq paper](https://doi.org/10.1126/science.1229386)). The current model assumes a constant elongation rate along the gene (see `modules/simulate.py`).

# Other software to consider

[INSPEcT](https://bioconductor.org/packages/INSPEcT/): a Bioconductor R package ([de Pretis et al.](https://doi.org/10.1093/bioinformatics/btv288)) that estimates RNA synthesis, processing (splicing) and degradation rates from metabolic-labeling time courses, using total and nascent RNA-seq

[rCube](https://github.com/gagneurlab/rCube): an R package from the Gagneur lab ([Wachutka et al. 2019](https://doi.org/10.7554/eLife.45056)) that estimates synthesis, splicing, and degradation rates from TT-seq and RNA-seq. `notebooks/archive/2022-12_rCube_vignette.Rmd` (December 2022) runs the package's T-cell activation vignette against a local fork of rCube that fixes an incompatibility with SummarizedExperiment ≥ 1.18 (passing `withDimnames=FALSE` to `assay()`/`assays()`). It was not applied to this project's data.