import os
import shutil
import logging
import pandas as pd
import numpy as np

import muon as mu
from muon import atac as ac
import mudata as md
from mudata import MuData

import anndata as ad
import scanpy as sc
import scipy.sparse as sp

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
setup_logging()

md.set_options(pull_on_update=False)

def read_mudata_dict(mudata_files: list) -> dict:
    """
    Read multiple MuData files into a dictionary.

    Parameters
    - mudata_files: list of paths to MuData .h5mu files

    Returns
    - dict mapping sample names (derived from filenames) to MuData objects
    """
    mudata_dict = {}
    for path in mudata_files:
        sample_name = os.path.splitext(os.path.basename(path))[0]
        logging.info("Loading MuData for sample %s from %s", sample_name, path)
        mudata = mu.read_h5mu(path)
        mudata_dict[sample_name] = mudata
    return mudata_dict

def create_dirs_from_samplesheet(samplesheet: pd.DataFrame, raw_out_base: str = "data/raw") -> None:
    """
    Read a samplesheet CSV with columns `sampleName` and `path` and run create_data_dirs for each row.

    Example CSV rows:
      sampleName,path
      Sample1,/path/to/sample1
      Sample2,/path/to/sample2
    """
    
    required = {"sampleName", "path"}
    if not required.issubset(samplesheet.columns):
        raise ValueError(f"samplesheet must contain columns: {required}")

    for _, row in samplesheet.iterrows():
        sample = row["sampleName"]
        data_dir = row["path"]
        new_dir = os.path.join(raw_out_base, sample)
        create_data_dirs(data_dir=data_dir, new_dir=new_dir, sample=sample)

def create_data_dirs(data_dir: str, new_dir: str, sample: str) -> None:
    """
    Copy ATAC fragments and filtered_feature_bc_matrix contents into a new directory.

    Parameters
    - data_dir: path to original sample directory (contains atac_fragments.tsv.gz and filtered_feature_bc_matrix/)
    - new_dir: destination directory to create (e.g., data/raw/<sample>)
    - sample: sample name (used for logging)
    """
    os.makedirs(new_dir, exist_ok=True)
    logging.info("Processing sample %s: data.dir=%s -> new.dir=%s", sample, data_dir, new_dir)

    # Copy fragments file (rename to fragments.tsv.gz) and its index (.tbi)
    src_frag = os.path.join(data_dir, "atac_fragments.tsv.gz")
    dst_frag = os.path.join(new_dir, "fragments.tsv.gz")
    if os.path.exists(src_frag):
        shutil.copy2(src_frag, dst_frag)
        logging.info("Copied fragments: %s -> %s", src_frag, dst_frag)
    else:
        logging.warning("Fragments file not found: %s", src_frag)

    src_frag_idx = src_frag + ".tbi"
    dst_frag_idx = dst_frag + ".tbi"
    if os.path.exists(src_frag_idx):
        shutil.copy2(src_frag_idx, dst_frag_idx)
        logging.info("Copied fragments index: %s -> %s", src_frag_idx, dst_frag_idx)
    else:
        logging.warning("Fragments index not found: %s", src_frag_idx)

    # Candidate locations for 10x filtered matrix: "filtered_feature_bc_matrix" or "outs/filtered_feature_bc_matrix"
    candidate_paths = [
        os.path.join(data_dir, "filtered_feature_bc_matrix"),
        os.path.join(data_dir, "outs", "filtered_feature_bc_matrix"),
    ]

    copied_any = False
    for src_dir in candidate_paths:
        if os.path.isdir(src_dir):
            for fname in os.listdir(src_dir):
                src_file = os.path.join(src_dir, fname)
                dst_file = os.path.join(new_dir, fname)
                try:
                    shutil.copy2(src_file, dst_file)
                    logging.info("Copied: %s -> %s", src_file, dst_file)
                    copied_any = True
                except Exception as e:
                    logging.warning("Failed to copy %s: %s", src_file, e)
            break

    if not copied_any:
        logging.warning(
            "No filtered_feature_bc_matrix found under expected locations: %s. Searched candidates.",
            ", ".join(candidate_paths),
        )

def read_10x_multiome(sample_dir: str, sample_name: str) -> mu.MuData:
    """
    Read 10x multiome outputs (filtered_feature_bc_matrix.h5) and create a MuData object
    Expects cellranger-arc output layout with a single filtered_feature_bc_matrix.h5
    """
    logging.info("Loading sample %s from %s", sample_name, sample_dir)

    obj = mu.read_10x_mtx(sample_dir)
    if isinstance(obj, mu.MuData):
        mudata = obj
    else:
        mudata = MuData({
            "rna":  obj[:, obj.var["feature_types"] == "Gene Expression"],
            "atac": obj[:, obj.var["feature_types"] == "Peaks"],
        })

    # add fragments file
    fragments_path = os.path.join(sample_dir, "fragments.tsv.gz")
    if os.path.exists(fragments_path):
        logging.info("Linking fragments file for sample %s: %s", sample_name, fragments_path)
        ac.tl.locate_fragments(mudata.mod["atac"], fragments_path)
    else:
        logging.warning("Fragments file not found for sample %s: %s", sample_name, fragments_path)

    # mudata.obs_names = [f"{sample_name}_{bc}" for bc in mudata.obs_names]
    mudata.obs["sample"] = sample_name
    # Ensure layers are dicts to avoid write_h5mu error
    for mod in mudata.mod.values():
        if mod.layers is None:
            mod.layers = {}
    logging.info("Loaded MuData for sample %s: %d cells (RNA: %d genes, ATAC: %d peaks)",
                 sample_name, mudata.n_obs, mudata.mod["rna"].n_vars, mudata.mod["atac"].n_vars)
    return mudata

def merge_mudata_dict(mudata_dict: dict) -> MuData:
    mudata = md.concat(list(mudata_dict.values()), join="outer", label="sample", keys=list(mudata_dict.keys()))
    return mudata

def call_macs2_peaks(sample_dir: str, out_dir: str, macs2_path: str = "macs2"):
    """
    Call peaks with MACS2 on fragments file(s). This is a wrapper that calls MACS2 from the shell.
    This function is intentionally simple: expects fragments file path(s) be present.
    """
    fragments = os.path.join(sample_dir, "atac_fragments.tsv.gz")
    if not os.path.exists(fragments):
        logging.warning("Fragments not found: %s (skipping MACS2)", fragments)
        return None

    out_prefix = os.path.join(out_dir, "macs2_peaks")
    os.makedirs(out_dir, exist_ok=True)

    # Example MACS2 call for ATAC pseudo-bulk:
    cmd = f"{macs2_path} callpeak -t {fragments} -f BED -g hs -n {out_prefix} --nomodel --shift -100 --extsize 200 -q 0.01"
    logging.info("Would run MACS2 (not executed here):\n%s", cmd)
    # If you want to run uncomment:
    # os.system(cmd)
    return out_prefix + "_peaks.narrowPeak"

def get_features_bed(features_bed_path: str, gff_path: str) -> pd.DataFrame:
    # Load features.bed if exists
    if os.path.exists(features_bed_path):
        logging.info("Loaded features.bed from %s", features_bed_path)
        return pd.read_csv(
            features_bed_path,
            sep="\t",
            header=None,
            names=["Chromosome", "Start", "End", "Name", "Strand"]
        )
    # If not, create from GFF
    if not os.path.exists(gff_path):
        raise FileNotFoundError(f"GFF file not found at {gff_path}; cannot create features.bed")

    logging.info("features.bed not found; generating from GFF at %s", gff_path)
    features_bed = gff3_to_tss_features(gff_path)
    features_bed.to_csv(features_bed_path, sep="\t", header=False, index=False)
    logging.info("features.bed created at %s", features_bed_path)
    
    return features_bed

def gff3_to_tss_features(gff3_file: str) -> pd.DataFrame:
    logging.info("Converting GFF3 to TSS features from %s", gff3_file)
    # Load GFF3
    gff = pd.read_csv(gff3_file, sep="\t", comment="#", header=None,
        names=["chrom", "source", "feature", "start", "end", "score", "strand", "phase", "attributes"]
    )

    # Keep only transcripts
    transcripts = gff[gff["feature"] == "transcript"].copy()
    
    # Extract gene_name from attributes
    def extract_gene_name(attr):
        for item in attr.split(";"):
            if item.startswith("gene_name="):
                return item.split("=")[1]
        return None

    transcripts["gene_name"] = transcripts["attributes"].apply(extract_gene_name)
    
    # Compute TSS coordinates
    transcripts["tss_start"] = transcripts.apply(
        lambda row: row["start"] - 1 if row["strand"] == "+" else row["end"] - 1, axis=1
    )
    transcripts["tss_end"] = transcripts.apply(
        lambda row: row["start"] if row["strand"] == "+" else row["end"], axis=1
    )
    
    # Create BED-like DataFrame
    tss_features = transcripts[["chrom", "tss_start", "tss_end", "gene_name", "strand"]].copy()
    tss_features.columns = ["Chromosome", "Start", "End", "Name", "Strand"]
    
    return tss_features

def compute_qc_metrics(mudata: mu.MuData, features_bed: pd.DataFrame = None) -> mu.MuData:
    logging.info("Computing QC metrics for MuData")
    if "rna" in mudata.mod:
        rna = mudata.mod["rna"]
        if rna.X.shape[0] > 0 and rna.X.shape[1] > 0:
            rna.var['mt'] = rna.var_names.str.startswith('MT-')
            sc.pp.calculate_qc_metrics(rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)  # n_genes_by_counts, total_counts, pct_counts_mt
    if "atac" in mudata.mod:
        atac = mudata.mod["atac"]
        if atac.X.shape[0] > 0 and atac.X.shape[1] > 0:
            sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)  # n_genes_by_counts, total_counts
            ac.tl.nucleosome_signal(atac, n=1e6)  # adds atac.obs['nucleosome_signal']
            if features_bed is not None:
                logging.info("Computing TSS enrichment using provided features bed")
                tss = ac.tl.tss_enrichment(atac, features=features_bed, n_tss=1000)  # adds atac.obs['tss_enrichment']
            else:
                logging.info("No features bed provided; skipping TSS enrichment calculation")
    return mudata, tss

def plot_qc_metrics(mudata: mu.MuData, tss, output_dir: str, sample_name: str) -> None:
    logging.info("Plotting QC metrics for sample %s", sample_name)
    # Make multipage PDF with QC plots
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    pdf_path = os.path.join(output_dir, f"{sample_name}_qc_metrics.pdf")
    with PdfPages(pdf_path) as pdf:

        # RNA QC
        if "rna" in mudata.mod:
            rna = mudata.mod["rna"]
            if rna.X.shape[0] > 0 and rna.X.shape[1] > 0:
                g = sc.pl.violin(rna,
                                   ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                                   jitter=0.4,
                                   multi_panel=True,
                                   show=False)
                pdf.savefig(g.fig)
                plt.close(g.fig)

        # ATAC QC
        if "atac" in mudata.mod:
            atac = mudata.mod["atac"]
            if atac.X.shape[0] > 0 and atac.X.shape[1] > 0:
                # Violin
                g = sc.pl.violin(atac,
                                   ['n_genes_by_counts', 'total_counts', 'nucleosome_signal', 'tss_score'],
                                   jitter=0.4,
                                   multi_panel=True,
                                   show=False)
                pdf.savefig(g.fig)
                plt.close(g.fig)

                # # Nucleosome histogram
                # fig, ax = plt.subplots()
                # mu.pl.histogram(atac, "nucleosome_signal", kde=False, ax=ax)
                # pdf.savefig(fig)
                # plt.close(fig)

                # # TSS enrichment
                # if tss is not None:
                #     fig = ac.pl.tss_enrichment(tss)
                #     pdf.savefig(fig)
                #     plt.close(fig)

    logging.info("Saved QC metrics plots to %s", pdf_path)

def filter_cells_by_qc(mudata: mu.MuData,
                       min_n_genes_by_counts_rna: int = 500,
                       max_n_genes_by_counts_rna: int = 500,
                       min_total_counts_rna: int = 1000,
                       max_total_counts_rna: int = 30000,
                       max_percent_mt: float = 20.0,
                       min_n_genes_by_counts_atac: int = 500,
                       max_n_genes_by_counts_atac: int = 500,
                       min_total_counts_atac: int = 1000,
                       max_total_counts_atac: int = 30000,
                       max_nucleosome_signal_atac: float = 4.0,
                       min_tss_score_atac: float = 2.0):
    """
    Simple QC filters applied to RNA obs columns. percent.mt requires MT genes in var names.
    """
    logging.info("Filtering cells by RNA QC thresholds")
    rna = mudata.mod["rna"]
    # percent.mt: compute if MT present
    mt_genes = [g for g in rna.var_names if g.upper().startswith("MT-") or g.upper().startswith("MT")]
    if len(mt_genes) > 0:
        rna.obs["percent_mt"] = np.array(rna[:, mt_genes].X.sum(axis=1)).ravel() / (rna.obs["nCount_RNA"] + 1e-9) * 100
    else:
        rna.obs["percent_mt"] = 0.0

    keep_rna = (
        (rna.obs["nCount_RNA"] >= min_nCount_RNA) &
        (rna.obs["nCount_RNA"] <= max_nCount_RNA) &
        (rna.obs["nFeature_RNA"] >= min_nFeature_RNA) &
        (rna.obs["percent_mt"] <= max_percent_mt)
    )
    kept_cells = rna.obs_names[keep_rna.values]
    logging.info("Keeping %d / %d cells after RNA QC", kept_cells.size, rna.n_obs)

    # Subset muData to kept cells
    mudata = mudata[kept_cells.tolist(), :]
    mudata.obs_names_make_unique()
    return mudata

def preprocess_rna(mudata: mu.MuData, n_top_genes: int = 2000):
    """
    Normalize/scale/find HVG for RNA (scanpy-style).
    """
    logging.info("Preprocessing RNA: normalize, log1p, HVG, scale")
    rna = mudata.mod["rna"]
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, n_top_genes=n_top_genes, flavor="seurat_v3")
    rna.raw = rna.copy()
    sc.pp.scale(rna, zero_center=True)
    sc.tl.pca(rna, n_comps=50, svd_solver="arpack")
    mudata.mod["rna"] = rna
    return mudata

def atac_tfidf_lsi(atac_adata: ad.AnnData, n_components: int = 30):
    """
    Compute TF-IDF and LSI for ATAC peaks (peaks x cells).
    Expects atac_adata.X to be peaks x cells; scanpy AnnData is cells x features,
    so atac_adata.X shape = (cells, peaks)
    """
    logging.info("Running TF-IDF + LSI on ATAC (sparse-friendly)")
    if atac_adata.X.shape[1] == 0:
        logging.warning("ATAC assay empty; skipping TF-IDF/LSI")
        return atac_adata

    # TF-IDF using sklearn on cells x peaks matrix
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    X_tfidf = tfidf.fit_transform(atac_adata.X)  # cells x peaks (sparse)
    # Run TruncatedSVD (LSI)
    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=0)
    lsi = svd.fit_transform(X_tfidf)  # cells x n_components
    for i in range(min(n_components, lsi.shape[1])):
        atac_adata.obsm[f"X_lsi_{i+1}"] = lsi[:, i]
    atac_adata.obsm["X_lsi"] = lsi
    logging.info("LSI computed shape=%s", lsi.shape)
    return atac_adata


def run_harmony(adata: ad.AnnData, key: str = "batch", max_iter_harmony: int = 100):
    """
    Run Harmony on adata.obsm['X_pca'] and store corrected embedding in .obsm['X_pca_harmony'].
    Requires harmonypy.
    """
    if not HARMONY_AVAILABLE:
        logging.warning("harmonypy not available; skipping Harmony")
        return adata

    if "X_pca" not in adata.obsm:
        logging.warning("PCA not present; skipping Harmony")
        return adata

    logging.info("Running Harmony on variable %s", key)
    ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs, key, max_iter_harmony=max_iter_harmony)
    adata.obsm["X_pca_harmony"] = ho.Z_corr.T
    return adata


def construct_wnn(mudata: mu.MuData,
                  rna_key: str = "X_pca",
                  atac_key: str = "X_lsi",
                  n_neighbors: int = 30,
                  rna_weight: float = 0.5,
                  atac_weight: float = 0.5):
    """
    Approximate a WNN by computing modality-specific neighbor graphs and combining adjacency weights.
    Returns an adjacency matrix and stores neighbor indices in mudata.obs as metadata for plotting.
    """
    logging.info("Constructing approximate WNN graph (combining RNA and ATAC neighbors)")
    rna = mudata.mod["rna"]
    atac = mudata.mod["atac"]

    # Get embeddings
    if rna_key in rna.obsm:
        rna_emb = rna.obsm[rna_key]
    else:
        logging.info("RNA embedding %s missing, computing PCA embedding", rna_key)
        sc.tl.pca(rna, n_comps=30)
        rna_emb = rna.obsm["X_pca"]

    if atac_key in atac.obsm:
        atac_emb = atac.obsm["X_lsi"]
    else:
        logging.info("ATAC embedding missing; run TF-IDF/LSI first")
        atac_emb = None

    # Compute neighbor graphs (use sklearn NearestNeighbors)
    n = rna.n_obs
    if rna_emb is None:
        raise RuntimeError("RNA embedding required for WNN")

    nbrs_rna = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(rna_emb)
    dist_rna, idx_rna = nbrs_rna.kneighbors(rna_emb)

    if atac_emb is not None and atac_emb.shape[0] == n:
        nbrs_atac = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(atac_emb)
        dist_atac, idx_atac = nbrs_atac.kneighbors(atac_emb)
    else:
        idx_atac = None

    # Store neighbors in obs (as simple counts or indices)
    mudata.obs["rna_nbrs_mean_dist"] = dist_rna.mean(axis=1)
    if idx_atac is not None:
        mudata.obs["atac_nbrs_mean_dist"] = dist_atac.mean(axis=1)

    # For visualization and clustering, pick to use combined embedding as concatenation weighted
    if atac_emb is not None and atac_emb.shape[0] == n:
        combined = np.hstack([rna_emb * rna_weight, atac_emb * atac_weight])
    else:
        combined = rna_emb

    # Create a single AnnData for neighbors and UMAP
    combined_adata = ad.AnnData(X=combined, obs=mudata.obs.copy())
    sc.pp.neighbors(combined_adata, n_neighbors=n_neighbors, use_rep="X")
    sc.tl.umap(combined_adata)
    # attach wnn umap coordinates back to mudata
    mudata.obsm["wnn_umap"] = combined_adata.obsm["X_umap"]
    return mudata


def find_markers_rna(mudata: mu.MuData, groupby: str = "wsnn_res", resolution: float = 0.5, n_top: int = 5):
    """
    Cluster by RNA PCA and find cluster markers (scanpy rank_genes_groups)
    """
    rna = mudata.mod["rna"]
    # run neighbors/UMAP/clustering if needed
    if "X_pca" not in rna.obsm:
        sc.tl.pca(rna, n_comps=30)
    sc.pp.neighbors(rna, use_rep="X_pca")
    sc.tl.umap(rna)
    sc.tl.leiden(rna, resolution=resolution, key_added=f"{groupby}")
    sc.tl.rank_genes_groups(rna, groupby, method="wilcoxon")
    # collect top markers
    markers = {}
    for g in rna.obs[f"{groupby}"].cat.categories:
        df = sc.get.rank_genes_groups_df(rna, group=g)
        markers[g] = df.head(n_top)
    return markers


def pseudobulk_de_by_celltype(mudata: mu.MuData, celltype_key: str = "celltype", group_key: str = "disease"):
    """
    Create pseudobulk (sum counts per sample per celltype) for RNA and run DE with statsmodels or edgeR/DESeq2 externally.
    This function returns a dataframe scaffold; performing DE in R (DESeq2/edgeR) is recommended for robust results.
    """
    logging.info("Creating pseudobulk counts per sample x celltype (RNA)")
    rna = mudata.mod["rna"]
    if "sample" not in rna.obs.columns:
        raise RuntimeError("Sample column required in rna.obs for pseudobulk")
    # Example: group by sample + celltype
    groups = rna.obs[[celltype_key, "sample", group_key]]
    cells = rna.obs_names.to_series()
    expr = rna.X  # sparse matrix cells x genes
    expr_df = pd.DataFrame.sparse.from_spmatrix(expr, index=rna.obs_names, columns=rna.var_names)
    # Sum per (sample, celltype)
    expr_df = expr_df.join(groups)
    pb = expr_df.groupby(["sample", celltype_key]).sum()
    return pb


def link_peaks_to_genes(mudata: mu.MuData, distance: int = 250000):
    """
    Correlate peak accessibility (ATAC) with gene expression (RNA) across cells to propose links.
    This is a simple correlation-based approach and does not replace Signac LinkPeaks.
    """
    logging.info("Linking peaks to genes via correlation (simple)")
    # Ensure ACTIVITY assay exists or compute gene activity
    if "atac" not in mudata.mod or mudata.mod["atac"].X.shape[1] == 0:
        logging.warning("No ATAC counts available for linking")
        return None

    # Compute gene activity (very naive): sum of peaks overlapping gene body +/- promoter region
    # NOTE: real pipeline should use genomic ranges; here we just create a placeholder.
    logging.info("Placeholder: creating ACTIVITY assay from ATAC counts via summing all peaks (not genomic-aware)")
    atac = mudata.mod["atac"]
    # sum across peaks to approximate activity per cell for demo (not correct biologically)
    gene_activity = np.array(atac.X.sum(axis=1)).ravel()
    # add to RNA obs for correlation
    rna = mudata.mod["rna"]
    rna.obs["pseudo_gene_activity_total_peaks"] = gene_activity
    # compute correlation between pseudo_activity and expression of top marker genes
    top_genes = rna.var_names[:50]
    corr = {}
    activity_vec = gene_activity
    for g in top_genes:
        expr = rna[:, g].X.toarray().ravel() if sp.issparse(rna[:, g].X) else rna[:, g].X.ravel()
        corr[g] = np.corrcoef(activity_vec, expr)[0, 1]
    corr_df = pd.Series(corr).sort_values(ascending=False).to_frame("corr_with_activity")
    return corr_df

def save_mudata(mudata: mu.MuData, filename: str, output_dir: str):
    save_path = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    logging.info("Saving MuData to %s", save_path)
    mudata.write_h5mu(save_path)