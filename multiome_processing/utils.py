import os
import shutil
import logging
import pandas as pd
import numpy as np
import gzip
import subprocess

import muon as mu
from muon import atac as ac
import mudata as md
from mudata import MuData

import anndata as ad
import scanpy as sc
import scipy.sparse as sp

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
setup_logging()

md.set_options(pull_on_update=False)

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

def merge_sourporcell_dfs(samplesheet: pd.DataFrame, output_dir: str = "inputs") -> None:
    """
    Merge SoupOrCellDF files specified in samplesheet into a single CSV in output_dir.
    """
    souporcell_dfs = []
    for _, row in samplesheet.iterrows():
        sample = row["sampleName"]
        if "SoupOrCellDF" in samplesheet.columns and pd.notna(row["SoupOrCellDF"]):
            df = pd.read_csv(row["SoupOrCellDF"], dtype=str)
            df["sample"] = sample
            souporcell_dfs.append(df)
    if souporcell_dfs:
        merged_df = pd.concat(souporcell_dfs, ignore_index=True)
        merged_df_path = os.path.join(output_dir, "merged_souporcell_df.csv")
        merged_df.to_csv(merged_df_path, index=False)
        logging.info("Merged SoupOrCellDF saved to %s", merged_df_path)

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

    mudata.obs["sample"] = sample_name
    # Ensure layers are dicts to avoid write_h5mu error
    for mod in mudata.mod.values():
        if mod.layers is None:
            mod.layers = {}
    logging.info("Loaded MuData for sample %s: %d cells (RNA: %d genes, ATAC: %d peaks)",
                 sample_name, mudata.n_obs, mudata.mod["rna"].n_vars, mudata.mod["atac"].n_vars)
    return mudata

def merge_mudata_list(mudata_list: dict, label: str) -> MuData:
    # add prefix for unique obs names
    # save current obs names in 'barcode' column
    for m in mudata_list.values():
        m.obs['barcode'] = m.obs_names.tolist()
    updated = {sample: add_sample_prefix(m, sample) for sample, m in mudata_list.items()}
    merged = md.concat(updated.values(), join="outer", label=label, keys=list(updated.keys()))
    return merged

def add_sample_prefix(mdata: mu.MuData, sample: str):
    import re
    # rename global obs_names
    new_names = [re.sub(r"-1$", f"-{sample}", bc) for bc in mdata.obs_names]
    mdata.obs_names = new_names

    # propagate to all modalities
    for mod in mdata.mod.values():
        mod.obs_names = new_names

    return mdata

def add_donor_metadata(mudata: mu.MuData, souporcell_df: pd.DataFrame) -> dict:
    """
    Add donor assignments to mudata.obs from souporcell_df.
    souporcell_df must contain "barcode" and "assignment" columns.
    """
    import re
    logging.info("Splitting MuData by donor using souporcell assignments")

    # add sample prefix to barcodes in souporcell_df to match mudata.obs_names
    obs_names = [re.sub(r"-1$", f"-{row['sample']}", row["barcode"]) for _, row in souporcell_df.iterrows()]
    souporcell_df.index = obs_names
    souporcell_df["assignment"] = souporcell_df["assignment"].fillna("unknown")

    # Merge souporcell assignments into mudata.obs
    mudata.obs = mudata.obs.join(
        souporcell_df[["assignment"]], how="left"
    )
    mudata.obs = mudata.obs.rename(columns={"assignment": "donor"})
    
    # Propagate to all modalities
    for mod in mudata.mod.values():
        mod.obs["donor"] = mudata.obs["donor"].values

    return mudata

def merge_fragments_tsv(samples: list, data_dir: str, merged_path: str) -> None:
    """
    Merge fragments.tsv.gz from multiple samples into a single merged_fragments.tsv.gz.
    Keeps all columns, skips comment lines starting with '#' and empty lines.
    """
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    logging.info("Merging %d fragment files into %s", len(samples), merged_path)

    with gzip.open(merged_path, 'wt') as fout:
        for sample in samples:
            frag_src = os.path.join(data_dir, sample, "fragments.tsv.gz")
            if not os.path.exists(frag_src):
                logging.warning("Fragments file not found for sample %s: %s", sample, frag_src)
                continue
            logging.info("Processing sample %s", sample)

            with gzip.open(frag_src, 'rt') as fin:
                for line in fin:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    fout.write(line + "\n")

    logging.info("Merged fragments.tsv.gz created: %s", merged_path)

def subset_fragments_tsv(fragments_path: str, barcodes: set, output_path: str) -> None:
    """
    Subset a fragments.tsv.gz file to only include fragments with barcodes in the provided set.
    Writes to output_path as fragments_subset.tsv.gz.
    """
    logging.info("Subsetting fragments from %s to %s", fragments_path, output_path)
    with gzip.open(fragments_path, 'rt') as fin, gzip.open(output_path, 'wt') as fout:
        for line in fin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 4:
                continue
            barcode = fields[3]
            if barcode in barcodes:
                fout.write(line + "\n")
    logging.info("Subsetted fragments written to %s", output_path)

def call_peaks(fragments_path: str, output_dir: str, filename: str) -> None:
    """
    Call peaks on a merged fragments.tsv.gz file using MACS3 (paired-end / BEDPE).
    Peaks are saved in objects_dir with project_prefix.
    """
    logging.info("Calling peaks for %s using MACS3", filename)
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "macs3", "callpeak",
        "-t", fragments_path,
        "-f", "BEDPE",  # for paired end
        "-n", filename,
        "--outdir", output_dir,
        "-g", "hs",
        "--nomodel",
        "--shift", "-100",
        "--extsize", "200",
        "-q", "0.01"
    ]
    
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def get_features_bed(features_bed_path: str, gff_path: str) -> pd.DataFrame:
    """
    Load features.bed if exists; otherwise create from GFF.
    """
    if os.path.exists(features_bed_path):
        logging.info("Loaded features.bed from %s", features_bed_path)
        return pd.read_csv(
            features_bed_path,
            sep="\t",
            header=None,
            names=["Chromosome", "Start", "End", "Name", "Strand"]
        )
    if not os.path.exists(gff_path):
        raise FileNotFoundError(f"GFF file not found at {gff_path}; cannot create features.bed")

    logging.info("features.bed not found; generating from GFF at %s", gff_path)
    features_bed = gff3_to_tss_features(gff_path)
    features_bed.to_csv(features_bed_path, sep="\t", header=False, index=False)
    logging.info("features.bed created at %s", features_bed_path)
    
    return features_bed

def gff3_to_tss_features(gff3_file: str) -> pd.DataFrame:
    """
    Convert GFF3 file to TSS features DataFrame.
    """
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
    """
    Compute QC metrics for RNA and ATAC modalities in MuData.
    """
    logging.info("Computing QC metrics for MuData")
    if "rna" in mudata.mod:
        rna = mudata.mod["rna"]
        if rna.X.shape[0] > 0 and rna.X.shape[1] > 0:
            # Check if metrics already exist
            if not all(metric in rna.obs.columns for metric in ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']):
                rna.var['mt'] = rna.var_names.str.startswith('MT-')
                sc.pp.calculate_qc_metrics(rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)  # n_genes_by_counts, total_counts, pct_counts_mt
    if "atac" in mudata.mod:
        atac = mudata.mod["atac"]
        if atac.X.shape[0] > 0 and atac.X.shape[1] > 0:
            # Check if metrics already exist
            if not all(metric in atac.obs.columns for metric in ['n_genes_by_counts', 'total_counts']):
                sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)  # n_genes_by_counts, total_counts
                ac.tl.nucleosome_signal(atac, n=1e6)  # adds atac.obs['nucleosome_signal']
                if features_bed is not None:
                    logging.info("Computing TSS enrichment using provided features bed")
                    tss = ac.tl.tss_enrichment(atac, features=features_bed, n_tss=1000)  # adds atac.obs['tss_enrichment']
                else:
                    logging.info("No features bed provided; skipping TSS enrichment calculation")

    return mudata

def summarize_qc_metrics_grouped(mudata: mu.MuData, grouping_var: str) -> dict:
    """
    Summarize QC metrics for each modality in MuData, grouped by a metadata variable.
    """
    from pandas.api.types import is_numeric_dtype
    logging.info("Summarizing QC metrics for MuData grouped by %s", grouping_var)

    if grouping_var not in mudata.obs.columns:
        raise ValueError(f"grouping_var '{grouping_var}' not found in mudata.obs")

    grouped_summary = {}
    for group, group_obs in mudata.obs.groupby(grouping_var):
        metrics_summary = {
            "num_cells": len(group_obs),
            "modalities": {}
        }

        for mod_name, mod in mudata.mod.items():
            # subset modality to cells in this group
            cell_mask = mod.obs_names.isin(group_obs.index)
            sub_mod = mod[cell_mask, :]

            mod_metrics = {
                "num_features": sub_mod.n_vars,
                "metrics": {}
            }
            for col in sub_mod.obs.columns:
                if is_numeric_dtype(sub_mod.obs[col]):
                    series = sub_mod.obs[col]
                    mod_metrics["metrics"][col] = {
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "mean": float(series.mean()),
                        "median": float(series.median()),
                        "std": float(series.std())
                    }

            metrics_summary["modalities"][mod_name] = mod_metrics

        grouped_summary[str(group)] = metrics_summary

    return grouped_summary

def plot_qc_metrics(mudata: mu.MuData, filepath: str, grouping_var: str = "sample") -> None:
    """
    Plot QC metrics for each modality in MuData and save to a multipage PDF.
    """
    logging.info("Plotting QC metrics")
    # Make multipage PDF with QC plots

    n_groups = len(mudata.obs[grouping_var].unique())
    fig_width = max(8, n_groups * 0.6)  # scale with number of groups
    fig_height = 10                     # constant height
    
    with PdfPages(filepath) as pdf:

        # RNA QC
        if "rna" in mudata.mod:
            rna = mudata.mod["rna"]
            metrics_to_plot = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
            log_plot = [True, True, False]
            if rna.X.shape[0] > 0 and rna.X.shape[1] > 0:
                for metric in metrics_to_plot:
                    if metric in rna.obs.columns:
                        plt.figure(figsize=(fig_width, fig_height))
                        sc.pl.violin(rna,
                                     metric,
                                     groupby=grouping_var,
                                     log=log_plot[metrics_to_plot.index(metric)],
                                     stripplot=False,
                                     show=False)
                        plt.title(f"RNA QC: {metric} by {grouping_var}", fontsize=14)                
                        plt.tight_layout()
                        pdf.savefig(plt.gcf())
                        plt.close(plt.gcf())

        # ATAC QC
        if "atac" in mudata.mod:
            atac = mudata.mod["atac"]
            metrics_to_plot = ['n_genes_by_counts', 'total_counts', 'nucleosome_signal', 'tss_score']
            log_plot = [True, True, False, False]
            if atac.X.shape[0] > 0 and atac.X.shape[1] > 0:
                for metric in metrics_to_plot:
                    if metric in atac.obs.columns:
                        plt.figure(figsize=(fig_width, fig_height))
                        sc.pl.violin(atac,
                                     metric,
                                     groupby=grouping_var,
                                     log=log_plot[metrics_to_plot.index(metric)],
                                     stripplot=False,
                                     show=False)
                        plt.title(f"ATAC QC: {metric} by {grouping_var}", fontsize=14)
                        plt.tight_layout()
                        pdf.savefig(plt.gcf())
                        plt.close(plt.gcf())

def filter_cells_by_qc(
    mudata: mu.MuData,
    modality: str = "rna",
    filter_all_modalities: bool = True,
    min_n_genes_by_counts: int = 500,
    min_total_counts: int = 1000,
    max_total_counts: int = 30000,
    max_percent_mt: float = 20.0,
    max_nucleosome_signal: float = 2.0,
    min_tss_score: float = 1.0) -> mu.MuData:

    """
    Filter cells in MuData based on QC metrics for a given modality.
    """
    logging.info("Filtering MuData cells using QC metrics from modality: %s", modality)

    if modality not in mudata.mod:
        logging.warning("Modality %s not found; skipping filtering", modality)
        return mudata

    mod = mudata.mod[modality]
    initial_n_cells = mudata.n_obs

    mask = np.ones(mod.n_obs, dtype=bool)

    if "n_genes_by_counts" in mod.obs:
        mask &= (mod.obs["n_genes_by_counts"] >= min_n_genes_by_counts)

    if "total_counts" in mod.obs:
        mask &= (mod.obs["total_counts"] >= min_total_counts)
        mask &= (mod.obs["total_counts"] <= max_total_counts)

    if modality == "rna" and "pct_counts_mt" in mod.obs:
        mask &= (mod.obs["pct_counts_mt"] <= max_percent_mt)

    if modality == "atac":
        if "nucleosome_signal" in mod.obs:
            mask &= (mod.obs["nucleosome_signal"] <= max_nucleosome_signal)
        if "tss_enrichment" in mod.obs:
            mask &= (mod.obs["tss_enrichment"] >= min_tss_score)

    if filter_all_modalities: # Filter both ATAC and RNA modalities
        mudata = mudata[mask, :]
    else: # Filter only the specified modality
        mudata.mod[modality] = mod[mask, :]

    logging.info(
        "Filtered cells by QC (%d -> %d cells)", initial_n_cells, mudata.n_obs
    )

    return mudata

def run_postqc_rna(mudata: mu.MuData, n_top_genes: int = 2000):
    """
    Normalize/scale/find HVG for RNA (scanpy-style).
    """
    logging.info("Preprocessing RNA: normalize, log1p, HVG, scale")
    rna = mudata.mod["rna"]
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)
    logging.info("Identified %d highly variable genes in RNA", rna.var['highly_variable'].sum())
    # sc.pp.highly_variable_genes(rna, n_top_genes=n_top_genes, flavor='seurat_v3')
    rna.raw = rna.copy()
    sc.pp.scale(rna, zero_center=True)
    sc.tl.pca(rna, n_comps=50, svd_solver="arpack")
    mudata.mod["rna"] = rna
    return mudata

def run_postqc_atac_lsi(mudata: mu.MuData):
    """
    TF-IDF/LSI for ATAC (scanpy-style).
    """
    logging.info("Preprocessing ATAC: TF-IDF, LSI")
    atac = mudata.mod["atac"]
    atac.layers["counts"] = atac.X.copy()
    ac.pp.tfidf(atac, scale_factor=1e4)
    atac.raw = atac.copy()
    ac.tl.lsi(atac)
    atac.obsm['X_lsi'] = atac.obsm['X_lsi'][:,1:]
    atac.varm["LSI"] = atac.varm["LSI"][:,1:]
    atac.uns["lsi"]["stdev"] = atac.uns["lsi"]["stdev"][1:]
    mudata.mod["atac"] = atac
    return mudata

def run_postqc_atac_pca(mudata: mu.MuData):
    """
    Normalize/scale/find HVG for ATAC (scanpy-style).
    """
    logging.info("Preprocessing ATAC: normalize, log1p, HVG, scale")
    atac = mudata.mod["atac"]
    atac.layers["counts"] = atac.X.copy()
    sc.pp.normalize_per_cell(atac, counts_per_cell_after=1e4)
    sc.pp.log1p(atac)
    sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=1.5, min_disp=0.5)  
    logging.info("Identified %d highly variable peaks in ATAC", atac.var['highly_variable'].sum())
    # sc.pp.highly_variable_genes(atac, n_top_genes=n_top_genes, flavor='seurat_v3')
    atac.raw = atac.copy()
    sc.pp.scale(atac, zero_center=True)
    sc.tl.pca(atac, n_comps=50, svd_solver="arpack")
    mudata.mod["atac"] = atac
    return mudata

def plot_postqc(mudata: mu.MuData, filepath: str, color_by: str = None, modality: str = "rna"):
    if modality not in mudata.mod:
        logging.warning("Modality %s not found; skipping", modality)
        return
    
    mod = mudata.mod[modality]
    with PdfPages(filepath) as pdf:
        # plot highly variable genes for rna
        if modality == "rna":
            plt.figure(figsize=(8, 6))
            sc.pl.highly_variable_genes(mod, show=False)
            plt.title("RNA Highly Variable Genes")
            plt.tight_layout()
            pdf.savefig(plt.gcf())
            plt.close(plt.gcf())

        # plot pca
        plt.figure(figsize=(8, 6))
        sc.pl.pca(mod, color=color_by, show=False)
        plt.title(f"{modality.upper()} PCA")
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        plt.close(plt.gcf())

        # plot variance ratio
        plt.figure(figsize=(8, 6))
        sc.pl.pca_variance_ratio(mod, log=True, show=False)
        plt.title(f"{modality.upper()} PCA Variance Ratio")
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        plt.close(plt.gcf())

def run_cluster_rna(mudata: mu.MuData, n_neighbors: int = 10, min_dist: float = 0.5, resolution: float = 0.5, random_state=9) -> mu.MuData:
    """
    Run UMAP on RNA modality in MuData.
    """
    # check if previously run
    if "rna_leiden" in mudata.mod["rna"].obs and "X_umap" in mudata.mod["rna"].obsm:
        logging.info("RNA UMAP and clustering already present; skipping")
        return mudata
        
    logging.info("Running UMAP on RNA modality")
    rna = mudata.mod["rna"]
    sc.pp.neighbors(rna, n_neighbors=n_neighbors, use_rep="X_pca")
    sc.tl.leiden(rna, resolution=resolution, key_added="rna_leiden")
    sc.tl.umap(rna, min_dist=min_dist, random_state=random_state)
    mudata.mod["rna"] = rna
    return mudata

def plot_rna_umap(mudata: mu.MuData, filepath: str, color_by: str = "sample"):
    """
    Plot RNA UMAP and save to PDF.
    """
    logging.info("Plotting RNA UMAP")
    rna = mudata.mod["rna"]
    with PdfPages(filepath) as pdf:
        plt.figure(figsize=(8, 6))
        sc.pl.umap(rna, color=color_by, show=False)
        plt.title("RNA UMAP")
        plt.tight_layout()
        pdf.savefig(plt.gcf())
        plt.close(plt.gcf())

### NOTE: Unreviewed functions

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

def save_mudata(mudata: mu.MuData, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    logging.info("Saving MuData to %s", filepath)
    mudata.write_h5mu(filepath)

### NOTE: Unused functions

# def read_mudata_list(mudata_files: list) -> dict:
#     """
#     Read multiple MuData files into a dictionary.

#     Parameters
#     - mudata_files: list of paths to MuData .h5mu files

#     Returns
#     - dict mapping sample names (derived from filenames) to MuData objects
#     """
#     mudata_list = {}
#     for path in mudata_files:
#         logging.info("Loading MuData from %s", path)
#         mudata = mu.read_h5mu(path)
#         sample_name = mudata.obs["sample"].iloc[0]
#         mudata_list[sample_name] = mudata
#     return mudata_list