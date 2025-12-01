import os
import sys
import argparse
import logging
from typing import List, Optional
import yaml
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

import anndata as ad
import scanpy as sc
import muon as mu

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

try:
    import harmonypy as hm
    HARMONY_AVAILABLE = True
except Exception:
    HARMONY_AVAILABLE = False

import utils

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def main(config_path: str = "config.yaml"):    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
 
    setup_config = config["setup"]

    # Set up directories
    RESOURCES_DIR = setup_config.get("resources_dir", ".")
    INPUTS_DIR = setup_config.get("inputs_dir", ".")
    BASE_PROJECT_DIR = setup_config.get("base_project_dir", "output")

    OUTPUTS_DIR = os.path.join(BASE_PROJECT_DIR, "outputs")
    DATA_DIR = os.path.join(BASE_PROJECT_DIR, "data/raw")

    for d in [BASE_PROJECT_DIR, OUTPUTS_DIR, DATA_DIR]:
        os.makedirs(d, exist_ok=True)

    # Load inputs
    samplesheet_path = os.path.join(INPUTS_DIR, setup_config.get("sample_sheet", "samplesheet.csv"))
    samplesheet = pd.read_csv(samplesheet_path, dtype=str)
    samples = samplesheet["sampleName"].tolist()
    mudata_files_list = setup_config.get("mudata_files") or []
    mudata_list = utils.read_mudata_list([os.path.join(OUTPUTS_DIR, f) for f in mudata_files_list]) if mudata_files_list else None
    # souporcelldfs = 
    
    # Steps to run
    STEPS_TO_RUN = config.get("steps_to_run", [])
    
    ### create_data_dirs: Create 10X Multiome data directories from sample sheet
    if "create_data_dirs" in STEPS_TO_RUN:
        print("Initializing data directories based on samplesheet")
        utils.create_dirs_from_samplesheet(samplesheet, DATA_DIR)

    ### create_mudata: Create and save MuData objects per sample in sample sheet
    if "create_mudata" in STEPS_TO_RUN:
        mudata_list = {}
        for sample in samples:
            sample_dir = os.path.join(DATA_DIR, sample)
            try:
                mudata_obj = utils.read_10x_multiome(sample_dir, sample)
                mudata_list[sample] = mudata_obj
                utils.save_mudata(mudata_obj, sample, OUTPUTS_DIR)
            except Exception as e:
                logging.exception("Error creating sample %s: %s", sample, e)

    ### call_peaks: Call MACS2 peaks per sample
    if "call_peaks" in STEPS_TO_RUN:
        macs2_path = None # TODO: Fix
        for sample in samples:
            sample_dir = samplesheet.loc[samplesheet["sampleName"] == sample, "path"].iloc[0]
            out_dir = os.path.join(OUTPUTS_DIR, "macs2")
            utils.call_macs2_peaks(sample_dir=sample_dir, out_dir=out_dir, macs2_path=macs2_path)

    ### qc_check: Quality control check
    if "qc_check" in STEPS_TO_RUN:
        features_bed_path = os.path.join(RESOURCES_DIR, "features.bed")
        gff_path = os.path.join(RESOURCES_DIR, "DNA_sequence", "gencode.v45.transcripts.annotation.gff3")
        features_bed = utils.get_features_bed(features_bed_path, gff_path)
        if mudata_list is None:
            raise RuntimeError("MuData not loaded. Run create_mudata or provide --mudata_files.")
        for sample, mudata_obj in mudata_list.items():
            logging.info("QC stage for sample: %s", sample)
            mudata_obj = utils.compute_qc_metrics(mudata_obj, features_bed=features_bed)
            filename = f"{sample}_qc"
            utils.save_mudata(mudata_obj, filename, OUTPUTS_DIR)
            metrics_summary = utils.summarize_qc_metrics(mudata_obj)
            with open(os.path.join(OUTPUTS_DIR, f"{filename}.json"), "w") as f:
                json.dump(metrics_summary, f, indent=4)
            # Plot if enabled
            plot_qc = config["qc_config"].get("plotting", True)
            if plot_qc:
                utils.plot_qc_metrics(mudata_obj, OUTPUTS_DIR, sample)

    ### filter: Filter cells by QC thresholds
    if "filter" in STEPS_TO_RUN:
        rna_filter_config = config.get("filter_config", {}).get("rna", {})
        atac_filter_config = config.get("filter_config", {}).get("atac", {})
        if mudata_list is None:
            raise RuntimeError("MuData not loaded. Run create_mudata or provide --mudata_files.")
        for sample, mudata_obj in mudata_list.items():
            logging.info("Filtering stage for sample: %s", sample)
            mudata_obj = utils.filter_cells_by_qc(mudata_obj, "rna", **rna_filter_config)
            mudata_obj = utils.filter_cells_by_qc(mudata_obj, "atac", **atac_filter_config)
            filename = f"{sample}_filtered"
            utils.save_mudata(mudata_obj, filename, OUTPUTS_DIR)
            metrics_summary = utils.summarize_qc_metrics(mudata_obj)
            with open(os.path.join(OUTPUTS_DIR, f"{filename}.json"), "w") as f:
                json.dump(metrics_summary, f, indent=4)
            # Plot if enabled
            plot_filter = config["filter_config"].get("plotting", True)
            if plot_filter:
                utils.plot_qc_metrics(mudata_obj, OUTPUTS_DIR, filename)

    if "postqc" in STEPS_TO_RUN:
        if mudata is None:
            raise RuntimeError("MuData not loaded.")
        # quick overview plots and statistics could be added here (skipping plotting code)
        logging.info("PostQC stage: calculating normalized RNA, HVG, ATAC LSI")
        mudata = preprocess_rna(mudata)
        mudata.mod["atac"] = atac_tfidf_lsi(mudata.mod["atac"], n_components=30)
        utils.save_mudata(mudata, os.path.join(OUTPUTS_DIR, f"{project_name}_postqc.h5mu"))

    ### cluster
    if "clustering" in STEPS_TO_RUN:
        if mudata is None:
            raise RuntimeError("MuData not loaded.")
        # run harmony optional
        rna = mudata.mod["rna"]
        if args.RunHarmony and HARMONY_AVAILABLE:
            run_harmony(rna, key="sample")
            # use harmony embedding for neighbors if present
            if "X_pca_harmony" in rna.obsm:
                rna.obsm["X_pca"] = rna.obsm["X_pca_harmony"]
        # construct WNN and cluster
        mudata = construct_wnn(mudata, rna_key="X_pca", atac_key="X_lsi", n_neighbors=30)
        # simple Leiden clustering on the combined embedding saved in mudata.obsm["wnn_umap"]
        combined_adata = ad.AnnData(X=mudata.obsm["wnn_umap"], obs=mudata.obs.copy())
        sc.pp.neighbors(combined_adata)
        sc.tl.leiden(combined_adata, key_added="wsnn_res")
        # transfer cluster ids back to mudata.obs
        mudata.obs["wsnn_res"] = combined_adata.obs["wsnn_res"].astype(str).values
        utils.save_mudata(mudata, os.path.join(OUTPUTS_DIR, f"{project_name}_clustered.h5mu"))
    
    ### merge_mudata: Merge all samples into single MuData object
    if "merge_mudata" in STEPS_TO_RUN:
        # merge behavior already handled in create; here we just ensure final object saved
        if mudata is None:
            raise RuntimeError("MuData not loaded.")
        utils.save_mudata(mudata, os.path.join(OUTPUTS_DIR, f"{project_name}_merged.h5mu"))

    ### link_peaks_to_genes: Link ATAC peaks to genes via correlation
    if "link_peaks_to_genes" in STEPS_TO_RUN:
        if mudata is None:
            raise RuntimeError("MuData not loaded.")
        corr_df = utils.link_peaks_to_genes(mudata)
        if corr_df is not None:
            out_csv = os.path.join(OUTPUTS_DIR, f"{project_name}_peak-gene-correlation.csv")
            corr_df.to_csv(out_csv)
            logging.info("Wrote peak-gene correlation scaffold to %s", out_csv)
    
    ### footprint: Perform TF footprinting analysis
    if "footprint" in STEPS_TO_RUN:
        if mudata is None:
            raise RuntimeError("MuData not loaded.")
        tf_motifs_path = os.path.join(RESOURCES_DIR, "tf_motifs.pwm")
        mudata = utils.perform_footprinting(mudata, tf_motifs_path)
        utils.save_mudata(mudata, os.path.join(OUTPUTS_DIR, f"{project_name}_footprinted.h5mu"))

    logging.info("Pipeline steps completed: %s", STEPS_TO_RUN)


if __name__ == "__main__":
    main()