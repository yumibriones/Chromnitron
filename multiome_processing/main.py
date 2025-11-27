import os
import sys
import argparse
import logging
from typing import List, Optional
import yaml

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

def main():    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
 
    # Setup
    setup_config = config["setup"]
    RESOURCES_DIR = setup_config.get("resources_dir", ".")
    INPUTS_DIR = setup_config.get("inputs_dir", ".")
    samplesheet_path = os.path.join(INPUTS_DIR, setup_config.get("sample_sheet", "samplesheet.csv"))
    mudata_files_list = setup_config.get("mudata_files") or []
    BASE_PROJECT_DIR = setup_config.get("base_project_dir", "output")
    
    project_prefix = setup_config.get("project_prefix", "muon_multiome")

    OUTPUTS_DIR = os.path.join(BASE_PROJECT_DIR, "outputs")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    DATA_DIR = os.path.join(BASE_PROJECT_DIR, "data/raw")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load inputs
    samplesheet = pd.read_csv(samplesheet_path, dtype=str)
    samples = samplesheet["sampleName"].tolist()
    mudata_dict = utils.read_mudata_dict([os.path.join(OUTPUTS_DIR, f) for f in mudata_files_list]) if mudata_files_list else None
    
    # Steps to run
    STEPS_TO_RUN = config.get("steps_to_run", [])
    
    ### Create 10X Multiome data directories from sample sheet
    if "create_data_dirs" in STEPS_TO_RUN:
        print("Initializing data directories based on samplesheet")
        utils.create_dirs_from_samplesheet(samplesheet, DATA_DIR)

    ### Create MuData objects
    if "create_mudata" in STEPS_TO_RUN:
        mudata_dict = {}
        # Create per-sample MuData objects and save intermediate files
        for sample in samples:
            sample_dir = os.path.join(DATA_DIR, sample)
            try:
                mudata_obj = utils.read_10x_multiome(sample_dir, sample)
                mudata_dict[sample] = mudata_obj
                save_path = os.path.join(OUTPUTS_DIR, f"{project_prefix}-{sample}.h5mu")
                utils.save_mudata(mudata_obj, save_path)
            except Exception as e:
                logging.exception("Error creating sample %s: %s", sample, e)

    # Call MACS2 peaks per sample
    if "peak_calling" in STEPS_TO_RUN:
        macs2_path = None # TODO: Fix
        for sample in samples:
            sample_dir = samplesheet.loc[samplesheet["sampleName"] == sample, "path"].iloc[0]
            out_dir = os.path.join(OUTPUTS_DIR, "macs2")
            utils.call_macs2_peaks(sample_dir=sample_dir, out_dir=out_dir, macs2_path=macs2_path)

    # Quality control
    if "qc" in STEPS_TO_RUN:
        features_bed_path = os.path.join(RESOURCES_DIR, "features.bed")
        gff_path = os.path.join(RESOURCES_DIR, "DNA_sequence", "gencode.v45.transcripts.annotation.gff3")
        features_bed = utils.get_features_bed(features_bed_path, gff_path)
        if mudata_dict is None:
            raise RuntimeError("MuData not loaded. Run create_mudata or provide --mudata_files.")
        for sample, mudata_obj in mudata_dict.items():
            logging.info("QC stage for sample: %s", sample)
            mudata_obj = utils.compute_qc_metrics(mudata_obj, features_bed=features_bed)
            save_path = os.path.join(OUTPUTS_DIR, f"{project_prefix}-{sample}_qc.h5mu")
            utils.save_mudata(mudata_obj, save_path)
        plot_qc = config["qc_config"].get("plotting", True)
        if plot_qc:
            utils.plot_qc_metrics(mudata_dict, OUTPUTS_DIR, project_prefix)

    # Filter cells by QC thresholds
    if "filter" in STEPS_TO_RUN:
        if mudata_dict is None:
            raise RuntimeError("MuData not loaded. Run create_mudata or provide --mudata_files.")
        for sample, mudata_obj in mudata_dict.items():
            logging.info("Filtering stage for sample: %s", sample)
            mudata_obj = utils.filter_cells_by_qc(mudata_obj)
            save_path = os.path.join(OUTPUTS_DIR, f"{project_prefix}-{sample}_filtered.h5mu")
            utils.save_mudata(mudata_obj, save_path)

    if "postqc" in STEPS_TO_RUN:
        if mudata is None:
            raise RuntimeError("MuData not loaded.")
        # quick overview plots and statistics could be added here (skipping plotting code)
        logging.info("PostQC stage: calculating normalized RNA, HVG, ATAC LSI")
        mudata = preprocess_rna(mudata)
        mudata.mod["atac"] = atac_tfidf_lsi(mudata.mod["atac"], n_components=30)
        utils.save_mudata(mudata, os.path.join(OUTPUTS_DIR, f"{project_prefix}_postqc.h5mu"))

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
        utils.save_mudata(mudata, os.path.join(OUTPUTS_DIR, f"{project_prefix}_clustered.h5mu"))
    
    if "merge" in STEPS_TO_RUN:
        # merge behavior already handled in create; here we just ensure final object saved
        if mudata is None:
            raise RuntimeError("MuData not loaded.")
        utils.save_mudata(mudata, os.path.join(OUTPUTS_DIR, f"{project_prefix}_merged.h5mu"))

    if "link_peaks_to_genes" in STEPS_TO_RUN:
        if mudata is None:
            raise RuntimeError("MuData not loaded.")
        corr_df = utils.link_peaks_to_genes(mudata)
        if corr_df is not None:
            out_csv = os.path.join(OUTPUTS_DIR, f"{project_prefix}_peak-gene-correlation.csv")
            corr_df.to_csv(out_csv)
            logging.info("Wrote peak-gene correlation scaffold to %s", out_csv)

    logging.info("Pipeline steps completed: %s", STEPS_TO_RUN)


if __name__ == "__main__":
    main()