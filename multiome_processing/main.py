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
import mudata as md

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

    ### Set up directories
    RESOURCES_DIR = setup_config.get("resources_dir", ".")
    INPUTS_DIR = setup_config.get("inputs_dir", ".")
    BASE_PROJECT_DIR = setup_config.get("base_project_dir", "output")
    PROJECT_PREFIX = setup_config.get("project_prefix", "merged")

    DATA_DIR = os.path.join(BASE_PROJECT_DIR, "data/raw")
    OUTPUTS_DIR = os.path.join(BASE_PROJECT_DIR, "outputs")
    OBJECTS_DIR = os.path.join(OUTPUTS_DIR, "objects")
    PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")

    for d in [BASE_PROJECT_DIR, OUTPUTS_DIR, DATA_DIR, OBJECTS_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)

    ### Load inputs
    samplesheet_path = os.path.join(INPUTS_DIR, setup_config.get("sample_sheet", "samplesheet.csv"))
    samplesheet = pd.read_csv(samplesheet_path, dtype=str)
    samples = samplesheet["sampleName"].tolist()
    mudata_file_path = setup_config.get("mudata_file", None)
    mudata_merged = md.read_h5mu(mudata_file_path) if mudata_file_path else None
    
    ### Steps to run
    STEPS_TO_RUN = config.get("steps_to_run", [])
    
    ### initialize: Create 10X Multiome data dirs and prepare inputs for donor demultiplexing/peak calling
    if "initialize" in STEPS_TO_RUN:
        print("Initializing data dirs and inputs from samplesheet")
        utils.create_dirs_from_samplesheet(samplesheet, DATA_DIR)
        utils.merge_sourporcell_dfs(samplesheet, INPUTS_DIR)
        merged_fragments_path = os.path.join(INPUTS_DIR, "merged_fragments.tsv.gz")
        # utils.merge_fragments_tsv(samples, DATA_DIR, merged_fragments_path)

    ### create_mudata: Create and save MuData objects per sample in sample sheet
    if "create_mudata" in STEPS_TO_RUN:
        mudata_list = {}
        for sample in samples:
            sample_dir = os.path.join(DATA_DIR, sample)
            try:
                mudata_obj = utils.read_10x_multiome(sample_dir, sample)
                mudata_list[sample] = mudata_obj
                # compute qc metrics right after creation
                features_bed_path = os.path.join(RESOURCES_DIR, "features.bed")
                gff_path = os.path.join(RESOURCES_DIR, "DNA_sequence", "gencode.v45.transcripts.annotation.gff3")
                features_bed = utils.get_features_bed(features_bed_path, gff_path)
                mudata_obj = utils.compute_qc_metrics(mudata_obj, features_bed)
            except Exception as e:
                logging.exception("Error creating sample %s: %s", sample, e)
        # merge mudata_list into one MuData object
        mudata_merged = utils.merge_mudata_list(mudata_list, "sample")
        mudata_filepath = os.path.join(OBJECTS_DIR, f"{PROJECT_PREFIX}.h5mu")
        utils.save_mudata(mudata_merged, mudata_filepath)

    ### add_donor_metadata: Add donor metadata to merged MuData object
    if "add_donor_metadata" in STEPS_TO_RUN:
        if mudata_merged is None:
            raise RuntimeError("MuData not loaded. Run create_mudata or provide --mudata_files.")
        
        logging.info("Adding donor metadata to merged MuData object")
        souporcell_df_path = os.path.join(INPUTS_DIR, "merged_souporcell_df.csv")
        if os.path.exists(souporcell_df_path):
            souporcell_df = pd.read_csv(souporcell_df_path, dtype=str)
            mudata_merged = utils.add_donor_metadata(mudata_merged, souporcell_df)
            mudata_filepath = os.path.join(OBJECTS_DIR, f"{PROJECT_PREFIX}_with_donors.h5mu")
            utils.save_mudata(mudata_merged, mudata_filepath)
        else:
            logging.info("No SoupOrCellDF found for merged data; skipping split.")

    ### call_peaks: Call ATAC peaks per sample and add to merged MuData object
    if "call_peaks" in STEPS_TO_RUN:
        # if mudata_merged is None:
        #     raise RuntimeError("MuData not loaded. Run create_mudata or provide --mudata_files.")
        
        logging.info("Calling ATAC peaks per sample")
        grouping_var = config["call_peaks_config"].get("grouping_var", "sample")

        for sample in samples:
            logging.info("Calling peaks for sample: %s", sample)
            sample_fragments_path = os.path.join(DATA_DIR, sample, "fragments.tsv.gz")
            utils.call_peaks(sample_fragments_path, output_dir=OBJECTS_DIR, filename=sample)
    
    ### qc: Quality control metrics summary and plots
    if "qc" in STEPS_TO_RUN:
        if mudata_merged is None:
            raise RuntimeError("MuData not loaded. Run create_mudata or provide --mudata_files.")
        
        logging.info("QC stage for merged MuData object")
        grouping_var = config["qc_config"].get("grouping_var", "sample")
        filename = f"{PROJECT_PREFIX}_by_{grouping_var}_qc"
        
        metrics_summary = utils.summarize_qc_metrics_grouped(mudata_merged, grouping_var)
        with open(os.path.join(OUTPUTS_DIR, f"{filename}.json"), "w") as f:
            json.dump(metrics_summary, f, indent=4)
        
        # Plot if enabled
        plot_qc = config["qc_config"].get("plotting", True)
        if plot_qc:
            filepath = os.path.join(PLOTS_DIR, f"{filename}.pdf")
            utils.plot_qc_metrics(mudata_merged, filepath, grouping_var)

    ### filter: Filter cells by QC thresholds
    if "filter" in STEPS_TO_RUN:
        if mudata_merged is None:
            raise RuntimeError("MuData not loaded. Run create_mudata or provide --mudata_files.")
        
        logging.info("Filtering stage for merged MuData object")
        rna_filter_config = config.get("filter_config", {}).get("rna", {})
        atac_filter_config = config.get("filter_config", {}).get("atac", {})
        filename = f"{PROJECT_PREFIX}_by_{grouping_var}_filtered"
        
        filter_all_modalities = config["filter_config"].get("filter_all_modalities", True)
        grouping_var = config["filter_config"].get("grouping_var", "sample")

        mudata_merged = utils.filter_cells_by_qc(mudata_merged, "rna", filter_all_modalities, **rna_filter_config)
        mudata_merged = utils.filter_cells_by_qc(mudata_merged, "atac", filter_all_modalities, **atac_filter_config)
        mudata_filepath = os.path.join(OBJECTS_DIR, f"{filename}.h5mu")
        utils.save_mudata(mudata_merged, mudata_filepath)
        metrics_summary = utils.summarize_qc_metrics_grouped(mudata_merged, grouping_var)

        with open(os.path.join(OUTPUTS_DIR, f"{filename}.json"), "w") as f:
            json.dump(metrics_summary, f, indent=4)

        # Plot if enabled
        plot_filter = config["filter_config"].get("plotting", True)
        if plot_filter:
            filepath = os.path.join(PLOTS_DIR, f"{filename}.pdf")
            utils.plot_qc_metrics(mudata_merged, filepath, grouping_var)

    ### postqc: Post-QC processing (normalization, HVG, LSI)
    if "postqc" in STEPS_TO_RUN:
        if mudata_merged is None:
            raise RuntimeError("MuData not loaded.")

        logging.info("PostQC stage: calculating normalized RNA, HVG, ATAC LSI")
        grouping_var = config.get("postqc_config", {}).get("grouping_var", "sample")
        filename = f"{PROJECT_PREFIX}_by_{grouping_var}_postqc"

        mudata_merged = utils.run_postqc_rna(mudata_merged)
        # mudata_merged = utils.run_postqc_atac_lsi(mudata_merged)
        mudata_filepath = os.path.join(OBJECTS_DIR, f"{filename}.h5mu")
        utils.save_mudata(mudata_merged, mudata_filepath)
        
        # Plot if enabled
        plot_postqc = config.get("postqc_config", {}).get("plotting", True)
        if plot_postqc:
            filepath = os.path.join(PLOTS_DIR, f"{filename}.pdf")
            utils.plot_postqc(mudata_merged, filepath, color_by=grouping_var)

    ### cluster: Clustering analysis
    if "cluster" in STEPS_TO_RUN:
        if mudata_merged is None:
            raise RuntimeError("MuData not loaded.")
        logging.info("Clustering stage for merged MuData object")
        cluster_params = config.get("cluster_config", {}).get("params", {})
        grouping_var = config.get("cluster_config", {}).get("grouping_var", "sample")
        filename = f"{PROJECT_PREFIX}_by_{grouping_var}_clustered"

        mudata_merged = utils.run_cluster_rna(mudata_merged, **cluster_params)
        mudata_filepath = os.path.join(OBJECTS_DIR, f"{filename}.h5mu")
        utils.save_mudata(mudata_merged, mudata_filepath)

        # Plot if enabled
        plot_cluster = config.get("cluster_config", {}).get("plotting", True)
        if plot_cluster:
            filepath = os.path.join(PLOTS_DIR, f"{filename}.pdf")
            utils.plot_rna_umap(mudata_merged, filepath, color_by=grouping_var)

    # ### link_peaks_to_genes: Link ATAC peaks to genes via correlation
    # if "link_peaks_to_genes" in STEPS_TO_RUN:
    #     if mudata is None:
    #         raise RuntimeError("MuData not loaded.")
    #     corr_df = utils.link_peaks_to_genes(mudata)
    #     if corr_df is not None:
    #         out_csv = os.path.join(OUTPUTS_DIR, f"{project_name}_peak-gene-correlation.csv")
    #         corr_df.to_csv(out_csv)
    #         logging.info("Wrote peak-gene correlation scaffold to %s", out_csv)
    
    # ### footprint: Perform TF footprinting analysis
    # if "footprint" in STEPS_TO_RUN:
    #     if mudata is None:
    #         raise RuntimeError("MuData not loaded.")
    #     tf_motifs_path = os.path.join(RESOURCES_DIR, "tf_motifs.pwm")
    #     mudata = utils.perform_footprinting(mudata, tf_motifs_path)
    #     utils.save_mudata(mudata, os.path.join(OUTPUTS_DIR, f"{project_name}_footprinted.h5mu"))

    # logging.info("Pipeline steps completed: %s", STEPS_TO_RUN)


if __name__ == "__main__":
    main()