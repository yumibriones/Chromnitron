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
        # merged_fragments_path = os.path.join(INPUTS_DIR, "merged_fragments.tsv.gz")
        # utils.merge_fragments_tsv(samples, DATA_DIR, merged_fragments_path)

    ### create_mudata: Create and save MuData objects per sample in sample sheet.
    ### If files exist, adds Souporcell donor metadata and CellRanger ATAC peak annotations.
    ### Computes QC metrics, and merges into one MuData object.
    if "create_mudata" in STEPS_TO_RUN:
        mudata_list = {}
        for sample in samples:
            sample_dir = os.path.join(DATA_DIR, sample)
            try:
                mudata_obj = utils.read_10x_multiome(sample_dir, sample)
                mudata_list[sample] = mudata_obj
                # compute qc metrics right after creation
                features_bed_path = os.path.join(RESOURCES_DIR, "DNA_sequence", "features.bed")
                gff_path = os.path.join(RESOURCES_DIR, "DNA_sequence", "gencode.v45.transcripts.annotation.gff3")
                features_bed = utils.get_features_bed(features_bed_path, gff_path)
                mudata_obj = utils.compute_qc_metrics(mudata_obj, features_bed)
            except Exception as e:
                logging.exception("Error creating sample %s: %s", sample, e)
        # merge mudata_list into one MuData object
        mudata_merged = utils.merge_mudata_list(mudata_list, "sample")
        mudata_filepath = os.path.join(OBJECTS_DIR, f"{PROJECT_PREFIX}.h5mu")
        utils.save_mudata(mudata_merged, mudata_filepath)

    ### add_donor_metadata: If donor metadata was not added in previous step, add it now
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

    ### call_peaks: Call ATAC peaks with MACS3 per sample
    ### TODO: Add grouping_var functionality, have to manipulate fragments files for this though
    ### TODO: Add peak annotations to MuData objects after peak calling
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
        
        logging.info("Summarizing and plotting QC metrics")

        # Global metrics summary
        global_filename = f"{PROJECT_PREFIX}_global_qc"
        global_metrics_summary = utils.summarize_qc_metrics(mudata_merged)
        with open(os.path.join(OUTPUTS_DIR, f"{global_filename}.json"), "w") as f:
            json.dump(global_metrics_summary, f, indent=4)

        # Grouped QC metrics summary and plots
        grouping_vars = config["qc_config"].get("grouping_vars", ["sample"])
        for grouping_var in grouping_vars:
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
        
        logging.info("Filtering cells by QC thresholds")
        rna_filter_config = config.get("filter_config", {}).get("rna", {})
        atac_filter_config = config.get("filter_config", {}).get("atac", {})
        filter_all_modalities = config["filter_config"].get("filter_all_modalities", True)

        # Global cells filtering
        mudata_merged = utils.filter_cells_by_qc(mudata_merged, "rna", filter_all_modalities, **rna_filter_config)
        mudata_merged = utils.filter_cells_by_qc(mudata_merged, "atac", filter_all_modalities, **atac_filter_config)
        global_filename = f"{PROJECT_PREFIX}_filtered"
        mudata_filepath = os.path.join(OBJECTS_DIR, f"{global_filename}.h5mu")
        utils.save_mudata(mudata_merged, mudata_filepath)

        # Global QC metrics summary after filtering
        global_metrics_summary = utils.summarize_qc_metrics(mudata_merged)
        with open(os.path.join(OUTPUTS_DIR, f"{global_filename}_qc.json"), "w") as f:
            json.dump(global_metrics_summary, f, indent=4)

        # Grouped QC metrics summary and plots after filtering
        grouping_vars = config["filter_config"].get("grouping_vars", ["sample"])
        for grouping_var in grouping_vars:
            filename = f"{PROJECT_PREFIX}_by_{grouping_var}_filtered_qc"
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

        logging.info("Normalizing and performing dimensionality reduction")

        # Global processing
        mudata_merged = utils.run_postqc_rna(mudata_merged)
        mudata_merged = utils.run_postqc_atac_lsi(mudata_merged)
        filename = f"{PROJECT_PREFIX}_postqc"
        mudata_filepath = os.path.join(OBJECTS_DIR, f"{filename}.h5mu")
        utils.save_mudata(mudata_merged, mudata_filepath)
        
        # Plot if enabled
        plot_postqc = config.get("postqc_config", {}).get("plotting", True)
        if plot_postqc:
            grouping_vars = config.get("postqc_config", {}).get("grouping_vars", ["sample"])
            rna_filepath = os.path.join(PLOTS_DIR, f"{filename}_rna.pdf")
            utils.plot_postqc(mudata_merged, rna_filepath, grouping_vars=grouping_vars, modality="rna")
            atac_filepath = os.path.join(PLOTS_DIR, f"{filename}_atac.pdf")
            utils.plot_postqc(mudata_merged, atac_filepath, grouping_vars=grouping_vars, modality="atac")

    ### if add_metadata: Add metadata from provided metadata file to MuData object
    if "add_metadata" in STEPS_TO_RUN:
        if mudata_merged is None:
            raise RuntimeError("MuData not loaded. Run create_mudata or provide --mudata_files.")
        
        logging.info("Adding metadata to merged MuData object")
        metadata_file = setup_config.get("metadata_file", None)
        metadata_key = setup_config.get("metadata_key", None)
        if metadata_file is not None and metadata_key is not None:
            metadata_df = pd.read_csv(metadata_file, index_col=0, dtype=str)
            mudata_merged = utils.add_metadata_to_mudata(mudata_merged, metadata_df, metadata_key)
            mudata_filepath = os.path.join(OBJECTS_DIR, f"{PROJECT_PREFIX}_with_metadata.h5mu")
            utils.save_mudata(mudata_merged, mudata_filepath)
        else:
            logging.info("No metadata_file or metadata_key provided; skipping adding metadata.")

    ### cluster: Clustering analysis
    if "cluster" in STEPS_TO_RUN:
        if mudata_merged is None:
            raise RuntimeError("MuData not loaded.")

        logging.info("Clustering")
        cluster_params = config.get("cluster_config", {}).get("params", {})
        mudata_merged = utils.run_cluster_rna(mudata_merged, **cluster_params)
        mudata_merged = utils.run_cluster_atac_lsi(mudata_merged, **cluster_params)
        filename = f"{PROJECT_PREFIX}_clustered"
        mudata_filepath = os.path.join(OBJECTS_DIR, f"{filename}.h5mu")
        utils.save_mudata(mudata_merged, mudata_filepath)

        # Plot if enabled
        plot_cluster = config.get("cluster_config", {}).get("plotting", True)
        if plot_cluster:
            grouping_vars = config.get("cluster_config", {}).get("grouping_vars", ["sample"])
            filepath = os.path.join(PLOTS_DIR, f"{filename}.pdf")
            utils.plot_clusters(mudata_merged, filepath, grouping_vars=grouping_vars)
    
    ### integrate_modalities: Integrate RNA and ATAC modalities
    if "integrate_modalities" in STEPS_TO_RUN:
        if mudata_merged is None:
            raise RuntimeError("MuData not loaded.")

        logging.info("Integrating RNA and ATAC modalities")
        # integration_params = config.get("integrate_modalities_config", {}).get("params", {})
        mofa_filepath = os.path.join(OBJECTS_DIR, f"{PROJECT_PREFIX}_mofa.hdf5")
        mudata_merged = utils.integrate_modalities(mudata_merged, mofa_filepath)
        filename = f"{PROJECT_PREFIX}_integrated"
        mudata_filepath = os.path.join(OBJECTS_DIR, f"{filename}.h5mu")
        utils.save_mudata(mudata_merged, mudata_filepath)

        # Plot if enabled
        plot_integration = config.get("integrate_modalities_config", {}).get("plotting", True)
        if plot_integration:
            grouping_vars = config.get("integrate_modalities_config", {}).get("grouping_vars", ["sample"])
            filepath = os.path.join(PLOTS_DIR, f"{filename}.pdf")
            utils.plot_integrated(mudata_merged, filepath, grouping_vars=grouping_vars)

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