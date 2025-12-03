## Setup

1. Set up `muon_env`.

```bash
conda create -n muon_env python=3.10 -y
conda activate muon_env
pip install -r requirements.txt  # install all other requirements
conda install -c bioconda -c conda-forge macs3  # install macs3
```

2. Prepare `samplesheet.csv` and fill in the following columns:

    * `sampleName`: Name of single 10X Multiome run
    * `path`: path to CellRanger output. Must contain the following files:
        * `filtered_feature_bc_matrix/`
            * `barcodes.tsv.gz`
            * `features.tsv.gz`
            * `matrix.mtx.gz`
        * `atac_fragments.tsv.gz`
    * `SoupOrCellDF`: path to Souporcell CSV containing donor assignments.


3. Prepare `inputs_dir` and place `samplesheet.csv` inside. 

    * The pipeline will generate other files that will be saved to `inputs_dir`.

    * While not required, it is recommended to create a `base_project_dir` in advance and place `inputs_dir` inside. This way, all project files are in one master folder. 

    * If you follow the above suggestion, your directory structure will look like this:

        ```text
        base_project_dir
        ├── data
        │   └── raw
        ├── inputs_dir
        │   └── samplesheet.csv
        └── outputs
            ├── objects
            └── plots
        ```

## Running Pipeline
1. Configure the `config.yaml` file (instructions are commented on each line).

2. To execute pipeline, run:

    ```
    python main.py config.yaml
    ```

## Notes
* 