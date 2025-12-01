## Setup

1. Set up `muon_env`.

```bash
conda create -n muon_env python=3.10 -y
conda activate muon_env
pip install -r requirements.txt
```

2. Set up MACS2/MACS3 environments (for peak calling).

```bash
#MACS2
conda create -n macs2 -c bioconda -c conda-forge bioconda::macs2
conda activate macs2
```

```bash
#MACS3
conda create -n macs3 -c bioconda -c conda-forge bioconda::macs3
conda activate macs3
```

## Running Pipeline
1. Configure the `config.yaml` file (instructions are commented on each line).

2. To execute pipeline, run:

    ```
    python main.py config.yaml
    ```

## Notes
* 