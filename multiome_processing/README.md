## Setup

1. Set up `muon_env`.
```bash
conda create -n muon_env python=3.10 -y
conda activate muon_env
pip install -r requirements.txt
```

2. Set up MACS2/MACS3 environments (for peak calling).

```bash
conda create -n macs2 -c bioconda -c conda-forge macs2
conda activate macs2
```

```bash
conda create -n macs3 -c bioconda -c conda-forge macs3
conda activate macs3
```

## Running Pipeline
1. Configure the `config.yaml` file to specify input fastq file and output locations. 
2. To execute pipeline, run 
```
python main.py config.yaml
```
3. For inference with Chromnitron, softlink (`ln -s `) the normalized zarr file to ATAC-seq directory with a **specified condition name** (`ln -s <path-to-output>s8_normalized/genrich_normalized.zarr <<path-to-chromnitron_resource>/input_resources/ATAC-seq/<condition-name>.zarr`). 
4. During inference setup, add the condition name to `celltype.txt` to enable inference on custom ATAC-seq generated here.

## Notes: additional data downloads on the first run
The pipeline will automatically download a few files on the first run. In case some links are broken, here is a list of all files that you can manually check and download:
1. Hisat2 alignment index: `https://genome-idx.s3.amazonaws.com/hisat/${ASSEMBLY}_genome.tar.gz`
2. cPeaks (hg38): `https://cloud.tsinghua.edu.cn/f/6eb530748b324f53bc1f/?dl=1` # Very likely to change in the future
3. chr_sizes (hg38): `https://raw.githubusercontent.com/igvteam/igv/main/genomes/sizes/hg38.chrom.sizes`
