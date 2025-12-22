# Shoebill

Shoebill is an interpretable protein crystallization propensity predictor built on an XGBoost framework. It evaluates whether a protein—assuming successful expression and purification—is likely to form diffraction-quality crystals by leveraging a compact yet comprehensive feature set derived from both protein sequence and AlphaFold2 (AF2)-predicted structures.

---

## Installation

### Requirements

Shoebill is implemented in Python and requires the following core dependencies:

- python=3.12.3
- xgboost=3.0.2
- pandas=2.2.3
- numpy=2.2.5
- shap==0.46.0
- matplotlib==3.9.2
- joblib

### Environment setup (recommended)

We strongly recommend creating a dedicated conda environment using the supplied configuration file `environment.yml` to ensure full compatibility.

If if you didn't install anaconda before, please see Install Anaconda section below.

```bash
conda env create -f environment.yml
conda activate shoebill_env
```

Once activated, all Shoebill scripts can be executed directly within this environment.

### External dependencies (required binaries)

This step will install all the dependencies required for running Shoebill. After installation, please place the corresponding executable or binary files into the bin/ directory to ensure correct path resolution during runtime.

#### Required external tools

 - Anaconda (if you have already install, just move on)
   
   Download Anaconda (64 bit) installer python3.x for linux : https://www.anaconda.com/distribution/#download-section
   
 - EDTSurf

   Please download and install EDTSurf from https://aideepmed.com/EDTSurf/

   Then, place the `EDTSurf` into the `bin/` directory
 
 - DSSP

   Please download and install mkdssp from https://github.com/cmbi/dssp

   Then, place the `mkdssp` into the `bin/` directory
  
 - Zernike3d
   
   Please download and install zernike3d from https://github.com/jerhoud/zernike3

   Then, place the `MakeShape` and `Shpae2Zernike` into the `bin/` directory

 - KORP

   Please download and install korpe from https://chaconlab.org/modeling/korp/down-korp

   Then, place the `korpe` and `korp6Dv1.bin` into the `bin/` directory

##### Note

If the binaries are installed in a different location, the corresponding paths can be manually updated in `shoebill_predict.py`.

## Usage

The Shoebill workflow consists of three main steps:

(i) 3D structure prediction using AlphaFold2

(ii) Feature generation from AF2 outputs

(iii) Crystallization propensity prediction

## (i) 3D structure prediction using AlphaFold2

Shoebill requires AlphaFold2-predicted protein structures as input.

A convenient way to generate AF2 models is via ColabFold:
https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb

Please download and unzip the AF2 output files for later steps.

## (ii) Feature generation

Step 1. Prepare input files

 - Navigate to the `Feature_generation/` directory

 - Provide the target protein sequence in FASTA format (with or without tag are all ok!) with file name: `TE_Sequence.fasta`

 - Place the unzipped AlphaFold2 output into the `AF_Result/` directory

Step 2. Preprocess AF2 outputs

```bash
python AF_Preprocessing.py TE_Sequence.fasta AF_Result Processed_AF_Result
```

Step 3. Extract features

```bash
python TE_feature.py TE_Sequence.fasta Processed_AF_Result TE_feature.csv --bin ./bin
```

This step generates 830 structural and sequence-derived features.

Detailed feature definitions and computational procedures are provided in Supplementary File 2.

## (iii) Crystallization propensity prediction

(1) Activate environment
conda activate shoebill_env
(2) Predict
python shoebill_predict.py \
    --model shoebill_model \
    --input feature.csv \
    --output preds.csv \
    --threshold 0.420 \

The output `PRED.csv` will contain:
- `ProteinID` (if provided via `--id-col`)
- `pred_proba` (probability of "crystallizable")
- `pred_label` (0/1 using the chosen threshold; default = 0.420)
