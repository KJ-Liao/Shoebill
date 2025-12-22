# Shoebill

Shoebill is an interpretable protein crystallization propensity predictor built on an XGBoost framework. It evaluates whether a protein—assuming successful expression and purification—is likely to form diffraction-quality crystals by leveraging a compact yet comprehensive feature set derived from both protein sequence and AlphaFold2 (AF2)-predicted structures.

# Requirements
  - python=3.12.3
  - xgboost=3.0.2
  - pandas=2.2.3
  - numpy=2.2.5
  - shap==0.46.0
  - matplotlib==3.9.2
  - joblib

Please create the working environment with conda using following command

```
conda env create -f environment.yml
conda activate shoebill_env
```

# How to use Shoebill
## 1) 3D structure prediction using AlphaFold2

ColabFold: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb

## 2) Feature generation
(1) Download and navigate to the Feature_generation folder

(2) Navigate to the bin folder,
    download and install the following six required execute/bin file to the bin folder
 
 - Molecular surface generation:   EDTSurf (https://aideepmed.com/EDTSurf/)
 - Secondary structure annotation: mkdssp  (https://github.com/cmbi/dssp)
 - 3DZD computation: 1. MakeShape          (https://github.com/jerhoud/zernike3d)
                     2. Shpae2Zernike      (https://github.com/jerhoud/zernike3d)
 - Statistical potential calculation:  1. korpe          (https://chaconlab.org/modeling/korp/down-korp)
                                       2. korp6Dv1.bin   (https://chaconlab.org/modeling/korp/down-korp)


(3) Paste the name and corresponding sequence (with or without tag are all ok!) in fasta format to TE_Sequence.fasta

(4) Put your unzipped AlphaFold2 output in the AF_Result folder

(5) Execute AF_Preprocessing.py to preprocess the AlphaFold2 output

python AF_Preprocessing.py TE_Sequence.fasta AF_Result Processed_AF_Result

(6) Execute TE_feature.py to extract the feature and output as a TE_feature.csv file

python TE_feature.py TE_Sequence.fasta Processed_AF_Result TE_feature.csv --bin ./bin

## 3) Predict based on the given feature csv file
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
