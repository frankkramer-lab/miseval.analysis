# Towards a Guideline for Evaluation Metrics in Medical Image Segmentation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5877797.svg)](https://doi.org/10.5281/zenodo.5877797)

The work was performed using the framework:  
miseval: a metric library for Medical Image Segmentation EVALuation  
https://github.com/frankkramer-lab/miseval

In the last decade, research on artificial intelligence has seen rapid growth with deep learning models, especially in the field of medical image segmentation. Various studies demonstrated that these models have powerful prediction capabilities and achieved similar results as clinicians. However, recent years revealed that the evaluation in image segmentation studies lacks reliable model performance assessment and showed statistical bias by incorrect metric implementation or usage. Thus, this work provides an overview and interpretation guide on the following metrics for medical image segmentation evaluation in binary as well as multi-class problems: Dice similarity coefficient, Jaccard, Sensitivity, Specificity, Rand index, ROC curves, Cohen’s Kappa, and Hausdorff distance. As a summary, we propose a guideline for standardized medical image segmentation evaluation to improve evaluation quality, reproducibility, and comparability in the research field.

The models, predictions, and evaluation (scores, figures) are available under the following link: https://doi.org/10.5281/zenodo.5877797

## Reproducibility

**Requirements:**
- Ubuntu 18.04
- Python 3.8
- NVIDIA QUADRO RTX 6000 or a GPU with equivalent performance

**Step-by-Step workflow:**

Download the code repository via git clone to your disk. Afterwards, install all required dependencies.

```sh
git clone https://github.com/frankkramer-lab/miseval.analysis
cd miseval.analysis/

python -m pip install -r requirements.txt
```

Now, you can reproduce this analysis:

```sh
# Preparation
python scripts/prepare/prepare.braintumor.py
python scripts/prepare/prepare.covid.py
python scripts/prepare/prepare.histopathology.py

# Training
nohup sh -c "python scripts/miscnn/train.covid.py" &> log.covid.txt &
nohup sh -c "python scripts/miscnn/train.braintumor.py" &> log.braintumor.txt &
nohup sh -c "python scripts/miscnn/train.histopathology.py" &> log.histopathology.txt &

# Inference
nohup sh -c "python scripts/miscnn/predict.covid.py" &> log.predict.covid.txt &
nohup sh -c "python scripts/miscnn/predict.braintumor.py" &> log.predict.braintumor.txt &
nohup sh -c "python scripts/miscnn/predict.histopathology.py" &> log.predict.histopathology.txt &

# Evaluation
nohup sh -c "python scripts/evaluate/eval.covid.py" &> log.evaluate.covid.txt &
nohup sh -c "python scripts/evaluate/eval.braintumor.py" &> log.evaluate.braintumor.txt &
nohup sh -c "python scripts/evaluate/eval.histopathology.py" &> log.evaluate.histopathology.txt &
```

## Author

Dominik Müller  
Email: dominik.mueller@informatik.uni-augsburg.de  
IT-Infrastructure for Translational Medical Research  
University Augsburg  
Bavaria, Germany

## How to cite / More information

Dominik Müller, Iñaki Soto-Rey and Frank Kramer. (2022)   
Towards a Guideline for Evaluation Metrics in Medical Image Segmentation.  
arXiv e-print: Coming soon

```
Article{misevalAnalysis2022Mueller,
  title={Towards a Guideline for Evaluation Metrics in Medical Image Segmentation},
  author={Dominik Müller, Iñaki Soto-Rey and Frank Kramer},
  year={2022},
  eprint={Coming Soon},
  archivePrefix={arXiv},
  primaryClass={X}
}
```

Thank you for citing our work.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3.  
See the LICENSE.md file for license rights and limitations.
