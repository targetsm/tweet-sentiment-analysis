# Model
DistilBERT model - 89.42% accuracy

# Reproducibility
In order to reproduce the result and generate `submission.csv`, execute the following commands:

Running on a local machine with gpu support and corresponding libraries installed:
* `python3 dBERTforSeqClass.py`

To run it on Euler/Leonhard first set up the environment:
* `module load gcc/8.2.0 python_gpu/3.8.`
* `module load eth_proxy`

To run it with GPU support on Euler/Leonhard run:
* `bsub -n 4 -W 24:00 -R "rusage[mem=10000, ngpus_excl_p=1]"  python3 dBERT_forSeqClass.py`

The submission file can be found under out/submission_dBERTforSeqClass.csv .
