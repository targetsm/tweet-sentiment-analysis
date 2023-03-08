# Model
ByT5 model - 87% accuracy

# Files
The already trained model is stored at `./models/t5_base_afdaa4da` (NB! model takes up 1GB)

The main code that was used during training is stored at `ByT5_model.py`
We omit describing the whole training process as running the model on the whole dataset was challenging and included manual changes during training due to the load on memory.

Kaggle submission file is stored at `submission.csv`

# Reproducibility
In order to reproduce the result and generate `submission.csv`, execute the following commands:
* `pip install transformers pytorch_lightning sentencepiece datasets`
* `python ByT5_model_submission.py`

To run it on Euler/Leonhard first set up the environment:
* `module load gcc/8.2.0 python_gpu/3.8.`
* `module load eth_proxy`
* `pip install transformers pytorch_lightning sentencepiece datasets`

To run it with GPU support on Euler/Leonhard run:
* `bsub -n 4 -W 6:00 -R "rusage[mem=65536, ngpus_excl_p=1]"  python ByT5_model_submission.py`
