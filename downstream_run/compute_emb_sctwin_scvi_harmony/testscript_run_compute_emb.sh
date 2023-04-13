#!/usr/local_rwth/bin/zsh

module load cuda

source ~/.zshrc

pwd; hostname; date

conda activate scib_new
echo "Activated scib_new env"

# Define a timestamp function
timestamp() {
  date +"%T" # current time
}

timestamp # print timestamp

echo "Executing run_compute_emb.py"

python compute_emb_sctwin_scvi_harmony/run_compute_emb.py

echo "Execution run_compute_emb.py finished"

timestamp # print timestamp
