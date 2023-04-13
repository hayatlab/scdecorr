#!/usr/local_rwth/bin/zsh


source ~/.zshrc

pwd; hostname; date



# Define a timestamp function
timestamp() {
  date +"%T" # current time
}


conda activate LIGER
echo "Activated LIGER env"


timestamp # print timestamp

echo "Executing run_compute_emb.py"

python compute_emb_liger/run_compute_emb.py

echo "Execution run_compute_emb.py finished"

timestamp # print timestamp

