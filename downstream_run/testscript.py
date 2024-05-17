import os
import subprocess


print("Current working directory: {0}".format(os.getcwd()))

home = "../cluster_logs/"

job_name ="compute_emb_sctwin_scvi_harmony"
mail_id="ritabrata.sanyal@rwth-aachen.de"

command = "sbatch -J " + job_name + " -o " + home+"cluster_out/" + job_name + "_out_bt.txt -e " + home+"cluster_err/" + job_name + "_err_bt.txt "
command += "--account rwth0535 -t 6:00:00 --nodes 1 --mem 30G -c 25 --gres gpu:1 --partition c18g " + "compute_emb_sctwin_scvi_harmony/testscript_run_compute_emb.sh "+"--mail-type begin --mail-type end --mail-type fail --mail-user "+mail_id

os.system(command + " " + job_name)
print(command + " " + job_name)
