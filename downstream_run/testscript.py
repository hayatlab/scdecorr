import os
import subprocess


print("Current working directory: {0}".format(os.getcwd()))

home = "../cluster_logs/"

job_name ="testscript_run_compute_features_1"
mail_id="ritabrata.sanyal@rwth-aachen.de"

command = "sbatch -J " + job_name + " -o " + home+"cluster_out/" + job_name + "_out_train_bt.txt -e " + home+"cluster_err/" + job_name + "_err_train_bt.txt "
command += "--account rwth0535 -t 1:00:00 --gres gpu:1 --mem 20G -c 20 --partition c18g " + "testscript_run_compute_features_1.sh "+"--mail-type begin --mail-type end --mail-type fail --mail-user "+mail_id

os.system(command + " " + job_name)
print(command + " " + job_name)
