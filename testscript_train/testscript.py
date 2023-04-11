import os
import subprocess


print("Current working directory: {0}".format(os.getcwd()))

home = "../cluster_logs/"

job_name ="cl_btwins_multidom_dsbn_integration_crosstissue_immune_train"
mail_id="ritabrata.sanyal@rwth-aachen.de"

command = "sbatch -J " + job_name + " -o " + home+"cluster_out/" + job_name + "_out_train_bt.txt -e " + home+"cluster_err/" + job_name + "_err_train_bt.txt "
command += "--account rwth0535 -t 15:00:00 --mem 30G -c 15 --gres gpu:2 --partition c18g " + "testscript_btwins_crosstissue_immune.sh "+"--mail-type begin --mail-type end --mail-type fail --mail-user "+mail_id

os.system(command + " " + job_name)
print(command + " " + job_name)
