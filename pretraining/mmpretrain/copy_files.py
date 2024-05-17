import os

destination = '/BS/dnn_interpretablity_robustness_representation_learning_2/work/my_projects/github_repos/how-to-probe/pretraining/mmpretrain/'

f = open('copy_files.txt')
for idx, file in enumerate(f):
    print(idx, file, "copying ...")
    os.system('cp -r {} {}'.format(file, destination))

print("COMPLETE!")
