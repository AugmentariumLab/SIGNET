Here we provide demo scripts to process the Stanford Light Field dataset (http://lightfield.stanford.edu/lfs.html) and train the neural representation with SIGNET.

Step 1:
Download specific datasets to your local directory. For example, put the unzipped Lego Knights images into a folder called "{ROOT}/data/lego".

Step 2:
Run "python preprocess.py --img_dir {ROOT}/data/lego/ --save_dir {ROOT}/patch_data/lego_patches"

This step is performed to preprocess the light field dataset into small batches convenient for network training.
You may adjust the batch size within this script, or increase the batch size in the dataloader during training.
Here the validation image is hardcoded as the "01_01" view. Feel free to adjust according to your needs.

Step 3:
Run "python train_net.py --root_dir {ROOT} --exp_name lego_test --trainset_dir {ROOT}/patch_data/lego_patches"

Training should begin following this command.
You should find the trained weights and validation output at folder "{ROOT}/{exp_name}" during training.
The image resolution is assumed to be less than 1024x1024. If you work on data with higher resolution, please use the "--img_W" and "--img_H" arguments to adjust accordingly.

To decode light field view at (u, v) from trained weights, please run "python eval_net.py --exp_dir {ROOT}/{exp_name} -u u -v v"

The purpose of these scripts is to provide a simple implementation that helps you kickstart your own experiments.
If you encounter any error or have any suggestion, please don't hesitate to reach out to yfeng97@umd.edu. Thank you!
