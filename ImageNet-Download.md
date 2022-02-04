# Download and Prepare ImageNet with Kaggle Step by step

The official kaggle website for ImageNet dataset is [here](https://www.kaggle.com/c/imagenet-object-localization-challenge/data).

1. run `pip3 install kaggle`.

2. Register an account at [kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data).

3. Agree the terms and conditions on the [dataset page](https://www.kaggle.com/c/imagenet-object-localization-challenge/data).
   
3. Go to your account page (https://www.kaggle.com/<username>/account). Select 'Create API Token' and this will trigger the download of `kaggle.json`, a file containing your API credentials.

4. Copy this file into your server at `~/.kaggle/kaggle.json`

5. run command `chmod 600 ~/.kaggle/kaggle.json` and make it visible only to yourself.

6. run command `kaggle competitions download -c imagenet-object-localization-challenge`.

7. Unzip the file `unzip -q imagenet-object-localization-challenge.zip` and `tar -xvf imagenet_object_localization_patched2019.tar.gz`

8. Enter the validation set folder `cd ILSVRC/Data/CLS-LOC/val`

7. run scrip `sh/prepare_imagenet.sh` provided by the PyTorch repository, to move the validation subset to the labeled subfolders.

8. The `--datadir` you need to feed into `train.py` is the path of the folder ILSVRC.
