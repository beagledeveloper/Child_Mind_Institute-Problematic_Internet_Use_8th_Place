# Child_Mind_Institute-Problematic_Internet_Use_8th_Place

Below you can find a outline of how to reproduce my solution for the "Child Mind Institute — Problematic Internet Use" competition.
https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use

If you run into any trouble with the setup/code or have any questions please contact me at knmtkzkdeveloper@gmail.com

---

## ARCHIVE CONTENTS

* data/train    : traing data (train.csv, series_train.parquet)
* data/test    : test data (test.csv, series_test.parquet)
* logs    : Standard output during training, including model features, the optimized QWK score, and its splitting thresholds
* models    : models(catboost, xgboost, lightgbm, IterativeImputer) trained using stratified 5-fold validation with 3 random seeds and serialized by joblib.By running test.sh, these models will reproduce the result of my solution.
* notebook    : notebooks capable of performing inference on Kaggle
* scripts    : traing_* .py - training scripts, predict_* .py - inference scripts
* submissions    : A .csv file of the inference results on the test data
* train.sh    : A shell script to run the training script located under the scripts directory
* test.sh    : A shell script to run the script predicting test data located under the scripts directory

## HARDWARE: (The following specs were used to create the original solution)

* Ubuntu 22.04 LTS
* CPU Intel Core i5-13500, 14 Cores
* RAM 64GB
* 1 x NVIDIA GeForce RTX 4060Ti (VRAM 16GB)

## SOFTWARE (python packages are detailed separately in `requirements.txt`):

* Python 3.10.12
* CUDA 12.3
* nvidia drivers v.545.23.06

## DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)

Please change the current directory to the top directory (where this README.md is located) of this repository, and then execute the following command:

1. Download the training and sample test data and unzip them.
```
$ kaggle competitions download -c child-mind-institute-problematic-internet-use
```
```
$ unzip child-mind-institute-problematic-internet-use.zip
```

2. Move the training and sample test data.
```
$ mv child-mind-institute-problematic-internet-use/train.csv child-mind-institute-problematic-internet-use/series_train.parquet data/train/
```
```
$ mv child-mind-institute-problematic-internet-use/test.csv child-mind-institute-problematic-internet-use/series_test.parquet data/test/
```

# Run Train Code.

1. Grant execution permission to the training shell scripts.
```
$ chmod +x train.sh
```

2. Run the training script. The trained and serialized model will be saved in the ./models directory.<br>   
Note: If serialized models already exist in ./models, they will be updated.
```
$ ./train.sh
```


# Run Inference Code.

1. Grant execution permission to the inference shell scripts.
```
$ chmod +x test.sh
```

2. Run the inference script. The predicted submission.csv file will be saved in the ./submissions directory.<br>   
Note: If predicted submission.csv file already exist in ./submissions, they will be updated.
```
$ ./test.sh
```
