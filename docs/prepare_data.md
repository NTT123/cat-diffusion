# Dataset

We use the Animal FacesHQ (AFHQ) dataset.

## Step 1: Download and unzip data

```sh
URL=https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0
ZIP_FILE=./data/afhq_v2.zip
mkdir -p ./data
wget -N $URL -O $ZIP_FILE
cd data
unzip afhq_v2.zip
```

## Step 2: Prepare tensorflow dataset

```sh
python prepare_tf_dataset.py --image-dir=data --output-dir=tfdata --image-size=64
```