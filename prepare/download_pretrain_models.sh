#!/bin/bash
echo -e "The pretrained models will stored in the 'models' folder\n"
mkdir -p models
python -m gdown.cli "https://drive.google.com/uc?id=1n6kRb-d2gKsk8EXfFULFIpaUKYcnaYmm"

echo -e "Please check that the md5sum is: 7b6d8814f9c1ca972f62852ebb6c7a6f"
echo -e "+ md5sum tmr_models.tgz"
md5sum tmr_models.tgz

echo -e "If it is not, please rerun this script"

sleep 5
tar xfzv tmr_models.tgz

echo -e "Cleaning\n"
rm tmr_models.tgz

echo -e "Downloading done!"
