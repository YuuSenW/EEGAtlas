

BCIC-2A and BCIC-2B dataset is downloaded from https://www.bbci.de/competition/iv/#datasets and save into the `Datasets/downstream/Raw_data` folder, which organized as:
```
BCIC-2A
datasets/downstream/Raw_data/BCICIV_2a_gdf/A01E.gdf
...
datasets/downstream/Raw_data/BCICIV_2a_gdf/A09T.gdf

Only A0xT.gdf data
Then run the following command to preprocess the data:
python process_bcic2a.py

Next, move the processed BCIC_2aT_0_38HZ data folder to the Downstream/BCIC-2A directory.
------------------------------------------------------------------------------------------
BCIC-2B
datasets/downstream/Raw_data/BCICIV_2b_gdf/B0101T.gdf
...
datasets/downstream/Raw_data/BCICIV_2b_gdf/B0903T.gdf

Only B0x0xT.gdf data
Then run the following command to preprocess the data:
python process_bcic2b.py

Next, move the processed BCIC_2bT_0_38HZ data folder to the Downstream/BCIC-2B directory.
```


KaggleERN datasets can be downloaded from https://www.kaggle.com/c/inria-bci-challenge/data and save into the `datasets/downstream/KaggleERN` folder, which organized as:

```
datasets/downstream/KaggleERN/TrainLabels.csv
datasets/downstream/KaggleERN/true_labels.csv
datasets/downstream/KaggleERN/train/Data_S02_Sess01.csv
...
datasets/downstream/KaggleERN/train/Data_S26_Sess05.csv
...
datasets/downstream/KaggleERN/test/Data_S01_Sess01.csv
...
datasets/downstream/KaggleERN/test/Data_S25_Sess05.csv
```

PhysioP300 datasets can be downloaded from https://physionet.org/content/erpbci/1.0.0/ and save into the `datasets/downstream/erp-based-brain-computer-interface-recordings-1.0.0` folder, which organized as:
```
datasets/downstream/erp-based-brain-computer-interface-recordings-1.0.0/files/s01/rc01.edf
...
datasets/downstream/erp-based-brain-computer-interface-recordings-1.0.0/files/s11/rc01.edf
...
Then run the following command to preprocess the data:
```preprocess
python prepare_PhysioNetP300.py
```


For preparing Sleep-EDF dataset, you can run the following command to preprocess the data:
```
python prepare_sleep.py
```