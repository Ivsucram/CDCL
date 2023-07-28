# Towards Cross-Domain Continual Learning
# CDCL: Cross-Domain Continual Learning

## Configuring datasets

The code will automatically download most datasets, as per file `src/utils/dataset_factory.py`. You can check the file to see which address are being used to download the datasets.

### Office31

The only dataset that can't be automatically downloaded and configured by our scripts is the Office31 Dataset.

Please, manually download the dataset from https://www.hemanthdv.org/officeHomeDataset.html and unzip in the folder ./data/office_home. (Create the folder, if it doesn't exist. If you run our scripts, it will automatically create this folder for you).


## Configuring the Python environment

Install the required libraries in the `requirements.txt` file. You can do so by using the following command:

```
pip install -r requirements.txt
```

## Running experiments

You can re-run all experiments by executing the file `cdcl_paper.sh`. You might need to edit its hidden characters (space character) if you are running it in Windows or Linux.
