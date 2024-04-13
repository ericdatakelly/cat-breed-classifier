# Cat Breed Classifier

**Notebook and script to train on the Oxford *Cats and Dogs Breeds Classification* dataset and predict cat breeds from thecatapi.com**

The notebook outputs up to three potential breed mixes because many cat pictures are not of full breeds.

## Installation
Install and activate the virtual environment with either Mamba or Conda using the environment file.
```bash
cd cat-breed-classifier
mamba env create -f environment.yaml
mamba activate cat-breed-classifier
```

### Classification
Get a free API key from [thecatapi.com](https://thecatapi.com/) and put it in your user folder (`~/.config/catapi/api_key.txt`).

Run the notebook.

### Retraining
Get [Cats and Dogs Breeds Classification Oxford Dataset](https://www.kaggle.com/datasets/zippyz/cats-and-dogs-breeds-classification-oxford-dataset) and put the annotation and image folders into the data folder of this project, e.g., `data/cat_dog_breeds_oxford/annotations/trainval.txt`

Run the `cat_breed_train.py` script. It will save a loss plot and a new model in the same directory.

View results of the new model by updating the model name in the notebook and running all cells.
