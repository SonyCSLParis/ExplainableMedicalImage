# Image Captioning for Interpretable Automatic Report Generation


- Martina Galletti - martina.galletti@sony.com
- Michela Proietti - mproietti@diag.uniroma1.it

## Dependencies & Python Version
- The requirements.txt file contains all the libraries needed to set up your environment and be able to run the code.
- This project requires Python 3.10.6 version to be run.

## Usage
- The _config_ folder contains the configuration file that allows changing the experiment's hyperparameters.
- In the *src* folder, you can find the folder with the data 
  - All the code to run the model is contained in the _src_ folder. In particular, by setting the image model as pre-trained in the configuration file (the weights are saved in the _trained_models_ folder in the file
  _image_model_512.pt_) and by setting the _pretrained_ variable for the combined model to False, it is possible to train it by running the _main.py_ file. We could not load weights for the combined model because 
  the file is too heavy, but we loaded the json file with the generated reports in the _trained_models_
  folder.
- In the *data* folder, you can find the folder with the data 
- In the *output* folder, the output is saved

## Data used 

- [Curated CXR report generation dataset](https://www.kaggle.com/datasets/financekim/curated-cxr-report-generation-dataset)
- Original Dataset Reference: OpenI, MIMIC-CXR
- Curated by authors of MediViLL (https://github.com/SuperSupermoon/MedViLL)
