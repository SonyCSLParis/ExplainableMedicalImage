# Image Captioning for Interpretable Automatic Report Generation
Martina Galletti, Michela Proietti

## Usage
The _config_ folder contains the configuration file that allows changing the experiment's hyperparameters.
All the code is contained in the _src_ folder.
In particular, by setting the image model as pre-trained in the configuration file (the weights are saved in the _trained_models_ folder in the file
_image_model_512.pt_) and by setting the _pretrained_ variable for the combined model to False, it is possible to train it by running the _main.py_ file.
We could not load weights for the combined model because the file is too heavy, but we loaded the json file with the generated reports in the _trained_models_
folder.
