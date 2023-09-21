# Image Captioning for Interpretable Automatic Report Generation

- Martina Galletti - martina.galletti@sony.com
- Michela Proietti - mproietti@diag.uniroma1.it

## Context 

In this project, we implemented a model for Image Captioning for Interpretable Automatic Report Generation. We decided to combine a Res-NET model with XAI component together with a Bi-LSTM for explainable report generation. 

We decided to use multi-modality for three main reasons : 
- Multi-modal pre-training : significant progress but general domain (using : MS-COCO)
- Vision and Language : most used information in clinical domain 
- Multi-purpose joint representations of vision and language demonstrated effectiveness for a series of downstream tasks (i.e. diagnosis classification, medical image-report retrieval, medical visual question answering, radiology report generation)

## Model  

<img width="1107" alt="Screenshot 2023-09-21 at 11 20 54" src="https://github.com/SonyCSLParis/ExplainableMedicalImage/assets/45358914/6da072fe-2d82-47b7-9fc0-45c27889034c">

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

## References
- Xue, Y., Xu, T., Rodney Long, L., Xue, Z., Antani, S., Thoma, G. R., & Huang, X. (2018). Multimodal recurrent model with attention for automated radiology report generation. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2018: 21st International Conference, Granada, Spain, September 16-20, 2018, Proceedings, Part I (pp. 457-466). Springer International Publishing.
- Moon, J. H., Lee, H., Shin, W., Kim, Y. H., & Choi, E. (2022). Multi-modal understanding and generation for medical images and text via vision-language pre-training. IEEE Journal of Biomedical and Health Informatics, 26(12), 6070-6080.
- Ramirez-Alonso, G., Prieto-Ordaz, O., López-Santillan, R., & Montes-Y-Gómez, M. (2022). Medical report generation through radiology images: an Overview. IEEE Latin America Transactions, 20(6), 986-999.

