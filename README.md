# Multimodal classification that leverages embedding
Machine learning pipeline which aims to classify a multi-class multimodal dataset using early and late fusion with embeddings of the data provided. <br>
The goal of the project is to train a reliable classificator without the need of fine tune a model, and to verify the effectiveness of the various frozen models.

## Models used for embedding (WIP)
- BERT
- VIT
- MedImageInsight


## RocoV2 dataset
The RocoV2 dataset is a multi-label multimodal medical dataset which, for every data, offers a Text+Image pair along with one or more Concept Unique Identifier, or CUI. Every CUI is linked to one or more Semantic Type. To reduce the complexity of the classification, we used the UMLS API to get the semantic type of every CUI, and selected only the ones with a relevant semantic type for Clinical classification. After that, we selected the CUIs with at least a frequency of 100.<br>
We identified 11 CUIs for our classification:
- C0000833: Abscess
- C0003962: Ascites
- C0006267: Bronchiectasis
- C0006826: Malignant neoplastic disease
- C0020295: Hydronephrosis
- C0025062: Mediastinal Emphysema
- C0031039: Pericardial effusion
- C0032285: Pneumonia
- C0032326: Pneumothorax
- C0497156: Lymphadenopathy
- C5203670: COVID19 (disease)


## Setup
1. Install the requirements.
2. Create a new folder called `data`. Inside, put three subdirectories `raw`, `interim`,`preprocessed`.
3. Download the [ROCOv2 dataset](https://zenodo.org/records/10821435) and put the files into the `raw` subfolder.
4. Download the [MedImageInsight model](https://huggingface.co/lion-ai/MedImageInsights/tree/main). DO NOT FOLLOW THE INSTRUCTIONS OF THIS PAGE; instead, do the following: <br>
    a. Clone the project
    b. got to `medimageinsightmodel.py` and change the following lines 
    ```python
    from MedImageInsight.UniCLModel import build_unicl_model
    from MedImageInsight.Utils.Arguments import load_opt_from_config_files
    from MedImageInsight.ImageDataLoader import build_transforms
    from MedImageInsight.LangEncoder import build_tokenizer
    ```
    to this:
    ```python
    from .MedImageInsight.UniCLModel import build_unicl_model
    from .MedImageInsight.Utils.Arguments import load_opt_from_config_files
    from .MedImageInsight.ImageDataLoader import build_transforms
    from .MedImageInsight.LangEncoder import build_tokenizer
    ```



## Run
Inside the virtual env, run the `main.py` file. You can use the following parameters:

| Parameter    | Arguments                                     | Description                                                                                               |
|--------------|-----------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| -text_model  | bert (default)                                | Text embedding model                                                                                      |
| -image_model | vit (default) medimageinsight                 | Image embedding model                                                                                     |
| -fusion      | both (default) early late late_mean late_meta | Fusion types: - early fusion - late fusion with mean and weighted mean - late fusion with meta-classifier |