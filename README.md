# Multimodal classification that leverages embedding
Machine learning pipeline which aims to classify a multi-class multimodal dataset using early and late fusion with embeddings of the data provided. <br>
The goal of the project is to train a reliable classificator without the need of fine tune a model, and to verify the effectiveness of the various frozen models.

## Models used for embedding (WIP)
- BERT
- VIT
- openClip


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

## Classification pipeline (WIP)

### Early-fusion pipeline (WIP)

### Late-fusion pipeline (WIP)

## Classifier selection (WIP)