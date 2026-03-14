# Data Preparation

In this project, in addition to the official **EXPR (Expression)** annotation data provided by the ABAW challenge, we have introduced the following two datasets for auxiliary training to enhance the model's generalization and robustness:
1. **AffectNet** (Only the manually annotated 8-class subset is used)
2. **RAF-DB**

If you wish to reproduce our experimental results, retrain the model, or evaluate our provided model weights, please ensure that the datasets are prepared according to the following format.

---

## 1. Original Dataset Directory Structure

Please download the datasets mentioned above and place them in the `Data` directory, which should be at the same level as the project code. Ensure the file hierarchy follows the structure below:

~~~text
. (Project Root)
├── ABAW10_expr_code\               # This project's code repository
└── Data\                           # Unified dataset storage directory
    ├── Dataset_ABAW10th\           # Official ABAW data
    │   ├── annotations\
    │   │   ├── AU_Detection_Challenge\...
    │   │   ├── EXPR_Recognition_Challenge\...
    │   │   └── VA_Estimation_Challenge\...
    │   ├── images\
    │   │   ├── batch1\...
    │   │   └── batch2\...
    │   └── videos\
    │       ├── batch1\...
    │       ├── batch2\...
    │       └── batch3\...
    ├── Dataset_AffectNet\          # Original AffectNet data
    │   ├── Manually_Annotated_file_lists\
    │   │   ├── training.csv
    │   │   └── validation.csv
    │   └── Manually_Annotated_Images\
    │       └── <img_folders>\<imgs>
    └── Dataset_RAF\                # Original RAF-DB data
        ├── train\
        │   ├── 0\...
        │   ├── 1\...
        │   └── ...
        └── valid\
            ├── 0\...
            ├── 1\...
            └── ...
~~~

---

## 2. Dataset Remapping and Unification

After preparing the datasets, to improve data loading efficiency and format consistency, we provide two automated data extraction scripts. 

These scripts use **fixed random seeds** to ensure that the dataset splitting (e.g., the 10-fold cross-validation split for AffectNet) is exactly the same every time, thereby guaranteeing the **reproducibility** of our experiments.

Please run the following scripts in your terminal:

~~~bash

# Extract and remap the RAF-DB dataset
python -m pipeline1_create_IMG_Dataset.extract_img_from_RAFDB

# Extract and remap the AffectNet dataset
python -m pipeline1_create_IMG_Dataset.extract_img_from_AffectNet
~~~

### Execution Results
After running these scripts, the AffectNet and RAF-DB datasets will be automatically extracted, renamed, and remapped into a unified directory structure based on their categories. The new path structure will be:
`Data/Dataset_IMG/<folder_name>/<class_name>/<img_name>`

> **Note**
>
> Once the scripts finish executing and the `Dataset_IMG` directory is successfully generated, we will **no longer need** to read the original `Dataset_AffectNet` and `Dataset_RAF` formats in the subsequent training and testing pipelines.
>
> The class labels are remapped to match the 8-class EXPR label order:
Neutral, Anger, Disgust, Fear, Happiness, Sadness, Surprise, Other
>
> The official ABAW EXPR dataset keeps its original frame-level structure and is not merged into Dataset_IMG.