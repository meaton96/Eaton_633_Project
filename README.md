# Predicting 30-Day Hospital Readmission Using Machine Learning Techniques

## 🏥 Project Overview

**The Challenge** Hospital readmissions within 30 days are a major financial burden, costing **$15k–$16k per case** and affecting ~14% of discharges (3.8M patients in 2019). Beyond cost, this is a key indicator of patient outcomes.

**Methodology**

-   **Dataset:** UCI Diabetes 130-US dataset (100k+ encounters, 1999–2008).
    
-   **Process:** Extensive cleaning, removal of information-leaking variables, and patient grouping.
    
-   **Models:** 8 models trained/tested using 5-fold cross-validation.
    

**Cost-Benefit Framework** We evaluated models not just on ROC-AUC, but on a custom savings function:

-   `R` (Readmission Cost):  **$16,300**
    
-   `cm` (Monitoring Cost):  **$330**
    
-   `e` (Effectiveness):  **0.33**
    


| Model | ROC-AUC | Est. Savings (Per Patient) | Note |
| :--- | :--- | :--- | :--- |
| **XGBoost** | **0.650** | **$37.69** | Best Performer |
| Random Forest | 0.647 | $35.01 | Close Second |
| KNN | 0.594 | $8.40 | Lowest Performer |


> **Conclusion:** Even modestly discriminative models can deliver meaningful economic value when evaluated through a cost-sensitive lens.

----------

## Readme
### Overview
The notebook code is largely illustrative and do not need to be run in order to view the aggregated results, or run any particular model testing pipeline.
The code is mostly imported from a pip package from my github repo and is run in the background, but all of the code that exists is in the collab notebook code reference section.
**The global cell and pip project import need to be run before any other cells can be run.**

### Notebook Sections 


The notebook is split into sections for each major piece to show code and reasoning for each step. The notebook contains the following sections:

1. Import Github project
2. Define global variables
3. Exploratory Data Analysis code, findings, and reasonings
4. Data Preparation, defines a few fixes before data is split
5. Data Cleaning Pipeline, demonstrates the code used for feature transformation and engineering
6. Modeling Runs, Code to run baseline, feature importance, hyperparameter sweeps, validation, and testing for each model
7. Aggregated results for each model to visualize or compare, this section imports pre-made metrics from previous runs and can be run independently of any modelling runs 

The only section needed to run models is the global section + any model section you wish. The modeling pipeline defined in these sub sections will handle the data reading, cleaning, training, validation, and testing.
You can define a few globals at the top to control if hyperparameter sweeps, baselines, and feature importances are run.
### Warning
I did all of the development locally, and while i can confirm that each of the cells runs in the collab notebook, several of the cells take a long time to execute. For example, the hyperparameter search for SVC takes ~10 minutes on my local machine, but >1.5 hours on collab.

You could clone the git repo and run the code locally if desired.

```
# Windows
python -m venv .venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source venv/bin/activate

pip -q install 'git+https://github.com/meaton96/Eaton_633_Project.git'
```


## Dataset
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008



