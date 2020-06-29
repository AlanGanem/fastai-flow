# IVA-classifier project report 

The goal of the project is to classify purchase order IVAs (tags that combine multiple taxes) based on multivariate historical data using fast.ai's framework for dense neural netowrks (DNN) and symbolic AI (user defined rules). The same model can be used to audit possible wrong IVAs in invoices.

## Checklist

Mark which tasks have been performed

- [ ] **Summary:** you have included a description, usage, output, accuracy and metadata of your model.
- [X] **Pre-processing:** you have applied pre-processing to your data and this function is reproducible to new datasets.
- [X] **Feature selection:** you have performed feature selection while modeling.
- [ ] **Modeling dataset creation:** you have well-defined and reproducible code to generate a modeling dataset that reproduces the behavior of the target dataset.This pipeline is also applicable to generate the deploy dataset.
- [X] **Model selection:** you have chosen a suitable model according to the project specification.
- [X] **Model validation:** you have validated your model according to the project specification.
- [ ] **Model optimization:** you have defined functions to optimize hyper-parameters and they are reproducible.
- [ ] **Peer-review:** your code and results have been verified by your colleagues and pre-approved by them.
- [X] **Acceptance:** this model report has been accepted by the Data Science Manager. State name and date.

---

## Summary

The model is designed to classify IVA based on certain features and user defined rules

## Usage

In order to have the best accuracy and also action on taxes as soon as possible for the First Time Right KPI, the predictions are going to be made on a daily basis, and the algorithm will be retrained on a weekly basis.

Make sure that ```python==3.6.9``` and Miniconda3 are installed on your machine and follow the steps:

1. In order to prepare your machine to run the model, first **create** and **activate** a virtual environment and install the requirements by running

```conda env create --file environment.yml ```

And then all dependencies should be installed.

2. When receiving any data extracted from the Data Engineering team, you should train the model once every week. With the historical data prepared, indicate the file and run

```python3 src/models/train_model.py```

This will create a pkl file to serialize the latest trained model

3. With the latest trained model pkl file, you will then finally run the prediction by calling

```python3 src/models/predict_model.py```

This will output a list under ```data/output/[TODAY's DATE]``` that is what should be then placed onto the business folder:

```\\acswpj6\su\Nova\Contract Operations\Compartilhado\102_Pré-aprovador de Pedidos\Machine Learning IVA```


---

## Output



The model has 3 main functionalities: Train, predict and auditting:
	
    - Train will train the model by running ../src/models/train_model.py with it's respective arguments such as input path and model hyperparms.
     The serialized trained model will be saved in ../models/iva_model.pkl and model info (metadata and validation) will be saved in 
     ../models/train_metadata/fitted_YYYY_mm_dd
    
    - Predict will infere model outputs by running ../src/models/predict_model.py with it's respective arguments such as input path. predictions 
     will be saved to ../models/outputs

    - Auditing will check top losses of model and sugest them as possible anomalies by running ../src/models/auditing_model.py with it's respective arguments such as input path. predictions 
     will be saved to ../models/auditing_outputs

---

## [Metadata](docs/project_metadata.json)

---

## Performance Metrics

Model's performance will be measured by accuracy, precision and recall, and all other metrics contained in ```models/train_metadata```

## Pre-processing

All preprocessing steps are contained in ```src/data/make_dataset.py```

1. all columns are casted to string
2. strip left zeros and blank spaces in str columns
3. drop duplicates according to 'duplicates_subset' parameter
4. empty IVAMIROs are dropped
5. 'OrderType' feature is created using the first 2 numbers of 'Nºdopedido'
6. cast closest match of 'Data do Documento' to datetime using 'date_format'
7. if 'drop_nat' is set to True, dropa rows containing NaT values for date


## Feature selection

Some feature selection is performed with pandas, like extracting "Descrição do Material"'s first word and checking wether the transaction occured
between different satates of Brazil. Since neuron Nets are used, as well as embedding layers, most feature engineering and selection is done by the 
model's architecture

**Features used**

Features can vary and features used in each trained model can be found in ```../models/train_metadata/fitted_YYYY-mm-dd/metadata.txt```

1. Material:Str -> material SAP code
2. Filial:Str -> unit SAP code
3. PEP:Str -> PEP element SAP code
4. Fornecedor:Str -> Suppliers SAP code
5. UF:Str -> Supplier state (Federal Unit)
6. TpImposto:Str -> Supplier's tax regime
7. desc1:Str -> First word of Material description
8. InterUF:Bool -> Checks whether Filial and Fornecedor are not in the same UF
9. IVAPC:Str -> IVA from the purchase order (can be dismissed with around 2-3% accuracy decrease)
10. IVAMIRO:Str -> Invoice's Tax Code (DEPENDENT VARIABLE)

## Modeling

The DNN architecture can vary, but basicaly it is composed of:
1. embedding layers for each categorical variables (embedding sizes are hyperparameters to the model).
2. The outputs of those layers are then concatenated and passed through a batch-norm layer and (optionally) a dropout layer (dropout usually = 0.1)
3. this output feeds a fully connected layer (usually 10~20 neurons) with another round of batch-norm. 
4. Finally, the output is fowarded to a softmax layer. 
Optimization is performed minimizing negative log-likelihood with fast.ai's default for stochastic gradient decent algorithm under the fit_one_cycle policy. For more information on the architeccture please refer to fastai.tabular.

## Model selection

The algorithm was choosen from experimentations done with a vast range of classification algorithms such as tree-based (RandomForests and gradient boosting), Naive Bayes, Logistic classification and KNN. DNN not only performed better than others, but fundamentally deals better with learnable feature extraction/engineering and
non-linear interactions between features. Also, the embedding layer is super usefull to map (in a supervised fashion) categorical features with high cardinality into lower dimensional, continuous spaces.
Some faetures, like ```Material``` have arround 65.000 unique entries, which makes tree algorithms unfeasible without some dimension reduction technique such as PCA, UMAP, t-SNE

## Model validation

Running '''../src/models/validate_model.py''' generates a couple of usefull validations like:
	
1. Confusion Matrix
2. Daily performance
3. Predicted Classes Pareto
4. Accuracy by threshold
5. Classification_report from sklearn

The validation files (.xlsx) are saved in ```../models/validation_outputs/<runtime_timestamp>```

## Model optimization

Optimization is still on its early stages and should be contained on a new release.

## Drifting and Retraining

The model should me retrained every week, taking into account new materials and suppliers.

## Foreseen improvements

- include hierarchical inference (rule based first, inference later)
- apply baesyan hyperparameter tuning
- implement output interpretation or interpretable architecture, such as SHAP, LIME, TabNET, etc.
- develop the auditing.py module together with the business team
- include logging through the code
- design unit tests