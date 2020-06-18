from fastai import *
from fastai.tabular import *
from ..models.model_utils import get_model_ready_to_validate, get_preds_new_data
from sklearn.metrics import classification_report,confusion_matrix


def pareto_ranking(y_true, y_proba_pred, classes):
    #creates a pareto ranking of corrected predicitons
    proba_df = pd.DataFrame(y_proba_pred, columns=classes)

    proba_df = pd.DataFrame(proba_df.apply(lambda x: list(x.sort_values(ascending=False).index), axis=1))
    proba_df.columns = ['ranking_list']
    proba_df['ground_truth'] = y_true
    def f(x):
        try:
            return x['ranking_list'].index(x['ground_truth'])
        except:
            return len(x['ranking_list'])-1
    ranking_pos = proba_df.apply(lambda x: f(x), axis=1)
    ranking_pos_counts = pd.DataFrame(ranking_pos.value_counts(normalize = True).sort_index(), columns = ['freq'])
    ranking_pos_counts['cumsum'] = ranking_pos_counts['freq'].cumsum()
    return proba_df, ranking_pos_counts


def get_losses(model, df, dependent_var):
    #returns df order by  '_LOSS'
    model, df = get_model_ready_to_validate(model,df, dependent_var)

    interp = ClassificationInterpretation.from_learner(model)
    top_losses_result = interp.losses.numpy()
    return_df = df.assign(_LOSS=top_losses_result)
    return return_df

def get_true_label_proba(y_true_idx, proba_preds_arr):
    proba_true = []
    for i in range(y_true_idx.shape[0]):
        if y_true_idx[i] == -1:
            proba_true.append(np.nan)
            continue
        try:
            proba_true.append(proba_preds_arr[i, y_true_idx[i]])
        except (IndexError,KeyError):
            proba_true.append(np.nan)


    return np.array(proba_true)

def validation_dict(model, data, dependent_var):
    #creates a dictionary with multiple validation artefacts
    data = data.copy()

    data[dependent_var] = data[dependent_var].fillna('NaN')
    model = get_model_ready_to_validate(model, data, dependent_var)

    allowed_labels = list(model.data.classes)
    unseen_classes = ~data[dependent_var].isin(allowed_labels)
    validatable = unseen_classes.mean()

    #
    #interp = ClassificationInterpretation.from_learner(model, ds_type = DatasetType.Valid)
    classes = model.data.classes
    #confusion_df = pd.DataFrame(interp.confusion_matrix(), columns = classes, index = classes)
    #most_confused_df = pd.DataFrame(interp.most_confused(), columns = ['actual','predicted','occurrences'])


    ## make metrics plot
    preds = get_preds_new_data(model,data)

    class_preds = preds['class_preds']
    proba_preds = preds['proba_preds']
    proba_array = preds['arr_proba_preds']
    # get proba for true label
    idx_to_class = dict(enumerate(classes))
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    # tweak to make a valid INTNAN. appending nan would make the entire column float and raise error in indexing
    y_true_idx = data[dependent_var].apply(
        lambda x: class_to_idx[x] if x in class_to_idx else '#INTNAN')
    true_label_proba = get_true_label_proba(y_true_idx,proba_array)
    #create validation columns
    data['_UNSEEN_CLASS'] = unseen_classes
    data['_TRUE_LABEL_PROBA'] = true_label_proba
    data['_CLASS_PREDS'] = class_preds
    data['_CLASS_PROBA'] = proba_preds
    data['_GOT_RIGHT'] = data[dependent_var] == data['_CLASS_PREDS']
    accuracy = data['_GOT_RIGHT'].mean()#model.validate()
    expected_accuracy = data['_CLASS_PROBA'].mean()
    ## create column for losses and sort for top losses
    #data = get_losses(model, data, dependent_var)
    # efficiency performance
    total = data.shape[0]
    thresh = np.linspace(0, 1, 101)
    eff = [data[data['_CLASS_PROBA'] >= i]['_GOT_RIGHT'].sum() / total for i in thresh]
    efficiency_df = pd.DataFrame([thresh,eff]).T
    efficiency_df.columns = ['threshold','accuracy']

    # make ranking pareto
    pareto_raking_df = pareto_ranking(data[dependent_var].values, proba_array, classes)[1]

    #make label-wise classification repport (sklearn)

    cls_report = classification_report(
        y_true = data[dependent_var].values.flatten(),
        y_pred = np.array(class_preds).flatten(),
        output_dict = True)
    classification_report_df = pd.DataFrame(cls_report).T

    cfs_mtrx = confusion_matrix(
        y_true=data[dependent_var].values.flatten(),
        y_pred=np.array(class_preds).flatten(),
        labels = model.data.classes
    )

    confusion_df = pd.DataFrame(cfs_mtrx, columns=model.data.classes, index=model.data.classes)

    return {
        'classification_report_df':classification_report_df,
        'pareto_ranking_df':pareto_raking_df,
        'efficiency_df':efficiency_df,
        #'most_confused_df':most_confused_df,
        'confusion_df':confusion_df,
        'accuracy':accuracy ,
        #'loss': accuracy[1].item(),
        'expected_accuracy':expected_accuracy,
        'data_df': data,
        'validation_class_consistency':validatable
    }
