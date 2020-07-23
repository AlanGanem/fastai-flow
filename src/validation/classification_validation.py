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

def get_true_label_proba(y_true_idx_list, proba_preds_arr):
    proba_true = []
    for i in range(len(y_true_idx_list)):
        try:
            proba_true.append(proba_preds_arr[i, y_true_idx_list[i]])
        except (IndexError,KeyError):
            proba_true.append(np.nan)

    return np.array(proba_true)

def calibration_curve(data):
    thresh = np.linspace(0, 1, 101)
    proba_dists = [data['_CLASS_PROBA'].between(i,i+.01) for i in thresh]
    density = pd.Series(i.mean() for i in proba_dists)
    acc = pd.Series([data[msk]['_GOT_RIGHT'].mean() for msk in proba_dists])
    efficiency_df = pd.DataFrame([thresh,acc,density]).T
    efficiency_df.columns = ['probability',
                             'accuracy','density']
    return efficiency_df


def classification_report_df(y_true, y_pred):
    cls_report = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        output_dict=True)
    classification_report_df = pd.DataFrame(cls_report).T
    return classification_report_df


def confusion_df(y_true, y_pred, labels):
    cfs_mtrx = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=model.data.classes
    )
    confusion_df = pd.DataFrame(cfs_mtrx, columns=labels, index=labels)
    return confusion_df

def validation_dict(
        model,
        data,
        dependent_var,
        reports = ['calibration_curve','pareto_raking','sklearn_classification_report','confusion_matrix']
):
    #creates a dictionary with multiple validation reports
    data = data.copy()
    val_dict = {}
    #Fill validation NaNs with str "NaN"
    data[dependent_var] = data[dependent_var].fillna('NaN')
    model = get_model_ready_to_validate(model, data, dependent_var)
    # check labels known by the model (in case there are new labels)
    allowed_labels = list(model.data.classes)
    unseen_classes_msk = ~data[dependent_var].isin(allowed_labels)
    unseen_classes_mean = unseen_classes_msk.mean()
    classes = model.data.classes

    ## make predictions
    preds = get_preds_new_data(model,data)
    class_preds = preds['class_preds']
    proba_preds = preds['proba_preds']
    proba_array = preds['arr_proba_preds']

    # get proba for true label
    idx_to_class = dict(enumerate(classes))
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    y_true_idx_list = [class_to_idx[i] if i in class_to_idx else np.nan for i in data[dependent_var].values.tolist()]
    true_label_proba = get_true_label_proba(y_true_idx_list,proba_array)

    #create validation columns
    data['_UNSEEN_CLASS'] = unseen_classes_msk
    data['_TRUE_LABEL_PROBA'] = true_label_proba
    data['_CLASS_PREDS'] = class_preds
    data['_CLASS_PROBA'] = proba_preds
    data['_GOT_RIGHT'] = (data[dependent_var] == data['_CLASS_PREDS']).astype(int)
    accuracy = data['_GOT_RIGHT'].mean()#learner.validate() for custom metric
    expected_accuracy = data['_CLASS_PROBA'].mean()
    #append to dict
    val_dict['accuracy'] = accuracy
    val_dict['expected_accuracy'] = expected_accuracy
    val_dict['data_df'] = data
    val_dict['unseen_labels'] = unseen_classes_mean

    # extra reports
    if 'calibration_curve' in reports:
        val_dict['calibration_curve_df'] = calibration_curve(data)
    if 'pareto_raking_df' in reports:
        val_dict['pareto_raking_df'] = pareto_ranking(data[dependent_var].values, proba_array, classes)[1]
    if 'sklearn_classification_report' in reports:
        #make label-wise classification repport (sklearn)
        y_true = data[dependent_var].values.flatten(),
        y_pred = data['_CLASS_PREDS'].flatten(),
        val_dict['classification_report_df'] = classification_report_df(y_true, y_pred)
    if 'confusion_matrix' in reports:
        val_dict['confusion_matrix_df'] = confusion_df(y_true, y_pred, labels = model.data.classes)

    return val_dict
