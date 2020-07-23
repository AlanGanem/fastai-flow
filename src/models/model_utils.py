from fastai import *
from fastai.tabular import *

def tolist(x):
    if x.__class__ in [list, tuple, set, dict]:
        x = list(x)
    else:
        x = [x]
    return x
    

def fit_learner(learner, fastai_cycles, **fitargs):
    # find best learning rate
    # learner.lr_find()
    # learner.recorder.plot()
    # lr = 2 * 1e-2
    # fit
    learner.fit_one_cycle(fastai_cycles, **{k:v for k,v in fitargs.items() if v})
    return learner

def create_multiclass_learner(
        db,
        fastai_layers_setup,
        cat_emb_szs,
        fastai_dropout,
        fastai_min_delta,
        fastai_patience,
):
    learner = tabular_learner(
        db,
        layers=fastai_layers_setup,
        emb_szs=cat_emb_szs,
        emb_drop=fastai_dropout,
        metrics=[accuracy],
        callback_fns=[partial(
            callbacks.tracker.EarlyStoppingCallback,
            monitor='accuracy',
            min_delta=fastai_min_delta,
            patience=fastai_patience)]
    )
    #TODO: include prameter for loss function
    #learner.loss_fn = F.kl_div#F.smooth_l1_loss
    return learner

def create_multiclass_db(train_data, val_data, cat_features, num_features,dependent_vars, fastai_bs):
    ## set train-val split slice
    idx1 = train_data.shape[0]
    idx2 = val_data.shape[0]
    val_slice = range(idx1,idx1+idx2)

    ## concatenate train and validation as required in the fast.ai API
    full_train_data = pd.concat([train_data,val_data]).reset_index()

    ## create data bunch
    procs = [FillMissing, Categorify, Normalize]
    db = (TabularList.from_df(
        path='.',
        df=full_train_data,
        procs=procs,
        cat_names=cat_features,
        cont_names = num_features,
    )
          .split_by_idx(val_slice)
          .label_from_df(cols=dependent_vars[0])
          .databunch(bs=fastai_bs, num_workers = 0)) # set_workers = 0 only for windows. dismiss this argument on linux machines
    
    return db


def get_model_ready_to_validate(model, data, dependent_var):
    # returns model with valid_dl = datalist of data
    data = data.copy()
    # filter unseen classes
    #allowed_labels = list(model.data.classes)
    #data = data[data[dependent_var].isin(allowed_labels)]
    data_test = TabularList.from_df(
        data,
        cat_names=model.data.cat_names,
        cont_names=model.data.cont_names,
        procs=model.data.processor[0].procs
    ).split_none().label_from_df(cols=dependent_var)  # .split_by_idx(range(0,data.shape[0]))

    data_test = data_test.databunch(num_workers = 0)
    model.data.valid_dl = data_test.train_dl
    return model


def get_preds_new_data(model, inf_data, dependent_var = None):
    #create pred tabular list and add to inference model as test set
    tabList = TabularList.from_df(
        inf_data,
        cat_names=model.data.cat_names,
        cont_names=model.data.cont_names,
        procs=model.data.processor[0].procs
    )

    # get preds
    model.data.add_test(tabList)
    #might raise error if another python kernel is running, like jupyter(????)
    preds = model.get_preds(ds_type = DatasetType.Test)
    arr_proba_preds = preds[0].numpy()
    max_idxs = preds[0].max(dim=1)
    idxs = [i.item() for i in max_idxs[1]]
    proba_preds = [i.item() for i in max_idxs[0]]
    class_preds = [model.data.classes[i] for i in idxs]

    return {'proba_preds':proba_preds, 'class_preds':class_preds, 'arr_proba_preds':arr_proba_preds,'idx_map':model.data.classes}

#def custom_most_confused(cfs_matrix, labels):
if __name__ == '__main__':
    pass