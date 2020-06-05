
def classification_consistency_train_val(train_df, val_df, dependent_vars):
    ## remove unseen (in train) labels from validation set
    val_msk = None
    unseen_classes = {}
    for var_ in dependent_vars:
        classes = train_df[var_].unique()
        unseen_classes[var_] = val_df[~val_df[var_].isin(classes)][var_].unique()
        if val_msk:
            val_msk = val_msk & val_df[var_].isin(classes)
        else:
            val_msk = val_df[var_].isin(classes)

    len1 = val_df.shape[0]
    val_df = val_df[val_msk]
    len2 = val_df.shape[0]
    print('Dropped classes: {}'.format(list(unseen_classes)))
    print('{}% of validation set kept'.format(round(len2 / len1 * 100, 2)))
    return train_df, val_df
