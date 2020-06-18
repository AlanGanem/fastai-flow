from copy import deepcopy
from fastai.tabular import *
from fastai import *
#Base classes
from src.Base import BasePipeline
# processing blocks
from src.validation import classification_validation
from src.data.io.loader import load_csv
from src.data.io.saver import make_fastai_serializable
from src.preprocessing.consistency import classification_consistency_train_val
from src.preprocessing.preprocess import drop_dependent_nan, df_split, str_caster
from src.models.model_utils import create_multiclass_db, create_multiclass_learner, fit_learner, tolist, get_preds_new_data, get_model_ready_to_validate


class ClassificationPipeline(BasePipeline):

    def __init__(
            self,
            model_id,
            cat_features,
            num_features,
            dependent_vars,
            train_frac_split,
            date_col = None,
            fastai_layers_setup=[20],
            fastai_dropout=0.1,
            cat_emb_szs=None,
            fastai_cycles=1,
            fastai_bs=256,
            fastai_patience=3,
            fastai_min_delta=0.001,
            validate=True,
            # loading args
            pd_sep=';',
            pd_encoding='ansi',
            #  split args
            val_days=21,
            test_days=14,
            **kwargs
    ):

        super().__init__(model_id)
        self.kwargs = kwargs
        self.fastai_layers_setup = tolist(fastai_layers_setup)
        self.fastai_dropout = fastai_dropout
        # TODO create optional param for embedding size setup (allow optimization)
        self.cat_emb_szs = cat_emb_szs
        self.fastai_cycles = fastai_cycles
        self.fastai_bs = fastai_bs
        self.fastai_patience = fastai_patience
        self.fastai_min_delta = fastai_min_delta
        # loading args
        self.pd_sep = pd_sep
        self.pd_encoding = pd_encoding
        #  split args
        self.val_days = val_days
        self.test_days = test_days
        self.pd_encoding = pd_encoding
        self.train_frac_split = train_frac_split
        self.cat_features = tolist(cat_features)
        self.dependent_vars = tolist(dependent_vars)
        self.num_features = tolist(num_features)
        self.date_col = date_col
        return

    def build_features(self, data):
        # feature engeineering can be defined inheriting (class CustomPipe(ClassificationPipeline))
        # and the method build features must be overwritten, taking df and outputting df.
        return data

    def load_and_preprocess_predict(self, PREDICT_DATA_PATH:PathOrStr = None, data = None):

        if  data.__class__ == type(None):
            print('loading data...')
            data = load_csv(PREDICT_DATA_PATH, encoding=self.pd_encoding, sep=self.pd_sep)

        # feature eng can be defined via heritance
        data = self.build_features(data)

        return data

    def load_and_preprocess_validate(self, VALIDATE_DATA_PATH: PathOrStr = None, data=None):
        data = self.load_and_preprocess_predict(VALIDATE_DATA_PATH,data)
        #keep NaNs
        # cast labels to str
        data = str_caster(data, self.dependent_vars)
        print(data.dtypes)
        return data

    def load_and_preprocess_fit(self, TRAIN_DATA_PATH:PathOrStr = None, data = None):

        if data.__class__ == type(None):
            print('loading data...')
            data = load_csv(TRAIN_DATA_PATH, encoding=self.pd_encoding, sep=self.pd_sep)

        # feature eng can be defined via heritance
        data = self.build_features(data)

        #drop label nans
        print('dropping NaNs...')
        data = drop_dependent_nan(data, self.dependent_vars)
        #cast labels to str
        data = str_caster(data, self.dependent_vars)
        print('Splitting data...')
        if self.date_col:
            data[self.date_col] = pd.to_datetime(data[self.date_col], errors = 'coerce')
        total_len = data.shape[0]
        train_data, test_data = df_split(data, train_frac=self.train_frac_split, date_col=self.date_col,
                                         test_days=None, start_from=None)
        train_data, val_data = df_split(train_data, train_frac=self.train_frac_split, date_col=self.date_col,
                                        test_days=None, start_from=None)
        # print size fractions
        print('train size: {}%\nvalidation size: {}%\ntest size: {}%'.format(
            round(train_data.shape[0] / total_len, 2) * 100,
            round(val_data.shape[0] / total_len, 2) * 100,
            round(test_data.shape[0] / total_len, 2) * 100, ))

        ## remove unseen (in train) labels from validation set # src\consistency\classification_consistency_train_val
        train_data, val_data = classification_consistency_train_val(train_data, val_data, self.dependent_vars)
        # little tweak to make fastai robust to nans on validation set
        train_data = train_data.append(train_data[self.dependent_vars].mode())
        return train_data, val_data, test_data

    def fit(self, TRAIN_DATA_PATH = None, data = None):
        train_data, val_data, test_data = self.load_and_preprocess_fit(TRAIN_DATA_PATH = TRAIN_DATA_PATH, data = data)
        # create databunch
        db = create_multiclass_db(train_data, val_data, self.cat_features, self.num_features, self.dependent_vars,
                                  self.fastai_bs)

        # create learner
        self.learner = create_multiclass_learner(
            db,
            self.fastai_layers_setup,
            self.cat_emb_szs,
            self.fastai_dropout,
            self.fastai_min_delta,
            self.fastai_patience,
        )
        db.show_batch()
        # fit learner
        fit_learner(self.learner, self.fastai_cycles)
        return self

    def validate(self, VALIDATE_DATA_PATH = None, data = None):
        # validate model
        learner = deepcopy(self.learner) #make a deepcopy to avoid saving errors
        data = self.load_and_preprocess_validate(VALIDATE_DATA_PATH, data)
        validation_dict = classification_validation.validation_dict(learner, data, self.dependent_vars[0])
        return validation_dict

    def predict(self, PREDICT_DATA_PATH:PathOrStr = None, data:Union[pd.DataFrame,None] = None):
        learner = deepcopy(self.learner)
        data = self.load_and_preprocess_predict(PREDICT_DATA_PATH = PREDICT_DATA_PATH, data = data)

        preds = get_preds_new_data(learner, data)

        data['proba_pred'] = preds['proba_preds']
        data['class_pred'] = preds['class_preds']
        return data

    def predict_proba(self, PREDICT_DATA_PATH:PathOrStr = None, data = None):

        learner = deepcopy(self.learner)
        data = self.load_and_preprocess_predict(PREDICT_DATA_PATH, data)
        preds = get_preds_new_data(learner, data)

        return preds['arr_proba_preds']

    def save(self, path, file_name):
        self.learner = make_fastai_serializable(self.learner)
        super().save(path,file_name)


