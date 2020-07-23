from ..data import io


class BasePipeline:

    @classmethod
    def load(cls, path):
        '''
        Loads object
        :param path:
        :return:
        '''
        obj = io.load_object(path)
        return obj

    def __init__(self, model_id, external_dependencies):
        '''
        Instantiates object with its id and its external dependencies like in dependencies.txt
        :param model_id:
        '''
        self.model_id = model_id
        self.external_dependencies = external_dependencies
        return

    def save(self, path, file_name):
        '''
        Saves model serialized object
        :param path:
        :param file_name:
        :return:
        '''
        io.save_object(self, path, file_name)
        return

    def __str__(self):
        '''
        Shows str representation object (str(self))
        :return:
        '''
        return self.model_id

    def build_features(self, data):
        '''
        Feature engineering step
        :param data:
        :return:
        '''
        return data

    def load_and_preprocess_fit(self):
        '''
        Load and preprocess for fit process
        :return:
        '''
        raise NotImplementedError

    def load_and_preprocess_predict(self):
        '''
        Load and preprocess for predict process
        :return:
        '''
        raise NotImplementedError

    def load_and_preprocess_validate(self):
        '''
        Load and preprocess for validation process
        :return:
        '''
        raise NotImplementedError

    def fit(self):
        '''
        Model train
        :return:
        '''
        raise NotImplementedError

    def predict(self):
        '''
        Trained model prediction
        :return:
        '''
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def keep_trainning(self):
        raise NotImplementedError

    def interpret(self):
        '''
        Model interpretation
        :return:
        '''
        raise NotImplementedError

