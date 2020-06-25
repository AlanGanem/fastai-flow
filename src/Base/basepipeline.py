from ..data import io


class BasePipeline:

    @classmethod
    def load(cls, path):
        obj = io.load_object(path)
        return obj

    def __init__(self, model_id):
        self.model_id = model_id
        return

    def save(self, path, file_name):
        io.save_object(self, path, file_name)
        return

    def __str__(self):
        return self.model_id

    def load_and_preprocess_fit(self):
        raise NotImplementedError

    def load_and_preprocess_predict(self):
        raise NotImplementedError

    def load_and_preprocess_validate(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def keep_trainning(self):
        raise NotImplementedError

