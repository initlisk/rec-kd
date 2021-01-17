from BaseConfig import BaseConfig

class KD_Config(BaseConfig):
    def __init__(self, kd_method, model_type, dataset):
        super(KD_Config, self).__init__(kd_method, model_type, dataset)

        
        self.teacher_model = No