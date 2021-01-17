from BaseConfig import BaseConfig

class SRS_Config(BaseConfig):
    def __init__(self, kd_method, model_type, dataset):
        super(SRS_Config, self).__init__(kd_method, model_type, dataset)
        self.model_type = model_type
        self.item_num = 100000
        self.embed_size = 256
        self.hidden_size = 256
        if self.model_type.lower() == 'nextitnet':
            self.dilations = [1, 4] * 8
            self.kernel_size = 3
        elif self.model_type.lower() == 'sasres':
            self.seq_len = 20
            self.num_head = 8
            self.dropout = 0.5
        

