import time


class BaseConfig():
	def __init__(self, args):
		self.lr = 0.001
		self.reg = 0.001
		self.batch_size = 256
		self.max_epoch = 1000
		self.early_stop = 20

		self.kd_method = args.kd_method
		self.srs = args.srs
		self.dataset = args.dataset
		self.device = args.device
        
		if self.dataset == 'weishi':
			self.dataset_path = 'Dataset/weishi.csv'
		else:
			self.dataset_path = None
		self.eval_percentage = 0.2
		
		self.log_path = "log/{}_{}_{}_{}.log".format(self.kd_method, self.srs, self.dataset, time.strftime("%Y%m%d%H%M%S", time.localtime()))

class KD_Config(BaseConfig):
    def __init__(self, args):
        super(KD_Config, self).__init__(args)

        
        self.teacher_model = None

class SRS_Config(BaseConfig):
	def __init__(self, args):
		super(SRS_Config, self).__init__(args)
		self.item_num = -1
		self.embed_size = 256
		self.hidden_size = 256
		if self.srs.lower() == 'nextitnet':
			self.dilations = [1, 4] * 3
			self.kernel_size = 3
		elif self.srs.lower() == 'sasres':
			self.seq_len = 20
			self.num_head = 8
			self.dropout = 0.5