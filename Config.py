import time
import os

from torch.utils.data import dataset


class BaseConfig():
	def __init__(self, args):
		self.lr = 0.001
		self.reg = 1e-7
		self.batch_size = 256
		self.max_epoch = 200
		self.early_stop = 20
		self.eval_begin_epochs = 20
		self.eval_per_epochs = 5

		self.kd_method = args.kd_method
		self.srs = args.srs
		self.dataset = args.dataset
		self.device = args.device
        
		if self.dataset == 'weishi':
			self.dataset_path = 'Dataset/weishi.csv'
		else:
			self.dataset_path = None
		self.eval_percentage = 0.1
		
		tmp = "outputs/{}/{}/".format(self.srs, self.dataset)
		if not os.path.exists(tmp):
			os.makedirs(tmp)
		tmp += "{}_{}".format(self.kd_method, time.strftime("%Y%m%d%H%M%S", time.localtime()))
		self.log_path = tmp + ".log"
		self.save_path = tmp + ".t7"
	
	def log(self, logger):
		logger.info("kd_method = {}, srs = {}, dataset = {}".format(self.kd_method, self.srs, self.dataset))
		logger.info("lr = {}, reg = {}, batch_size = {}".format(self.lr, self.reg, self.batch_size))

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
			self.dilations = [1, 4] * 12
			self.kernel_size = 3
		elif self.srs.lower() == 'sasres':
			self.seq_len = 20
			self.num_head = 8
			self.dropout = 0.5

	def log(self, logger):
		super().log(logger)
		logger.info("embed_size = {}, hidden_size = {}, item_num = {}".\
			format(self.embed_size, self.hidden_size, self.item_num))
		if self.srs.lower() == 'nextitnet':
			logger.info("dilations = {}, kernel_size = {}".format(self.dilations, self.kernel_size))
		elif self.srs.lower() == 'sasres':
			logger.info("seq_len = {}, num_head = {}, dropout = {}".format(self.seq_len, self.num_head, self.dropout))
			