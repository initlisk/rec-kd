import time
import os

from torch import set_flush_denormal


class BaseConfig():
	def __init__(self, args):
		if args.srs.lower() == "nextitnet":
			self.lr = 1e-3
		else:
			self.lr = 1e-4

		self.reg = 0
		self.batch_size = 128
		self.max_epoch = 100
		self.early_stop = 10
		self.eval_begin_epochs = 0
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
	
	def setTwoVal(self, item_num, seq_len):
		pass

class KD_Config(BaseConfig):
	def __init__(self, args):
		super(KD_Config, self).__init__(args)
		self.student_config = SRS_Config(args)
		args.hidden_size *= 2
		args.embed_size *= 2
		args.block_num *= 2
		self.teacher_config = SRS_Config(args)
		self.teacher_path = "outputs/{}/{}/scratch_teacher.t7".format(self.srs, self.dataset)
		if self.kd_method == "bertemd":
			self.use_attn = args.use_attn
			self.update_weight = args.update_weight

	def log(self, logger):
		self.student_config.log(logger)
		logger.info("teacher_path = {}".format(self.teacher_path))
	
	def setTwoVal(self, item_num, seq_len):
		self.teacher_config.setTwoVal(item_num, seq_len)
		self.student_config.setTwoVal(item_num, seq_len)

class SRS_Config(BaseConfig):
	def __init__(self, args):
		super(SRS_Config, self).__init__(args)
		self.item_num = -1
		self.embed_size = args.embed_size
		self.hidden_size = args.hidden_size
		self.seq_len = -1
		self.block_num = args.block_num
		if self.srs.lower() == 'nextitnet':
			self.dilations = [1, 4] * self.block_num
			self.kernel_size = 3
		elif self.srs.lower() == 'sasrec':
			self.num_head = 2
			self.dropout = 0

	def log(self, logger):
		super().log(logger)
		logger.info("bolck_num = {}, embed_size = {}, hidden_size = {}, item_num = {}, seq_len = {}".\
			format(self.block_num, self.embed_size, self.hidden_size, self.item_num, self.seq_len))
		if self.srs.lower() == 'nextitnet':
			logger.info("kernel_size = {}".format(self.kernel_size))
		elif self.srs.lower() == 'sasrec':
			logger.info("num_head = {}, dropout = {}".format(self.num_head, self.dropout))
	
	def setTwoVal(self, item_num, seq_len):
		self.item_num = item_num
		self.seq_len = seq_len