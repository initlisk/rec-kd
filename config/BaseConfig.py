class BaseConfig():
	def __init__(self, kd_method, model_type, dataset):
		self.lr = 0.001
		self.reg = 0.001
		self.batch_size = 1024
		self.device = "cpu"
		self.max_epoch = 1000
		self.early_stop = 20

		self.kd_method = kd_method
		self.model_type = model_type
		self.dataset = dataset