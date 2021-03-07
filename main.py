import argparse
from utils import *



if __name__ == '__main__':
	def str2bool(v):
		if v.lower() in ('yes', 'true', 't', 'y', '1'):
			return True
		elif v.lower() in ('no', 'false', 'f', 'n', '0'):
			return False
		else:
			raise argparse.ArgumentTypeError('Unsupported value encountered.')

	parser = argparse.ArgumentParser()

	parser.add_argument('--kd_method', type=str, default='de')
	parser.add_argument('--srs', type=str, default='nextitnet')
	# dataset

	parser.add_argument('--block_num', type=int, default=4)
	parser.add_argument('--embed_size', type=int, default=128)
	parser.add_argument('--hidden_size', type=int, default=128)

	parser.add_argument('--dataset', type=str, default='weishi')
	
	parser.add_argument('--device', type=str, default='cpu')

	parser.add_argument('--update_weight', default="true", type=str2bool)
	parser.add_argument('--use_attn', default="true", type=str2bool)

	args = parser.parse_args()

	if args.kd_method == "scratch":
		from Config import SRS_Config as Config
		if args.srs == 'nextitnet':
			from models.NextItNet import NextItNet as Model
		elif args.srs == 'sasrec':
			from models.SASRec import SASRec as Model
	else:
		from models.SRS_KD import SRS_KD as Model
		from Config import KD_Config as Config

	config = Config(args)
	train_dataloader, eval_dataloader, item_num, seq_len = get_dataloader(config)
	config.setTwoVal(item_num, seq_len)
	config.item_num = item_num
	config.seq_len = seq_len
	model = Model(config).to(config.device)

	train(model, config, train_dataloader, eval_dataloader)