import argparse
from utils import *



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--kd_method', type=str, default='vanilla_kd')
	parser.add_argument('--srs', type=str, default='nextitnet')
	# dataset

	parser.add_argument('--block_num', type=int, default=4)
	parser.add_argument('--embed_size', type=int, default=128)
	parser.add_argument('--hidden_size', type=int, default=128)

	parser.add_argument('--dataset', type=str, default='weishi')
	
	parser.add_argument('--device', type=str, default='cpu')

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