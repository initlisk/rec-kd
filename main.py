import argparse
from utils import *



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--kd_method', type=str, default='scratch')
	parser.add_argument('--srs', type=str, default='nextitnet')
	# dataset
	parser.add_argument('--dataset', type=str, default='weishi')
	
	parser.add_argument('--device', type=str, default='cpu')

	args = parser.parse_args()
	print(args)
	print ('\n')

	if args.kd_method == "scratch":
		from Config import SRS_Config as Config
		if args.srs == 'nextitnet':
			from models.NextItNet import NextItNet as Model
		elif args.srs == 'sasrec':
			from models.SASRec import SASRec as Model
	elif args.kd_method == 'bertemd':
		pass

	config = Config(args)
	train_dataloader, eval_dataloader, item_num = get_dataloader(config)
	config.item_num = item_num
	model = Model(config).to(config.device)

	train(model, config, train_dataloader, eval_dataloader)