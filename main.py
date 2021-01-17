import argparse
import os

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--kd_method', type=str, default='null')
	parser.add_argument('--srs', type=str, default='nextitnet')
	# dataset
	parser.add_argument('--dataset', type=str, default='weishi')
	
	parser.add_argument('--device', type=str, default='cuda:0')

	opt = parser.parse_args()
	print(opt)
	print ('\n')

	if opt.kd_method == "null":
		config = SRS_Config()
	elif opt.kd_method == "bertemd":
		config = KD_Config()
	
	model = get_model(config)

	train_dataloader, eval_dataloader = get_loaders()

	train()