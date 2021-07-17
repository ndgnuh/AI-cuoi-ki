from argparse import ArgumentParser
from pprint import pprint

parser = ArgumentParser(prog="python3 main.py")
parser.add_argument("--model", "-m", dest="model", help="select model", type=str)
parser.add_argument("--model-path", dest="model_path", help="model checkpoint path", type=str)
parser.add_argument("--model-args", dest="model_args", help="model arguments, in python dict or list syntax (eg: '[300, 1000]')", type=str, default="[]")
parser.add_argument('--device', '-d', dest='device', help="torch device", default=None)
parser.add_argument("--list-model", dest="list_model", action="store_true")
parser.add_argument("--list-preset", dest="list_preset", action="store_true")
parser.add_argument("--old-main", dest="run_old_main", action="store_true", help="run old main function")
parser.add_argument("--start-epoch", dest="start_epoch", default=1, help="epoch start offset", type=int)
parser.add_argument("--end-epoch", "--epoch", dest="end_epoch", default=100000, help="max epochs", type=int)
parser.add_argument("--batch-size", "--bs", dest="batch_size", default=100, help="batch size", type=int)

# LEARNING RATE
parser.add_argument("--lr", dest="lr", default=1e-3, help="set learning rate", type=float)
parser.add_argument('--decay-every', dest='decay_every', defaul=30, help="set N, decay learning rate every N epoch", type=int)
parser.add_argument('--decay-rate', dest='decay_rate', defaul=0.1, help="learning rate decay coeff", type=float)

# PRESETS
parser.add_argument("--preset", dest="preset", help="Preset", type=str)
presets = {
		'MPL1': [
			'--model', 'MPL1',
			'--model-path', 'checkpoint/MPL1.pth',
			'--model-args', '[3072, 100]',
			'--batch-size', '100'
			],
		'MPL2': [
			'--model', 'MPL2',
			'--model-path', 'checkpoint/MPL2.pth',
			'--model-args', '[[3072, 2048, 1024, 100], ["LeakyReLU", "Sigmoid"]]',
			]
		}

def patch():
	import models
	import torch
	from os.path import isfile

	def init_model(key: str, arg):
		for name in models.all_models.keys():
			if key.lower() == name.lower():
				model = models.all_models[key]
				if type(arg) == type([]):
					return model(*arg)
				elif type(arg) == type({}):
					return model(**arg)
		raise Exception(f"Model {name} not found")

	def load_model(key: str):
		if isfile(key):
			return torch.load(key)
		return None

	pa = parser.parse_args
	def parse_args(*args, **kwargs):
		config = pa(*args, **kwargs)

		# Model function
		if config.list_model:
			pprint(models.all_models)
			exit(1)
			return config
		if config.list_preset:
			pprint(presets)
			exit(1)
		if config.preset in presets:
			import sys
			args = list(args) + sys.argv[1:]
			preset = presets[config.preset]
			for i, arg in enumerate(args):
				if arg == '--preset':
					args.pop(i)
					args.pop(i)
				if arg in preset:
					if arg in ['--batch-size', '--model-path']:
						j = preset.index(arg)
						preset.pop(j)
						preset.pop(j)

			args = args + preset
			print(args)
			return parse_args(args)
		if config.model is None and config.model_path is None:
			parser.print_help()
			print("Please provide a model or model_path")
			exit(-1)

		# Load model
		config.model_args = eval(config.model_args)
		model = None
		if config.model_path is not None:
			model = load_model(config.model_path)

		if model is None:
			model = init_model(config.model, config.model_args)
		config.model = model

		# Transfer to device
		if config.device is None: 
			if torch.cuda.is_available():
				config.device = torch.device('cuda')
			else:
				config.device = torch.device('cpu')
			pass
		else:
			config.device = torch.device(config.device)
		pass
		config.model = config.model.to(config.device)

		return config
	pass
	
	# Patch
	parser.parse_args = parse_args

patch()

if __name__ == "__main__":
	print(parser.parse_args())
