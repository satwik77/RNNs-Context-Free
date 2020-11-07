import os
import sys
import numpy as np
from src.args import build_parser
import pandas as pd
import ipdb as pdb

def main():
	'''read arguments'''
	parser = build_parser()
	args = parser.parse_args()
	config = args
	results_df = pd.read_json(os.path.join('out', 'val_results_{}.json'.format(config.dataset)))
	results_df = results_df.transpose()
	for run_name in results_df.run_name.values:
		cmd = 'python -m src.main -mode test -dataset {} -run_name {} -gpu {} -test_prefix {}'.format(config.dataset, run_name, config.gpu, config.test_prefix)
		print("Command: {}".format(cmd))
		os.system(cmd)


if __name__ == "__main__":
	main()