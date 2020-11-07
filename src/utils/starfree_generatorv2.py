import numpy as np
import torch
import ipdb as pdb
import textwrap

def get_sigma_star(sigma, length):
	choices = sigma
	string = ''
	while len(string) < length:
		symbol = np.random.choice(choices)
		string += symbol
	return string

class Concatv1Language(object):
    '''
    Regular Expression: (a|b)*d(b|c)*
    '''
    def __init__(self):
		self.sigma = ['a', 'b', 'c', 'd']
		self.n_outputs = len(self.sigma)

    def generate_string(self, maxlength):
        len1 = np.random.randint(0, maxlength)
        part1 = get_sigma_star(['a','b'], len1)

        len2 = np.random.randint(0, maxlength - len1)
        part2 = get_sigma_star(['b','c'], len2)

        string = part1 + 'd' + part2

        return string
    
    def output_generator(self, seq):
        found_d = False
        out_str = ''
        for i,ch in enumerate(seq):
            if not found_d:
                if ch == 'd':
                    found_d = True
                    out_str += '0110'
                else:
                    out_str += '1101'
            else:
                out_str += '0110'
        return out_str

	def generate_list(self, num, min_length, max_length):
		arr = []
		while len(arr) < num:
			string = self.generate_string(max_length)
			if string in arr:
				continue
			if len(string) >= min_length and len(string) <= max_length:
				arr.append(string)
				print("Generated {}/{} samples".format(len(arr), num), end = '\r', flush = True)
		print()
		return arr

	def training_set_generator(self, num, min_size, max_size):
		input_arr = self.generate_list (num, min_size, max_size)
		output_arr = []
		for seq in input_arr:
			output_arr.append (self.output_generator (seq))
		return input_arr, output_arr

	def lineToTensorOutput(self, line):
		assert len(line) % 4 == 0
		parts = textwrap.wrap(line, 4)
		tensor = []
		for i,part in enumerate(parts):
			tensor.append(list(map(float, list(part))))

		tensor = torch.tensor(tensor)
		return tensor

class Concatv2Language(object):
    '''
    Regular Expression: (a|b)*d(b|c)*d(c|a)*
    '''
    def __init__(self):
		self.sigma = ['a', 'b', 'c', 'd']
		self.n_outputs = len(self.sigma)

    def generate_string(self, maxlength):
        len1 = np.random.randint(0, maxlength)
        part1 = get_sigma_star(['a','b'], len1)

        len2 = np.random.randint(0, maxlength - len1)
        part2 = get_sigma_star(['b','c'], len2)

        len3 = np.random.randint(0, maxlength - len1 - len2)
        part3 = get_sigma_star(['c','a'], len3)

        string = part1 + 'd' + part2 + 'd' + part3
        return string

        return string
    
    def output_generator(self, seq):
        d_count = 0
        out_str = ''
        for i,ch in enumerate(seq):
            if d_count == 0:
                if ch == 'd':
                    d_count += 1
                    out_str += '0110'
                else:
                    out_str += '1101'
            elif d_count == 1:
                if ch == 'd':
                    d_count += 1
                    out_str += '1010'
                else:
                    out_str += '0111'
            else:
                out_str += '1010'
        return out_str

	def generate_list(self, num, min_length, max_length):
		arr = []
		while len(arr) < num:
			string = self.generate_string(max_length)
			if string in arr:
				continue
			if len(string) >= min_length and len(string) <= max_length:
				arr.append(string)
				print("Generated {}/{} samples".format(len(arr), num), end = '\r', flush = True)
		print()
		return arr

	def training_set_generator(self, num, min_size, max_size):
		input_arr = self.generate_list (num, min_size, max_size)
		output_arr = []
		for seq in input_arr:
			output_arr.append (self.output_generator (seq))
		return input_arr, output_arr

	def lineToTensorOutput(self, line):
		assert len(line) % 4 == 0
		parts = textwrap.wrap(line, 4)
		tensor = []
		for i,part in enumerate(parts):
			tensor.append(list(map(float, list(part))))

		tensor = torch.tensor(tensor)
		return tensor



