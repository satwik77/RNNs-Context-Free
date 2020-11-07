import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.components.utils import MultiHeadedAttention, SublayerConnection
import pdb

class SALayer(nn.Module):

	def __init__(self, hidden_size, heads=2, dropout=0.3):
		super(SALayer, self).__init__()
		self.hidden_size = hidden_size
		self.heads= heads
		self.dropout = dropout
		self.Mah = MultiHeadedAttention(heads, hidden_size, dropout)
		self.sublayer = SublayerConnection(hidden_size, dropout)

	def forward(self, q, Key, Value):
		Key_l = Key[0]
		Value_l = Value[0]
		h_att = self.sublayer(q, lambda x: self.Mah(x, Key_l, Value_l))
		# h_att = self.Mah(q, Key_l, Value_l)
		
		return h_att





class SARNNModel(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, heads=2):
		super(SARNNModel, self).__init__()
		self.drop = nn.Dropout(dropout)
		self.encoder = nn.Embedding(ntoken, ninp)
		'''
		if rnn_type in ['LSTM', 'GRU']:
			self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
		else:
			try:
				nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
			except KeyError:
				raise ValueError( """An invalid option for `--model` was supplied,
								 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
			self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
		'''
		self.rnn = getattr(nn, 'LSTM')(ninp, nhid, nlayers, dropout=dropout)
		self.decoder = nn.Linear(nhid, ntoken)
		self.self_attn = SALayer(nhid, heads)
		self.sigmoid = nn.Sigmoid()
		# self.gh = nn.Linear(nhid, nhid)
		# self.gx = nn.Linear(ninp, nhid)

		# Optionally tie weights as in:
		# "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
		# https://arxiv.org/abs/1608.05859
		# and
		# "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
		# https://arxiv.org/abs/1611.01462
		if tie_weights:
			if nhid != ninp:
				raise ValueError('When using the tied flag, nhid must be equal to emsize')
			self.decoder.weight = self.encoder.weight

		self.init_weights()

		self.rnn_type = rnn_type
		self.nhid = nhid
		self.nlayers = nlayers

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, input, hidden):
		inp_len = input.size(0)
		decoded_list = []
		for t in range(inp_len):
			input_t = input[t].unsqueeze(0)
			hidden_vector, Memory = hidden
			H_t, C_t = Memory
			h_t, c_t = hidden_vector

			# q = H_t[:, -1]

			emb_t = self.drop(self.encoder(input_t))
			# gate_att = self.sigmoid(self.gh(h_t) + self.gx(emb_t))
			# h_att = self.self_attn(q.clone(), H_t.clone()) + h_t
			h_att = self.self_attn(emb_t, Key = H_t.clone(), Value= H_t.clone()) + h_t

			hidden_vector = (h_att, c_t)
			output_t, hidden_vector = self.rnn(emb_t, hidden_vector)
			output_t = self.drop(output_t)
			decoded = self.decoder(output_t)
			decoded_list.append(decoded)

			for i in range(len(H_t)):
				h_state= hidden_vector[0][i].unsqueeze(0).clone().detach()
				c_state = hidden_vector[1][i].unsqueeze(0).clone().detach()
				H_t[i]= torch.cat([H_t[i][1:], h_state])
				C_t[i] = torch.cat([C_t[i][1:], c_state])

		
			hidden = (hidden_vector, (H_t, C_t))
		decoded = torch.cat(decoded_list, dim = 0)
		return decoded, hidden

	def init_hidden(self, bsz, bptt=35):
		weight = next(self.parameters())
		if self.rnn_type == 'LSTM':
			return ((weight.new_zeros(self.nlayers,  bsz, self.nhid),
					weight.new_zeros(self.nlayers,  bsz, self.nhid)),
					(weight.new_zeros(self.nlayers, bptt, bsz, self.nhid),
					weight.new_zeros(self.nlayers, bptt, bsz, self.nhid)))
		else:
			return weight.new_zeros(self.nlayers, bptt, bsz, self.nhid)