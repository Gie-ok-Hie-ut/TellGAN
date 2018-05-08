import os
import numpy as np
import torch
import torch.nn as nn




try:
	dictionary = np.load('grid_embedding.npy').item()
	print "[Dictionary] Loading Existing Embedding Dictionary"

except IOError as e:
	dictionary = {'new':-1}
	print "[Dictionary] Building New Word Embedding Dictionary"

#word = ['hello','world','everyone','good','morning','new','world2','good','bad','good','ki']
word = ['ello','wod','evne','od','morn','new','wod2','gd','bad','good','ki']
word = ['el','wo','vne','o','mon','nw','wod2','gd','bad','go','ki']



dictionary_size=100
for i in range(0,11):

	if dictionary.get(word[i],-1) == -1:
		dictionary.update({word[i]:float((len(dictionary)+1))/dictionary_size})
		print word[i]
		print dictionary[word[i]]

		#FT=torch.FloatTensor([dictionary[word[i]]]).new_full()
		expand = np.full((10,10),dictionary[word[i]])
		tensor_expand = torch.from_numpy(expand)

		print tensor_expand




np.save('grid_embedding.npy',dictionary)
print dictionary


