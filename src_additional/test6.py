import torch


w = 3
h = 3
c_a = 4
c_b = 1


a1 = torch.ones(c_a,w,h)
a2 = torch.ones(c_a,w,h)

b1 = torch.zeros(c_b,w,h)
b2 = torch.zeros(c_b,w,h)


a_s = torch.stack((a1,a2),0)
b_s = torch.stack((b1,b2),0)

print a_s
print b_s


c_s= torch.cat((a_s,b_s),1)
print c_s
