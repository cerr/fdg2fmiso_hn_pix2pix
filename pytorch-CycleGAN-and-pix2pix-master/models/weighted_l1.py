
import torch

'''WeightedL1Loss: input (fake, real), output weighted L1 loss, hypoxia threshold = 1.2 is for TBR 
weight is defeined by a segment function where the weight is 1 for any real value is smaller than threshold
value and 1 to 1.5 for any real value is bigger than threshold. This is to give more penalty for errors in the pixels 
that is bigger than threshold in real image'''
cuda = torch.device('cuda')

def WeightedL1Loss(fake, real, threshold = 1.2):
    diff = torch.abs(fake - real)
    thres = torch.tensor(2 * threshold/4.7117 - 1, device=cuda)
    tensor_size = real.size()
    weights = torch.maximum(torch.full(tensor_size, 1, device=cuda), torch.sigmoid(4.7117/2*(real - thres)) + 0.5)
    return torch.mean(weights * diff)    

# class WeightedL1Loss(torch.nn.Module):
#     def __init__(self):
#         # self.fake = fake
#         # self.real = real
#         # self.thres = torch.tensor(2 * threshold/4.7117 - 1)
#         # tensor_size = self.real.size()
#         # self.weights = torch.maximum(torch.full(tensor_size, 1), torch.sigmoid(self.real - self.thres) + 0.5)
#         super(WeightedL1Loss, self).__init__()
        
#     def forward(self, fake, real, threshold = 1.2):
#         diff = torch.abs(fake - real)
#         thres = torch.tensor(2 * threshold/4.7117 - 1)
#         tensor_size = real.size()
#         weights = torch.maximum(torch.full(tensor_size, 1), torch.sigmoid(self.real - thres) + 0.5)
#         return torch.mean(weights * diff)

# a = torch.tensor((-0.6, -.15))
# b = torch.tensor((-0.3, -.05))
# c = torch.tensor((-0.5, -.45))
# loss = WeightedL1Loss(b, a)
# loss1 =WeightedL1Loss(c, a)