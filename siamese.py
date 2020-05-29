import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNet(nn.Module):
    def __init__(self, image_size):
        super(SiameseNet, self).__init__()
        self.h, self.w = image_size[0]//16, image_size[1]//16
        
        self.C1 = nn.Conv2d(1, 64, 4, 2, 1)
        self.C2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.C3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.C4 = nn.Conv2d(256, 256, 4, 2, 1)
        self.C5 = nn.Conv2d(256, 256, 3, 1, 1)
        self.D1 = nn.Linear(self.h*self.w*256, 256)
        self.D2 = nn.Linear(256, 1)
        
        self.B1 = nn.BatchNorm2d(64)
        self.B2 = nn.BatchNorm2d(128)
        self.B3 = nn.BatchNorm2d(256)
        self.B4 = nn.BatchNorm2d(256)
        
        modules = list(self.modules())
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.orthogonal_(m.weight)
        
        self.Convs = nn.ModuleList([self.C1, self.C2, self.C3, self.C4, self.C5])
        self.BNs = nn.ModuleList([self.B1, self.B2, self.B3, self.B4])
        self.Pool = nn.MaxPool2d(3, stride = 2)
        
    def encode(self, x):
        for C, B in zip(self.Convs, self.BNs):
            x = F.leaky_relu(B(C(x)))
            #x = self.Pool(x)
        x = x.view(x.size(0), -1)
        return x
        
    def forward(self, x, y):
        x = self.encode(x)
        y = self.encode(y)
        d = torch.abs(x - y)
        d = F.leaky_relu(self.D1(d))
        d = self.D2(d)
        return d
    
def train_step(netS, optimizer, loss_fn, x, y, sim):
    netS.train()
    netS.zero_grad()
    out = netS(x, y)
    L = loss_fn(out, sim)
    L.backward()
    optimizer.step()
    return L.detach().item()

def test_step(netS, loss, x, y, sim):
    netS.eval()
    out = netS(x, y)
    L = loss(out, sim)
    return L.detach().item()

def oneshot_test(netS, support_set, query_set, class_labels):
    correct_pred = 0

    with torch.no_grad():
        for i in range(len(query_set)):
            q = query_set[i].unsqueeze(0)

            out = netS(support_set, q.repeat(support_set.size(0), 1, 1, 1))
            out = torch.sigmoid(out)
            index = out.argmax()

            if index.item() == class_labels[i]:
                correct_pred += 1

            if i%50 == 0:
                print(i, "/",len(query_set))
    
    print("One shot accuracy: ", correct_pred/len(query_set), "\n")