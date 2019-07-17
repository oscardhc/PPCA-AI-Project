import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import model as m


epoch = 100
batch_size = 16



if __name__ == "__main__":
    CEL_criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    E = m.Encoder()
    D = m.Decoder()
    dis = m.Discriminator()
    I = m.Interp()

    E_optim = optim.Adam(E.parameters(), lr=0.0002)
    D_optim = optim.Adam(D.parameters(), lr=0.0002)
    dis_optim = optim.Adam(dis.parameters(), lr=0.0002)
    I_optim = optim.Adam(I.parameters(), lr=0.0002)

    MSE_criterion = nn.MSELoss()
    BCE_criterion = nn.BCELoss()

    # full_strenth is a all-1-vector
    dataset = 0

    for i in range(epoch):
        for t, images in enumerate(dataset):
            pass