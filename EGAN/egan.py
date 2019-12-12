import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transforms
import torch.utils.data as data
from PIL import Image
from net import *
import time
import matplotlib.pyplot as plt

def make_datapath_list():
    """学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する。 """

    train_img_list = list()  # 画像ファイルパスを格納

    for img_idx in range(200):
        img_path = "./data/img_78_28size/img_7_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

        img_path = "./data/img_78_28size/img_8_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

    return train_img_list

def make_test_datapath_list():
    test_img_list = []
    for img_idx in range(5):
        img_path = "./data/test_28size/img_7_" + str(img_idx) + ".jpg"
        test_img_list.append(img_path)
        
        img_path = "./data/test_28size/img_8_" + str(img_idx) + ".jpg"
        test_img_list.append(img_path)

        img_path = "./data/test_28size/img_2_" + str(img_idx) + ".jpg"
        test_img_list.append(img_path)

    return test_img_list

class ImageTransform():
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)


class GAN_Img_Dataset(data.Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img)

        return img_transformed

def Anomaly_score(x, fake_image, z_out_real, D, Lambda=0.1):
    residual_loss = torch.abs(x - fake_image)
    residual_loss = residual_loss.view(residual_loss.size()[0], -1)
    residual_loss = torch.sum(residual_loss, dim=1)

    _, x_feature = D(x, z_out_real)
    _, G_feature = D(fake_image, z_out_real)

    discremination_loss = torch.abs(x_feature - G_feature)
    discremination_loss = discremination_loss.view(discremination_loss.size()[0], -1)
    discremination_loss = torch.sum(discremination_loss, dim=1)

    loss_each = (1-Lambda) * residual_loss + Lambda * discremination_loss
    # ミニバッチ全体の損失
    toral_loss = torch.sum(loss_each)

    return toral_loss, loss_each, residual_loss

class Generator(nn.Module):
    def __init__(self, z_dim=20):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1024, 7*7*128),
            nn.BatchNorm1d(7*7*128),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64,
            kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.last = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = out.view(z.shape[0], 128, 7, 7)
        out = self.layer3(out)
        out = self.last(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, z_dim=20):
        super(Discriminator, self).__init__()
        self.x_layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.x_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.z_layer1 = nn.Linear(z_dim, 512)

        self.last1 = nn.Sequential(
            nn.Linear(3648, 1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.last2 = nn.Linear(1024, 1)

    def forward(self, x, z):
        x_out = self.x_layer1(x)
        x_out = self.x_layer2(x_out)

        z = z.view(z.shape[0], -1)
        z_out = self.z_layer1(z)

        x_out = x_out.view(-1, 64 * 7 * 7)
        out = torch.cat([x_out, z_out], dim=1)
        out = self.last1(out)

        feature = out
        feature = feature.view(feature.size()[0], -1)

        out = self.last2(out)

        return out, feature

class Encorder(nn.Module):
    def __init__(self, z_dim=20):
        super(Encorder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.last = nn.Linear(128 * 7 * 7, z_dim)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(-1, 128 * 7 * 7)
        out = self.last(out)

        return out

def train_model(G, D, E, dataloader, num_epochs):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    lr_ge = 0.0001
    lr_d = 0.0001/4
    beta1, beta2 = 0.5, 0.999
    g_optimizer = torch.optim.Adam(G.parameters(), lr_ge, [beta1, beta2])
    e_optimizer = torch.optim.Adam(E.parameters(), lr_ge, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), lr_d, [beta1, beta2])

    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    z_dim=20
    mini_batch_size = 64

    G.to(device)
    E.to(device)
    D.to(device)

    G.train()
    E.train()
    D.train()

    torch.backends.cuda.benchmark = True

    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    iteration = 1
    logs = []

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        epoch_g_loss = 0.0
        epoch_e_loss = 0.0
        epoch_d_loss = 0.0
        print("-"*30)
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-"*30)
        print("(train)")

        for images in dataloader:
            if images.size()[0] == 1:
                continue

            mini_batch_size = images.size()[0]
            label_real = torch.full((mini_batch_size,), 1).to(device)
            label_fake = torch.full((mini_batch_size,), 0).to(device)

            images =images.to(device)

            # Discriminatorの学習
            z_out_real = E(images)
            d_out_real, _ = D(images, z_out_real)

            input_z = torch.rand(mini_batch_size, z_dim).to(device)
            fake_images = G(input_z)
            d_out_fake, _ = D(fake_images, input_z)

            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Generatorの学習
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            fake_images = G(input_z)
            d_out_fake, _ = D(fake_images, input_z)

            g_loss = criterion(d_out_fake.view(-1), label_real)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Encoderの学習
            z_out_real = E(images)
            d_out_real, _ = D(images, z_out_real)

            e_loss = criterion(d_out_real.view(-1), label_fake)

            e_optimizer.zero_grad()
            e_loss.backward()
            e_optimizer.step()

            # 記録
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_e_loss += e_loss.item()
            iteration += 1

        t_epoch_finish = time.time()
        print("-"*30)
        print("Epoch {} || Epoch_D_Loss:{:.4f} || Epoch_G_Loss:{:.4f} || Epoch_E_Loss:{:.4f}".format(
            epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size, epoch_e_loss/batch_size
        ))
        print("timer: {:.4f} sec.".format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()
    
    print("total iteration :", iteration)
    return G, D, E

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        m.bias.data.fill_(0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    G = Generator(z_dim=20)
    G.train()

    input_z = torch.randn(2, 20)

    fake_images = G(input_z)
    img_transformed = fake_images[0][0].detach().numpy()
    plt.imshow(img_transformed)
    plt.show()
    plt.savefig("egan_gen_test")

    D = Discriminator(z_dim=20)

    input_z = torch.randn(2, 20)
    fake_images = G(input_z)

    d＿out, _ = D(fake_images, input_z)
    print(nn.Sigmoid()(d_out))

    E = Encorder(z_dim=20)
    x = fake_images
    z = E(x)

    print(z.shape)
    print(z)

    # 学習 
    train_img_list = make_datapath_list()
    mean = (0.5,)
    std = (0.5,)

    train_dataset = GAN_Img_Dataset(
        file_list=train_img_list, transform=ImageTransform(mean, std)
    )

    batch_size = 64
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    G.apply(weights_init)
    D.apply(weights_init)
    E.apply(weights_init)
    print("Network was initialized!")

    num_epochs = 1500
    G_update, D_update, E_update = train_model(
        G, D, E, dataloader=train_dataloader, num_epochs=num_epochs
    )


    # anomaly detection
    test_img_list = make_test_datapath_list()

    test_dataset = GAN_Img_Dataset(
        file_list=test_img_list, transform=ImageTransform(mean, std))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 8
    z_dim = 20
    fixed_z = torch.randn(batch_size, z_dim)
    fake_images = G_update(fixed_z.to(device))

    batch_iterator = iter(train_dataloader)
    images = next(batch_iterator)

    fig = plt.figure(figsize=(15, 6))
    for i in range(0, 5):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i][0].cpu().detach().numpy(), "gray")
        plt.subplot(2, 5, 5+i+1)
        plt.imshow(fake_images[i][0].cpu().detach().numpy(), "gray")
        plt.savefig("egan_real_vs_fake")

    x  = images[0:5]
    x = x.to(device)

    # images -> images[0:5]に修正
    z_out_real = E_update(images[0:5].to(device))
    images_reconstruct = G_update(z_out_real)

    loss, loss_each, residual_loss_each, = Anomaly_score(
        x, images_reconstruct, z_out_real, D_update, Lambda=0.1
    )
    loss_each = loss_each.cpu().detach().numpy()
    print("total loss:", np.round(loss_each, 0))

    fig = plt.figure(figsize=(15, 6))
    for i in range(0, 5):
        # 上段にテストデータを
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i][0].cpu().detach().numpy(), 'gray')

        # 下段に生成データを表示する
        plt.subplot(2, 5, 5+i+1)
        plt.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')

        plt.savefig("./egan_result.png")