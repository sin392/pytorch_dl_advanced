import torch
import torch.nn.functional as F
import torch.nn as nn
# import torchvision.transforms as transforms
import numpy as np
import transforms
import torch.utils.data as data
from PIL import Image
from net import *
import time
import matplotlib.pyplot as plt

def Anomaly_score(x, fake_image, D, Lambda=0.1):
    residual_loss = torch.abs(x - fake_image)
    residual_loss = residual_loss.view(residual_loss.size()[0], -1)
    residual_loss = torch.sum(residual_loss, dim=1)

    _, x_feature = D(x)
    _, G_feature = D(fake_image)

    discremination_loss = torch.abs(x_feature - G_feature)
    discremination_loss = discremination_loss.view(discremination_loss.size()[0], -1)
    discremination_loss = torch.sum(discremination_loss, dim=1)

    loss_each = (1-Lambda) * residual_loss + Lambda * discremination_loss
    # ミニバッチ全体の損失
    toral_loss = torch.sum(loss_each)

    return toral_loss, loss_each, residual_loss

def make_datapath_list():
    """学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する。 """

    train_img_list = list()  # 画像ファイルパスを格納

    for img_idx in range(200):
        img_path = "./data/img_78/img_7_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

        img_path = "./data/img_78/img_8_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

    return train_img_list

def make_test_datapath_list():
    test_img_list = []
    for img_idx in range(5):
        img_path = "./data/test/img_7_" + str(img_idx) + ".jpg"
        test_img_list.append(img_path)
        
        img_path = "./data/test/img_8_" + str(img_idx) + ".jpg"
        test_img_list.append(img_path)

        img_path = "./data/test/img_2_" + str(img_idx) + ".jpg"
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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




def train_model(G, D, dataloader, num_epochs, device):
    print("device is", device)

    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])

    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    z_dim = 20
    mini_batch_size = 64

    G.to(device)
    D.to(device)

    G.train()
    D.train()

    torch.backends.cudnn.benchmark = True

    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    iteration = 1
    logs = []

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        print("-----------------------------")
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-----------------------------")
        print(" (train) ")

        for images in dataloader:
            if images.size() == 1:
                continue

            images = images.to(device)

            mini_batch_size = images.size()[0]
            label_real = torch.full((mini_batch_size,), 1).to(device)
            label_fake = torch.full((mini_batch_size,), 0).to(device)

            d_out_real, _ = D(images)

            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_image = G(input_z)
            d_out_fake, _ = D(fake_image)

            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()

            # Genaratorの学習
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_image = G(input_z)
            d_out_fake, _ = D(fake_image)

            g_loss = criterion(d_out_fake.view(-1), label_real)

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1

        t_epoch_finish = time.time()
        print("-----------------------------")
        print("epoch {} || Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f}".format(
            epoch, epoch_d_loss / batch_size, epoch_g_loss / batch_size
        ))
        print("timer: {:.4f}, sec.".format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

    return G, D


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_img_list = make_datapath_list()
    mean = (0.5,)
    std = (0.5,)

    train_dataset = GAN_Img_Dataset(
    file_list=train_img_list, transform=ImageTransform(mean, std))

    # DataLoaderを作成
    batch_size = 64

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    G = Generator(z_dim=20, image_size=64)
    D = Discriminator(z_dim=20, image_size=64)

    G.apply(weights_init)
    D.apply(weights_init)

    print("ネットワークの初期化完了")

    num_epochs = 300
    G_update, D_update = train_model(
        G, D, dataloader=train_dataloader, num_epochs=num_epochs, device=device)


########################################

    test_img_list = make_test_datapath_list()

    test_dataset = GAN_Img_Dataset(
        file_list=test_img_list, transform=ImageTransform(mean, std))

    batch_size = 5

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # テストデータの確認
    batch_iterator = iter(test_dataloader)  # イテレータに変換
    images = next(batch_iterator)  

    # 1番目のミニバッチを取り出す
    fig = plt.figure(figsize=(15, 6))
    for i in range(0, 5):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i][0].cpu().detach().numpy(), 'gray')

    x  = images[0:5]
    x = x.to(device)

    z = torch.randn(5, 20).to(device)
    z = z.view(z.size(0), z.size(1), 1, 1)

    z.requires_grad = True
    z_optimizer = torch.optim.Adam([z], lr=1e-3)

    for epoch in range(5000+1):
        fake_img = G_update(z)
        loss, _, _ = Anomaly_score(x, fake_img, D_update, Lambda=0.1)

        z_optimizer.zero_grad()
        loss.backward()
        z_optimizer.step()

        if epoch % 1000 == 0:
            print('epoch {} || loss_total:{:.0f} '.format(epoch, loss.item()))

    # 画像を生成
    fake_img = G_update(z)

    # 損失を求める
    loss, loss_each, residual_loss_each = Anomaly_score(
        x, fake_img, D_update, Lambda=0.1)

    # 損失の計算。トータルの損失
    loss_each = loss_each.cpu().detach().numpy()
    print("total loss：", np.round(loss_each, 0))

    # 画像を可視化
    fig = plt.figure(figsize=(15, 6))
    for i in range(0, 5):
        # 上段にテストデータを
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i][0].cpu().detach().numpy(), 'gray')

        # 下段に生成データを表示する
        plt.subplot(2, 5, 5+i+1)
        plt.imshow(fake_img[i][0].cpu().detach().numpy(), 'gray')

        plt.savefig("./result.png")