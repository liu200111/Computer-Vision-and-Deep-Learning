from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFrame, QMessageBox

import os
import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
from window import Ui_Form

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg19_bn
import torchvision.datasets as dset
import torchvision.utils as vutils
from PIL import Image
import torchsummary
from PyQt5.QtGui import QFont, QPixmap, QImage
from torch.utils.data import TensorDataset
import zipfile

# methods and operations
class Main_Frame(QFrame, Ui_Form):
    def __init__(self):
        super(Main_Frame, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        
        # Q1 VGG19
        self.ui.button_1_1.clicked.connect(self.load_image)
        self.ui.button_1_2.clicked.connect(self.show_augmented_image)
        self.ui.button_1_3.clicked.connect(self.show_model_structure_vgg19)
        self.ui.button_1_4.clicked.connect(self.show_accuracy_loss)
        self.ui.button_1_5.clicked.connect(self.inference)
        
        # Q2 DcGAN
        self.ui.button_2_1.clicked.connect(self.show_training_images)
        self.ui.button_2_2.clicked.connect(self.show_model_structure_dc_gan)
        self.ui.button_2_3.clicked.connect(self.show_training_loss)
        self.ui.button_2_4.clicked.connect(self.show_generated_images)
        
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
    # Q1 Training a CIFAR10 Classifier Using VGG19 with BN   
    def load_image(self):
        self.file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "select image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if self.file_path:
            self.image = Image.open(self.file_path)
            pixmap = QPixmap(self.file_path).scaled(128, 128)
            self.ui.image_label.setPixmap(pixmap)
           
    def show_augmented_image(self):
        folder_path = 'Dataset_CvDl_Hw2\Q1_image\Q1_1'
        aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ])

        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        loaded_images = []
        labels = []

        for img_file in image_files[:9]:  # Load first 9 images
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path)
            augmented_img = aug_transform(img)
            loaded_images.append(augmented_img)
            labels.append(os.path.splitext(img_file)[0])  # Use file name as label

        loaded_images = torch.stack(loaded_images)

        def imshow_with_labels(imgs, labels):
                    # imgs = imgs / 2 + 0.5  # unnormalize
                    npimg = imgs.numpy()
                    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
                    for i, ax in enumerate(axes.flat):
                        ax.imshow(np.transpose(npimg[i], (1, 2, 0)))
                        ax.set_title(labels[i])
                        # ax.axis('off')
                    plt.tight_layout()
                    plt.show(block=False)

        # Display augmented images with labels
        imshow_with_labels(loaded_images, labels)
        
    def show_model_structure_vgg19(self):
        self.model = vgg19_bn(pretrained=False, num_classes=10)
        
        # Move model to device (GPU if available) before calling summary
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Create a sample input tensor on the same device
        input_data = torch.randn(1, 3, 32, 32).to(self.device)

        torchsummary.summary(self.model, input_data.shape[1:]) # Pass input_size without batch dimension
        
    def show_accuracy_loss(self):
        img_acc_loss = cv2.imread('acc_loss_vgg19.png')
        cv2.imshow('Accuracy and Loss', img_acc_loss)
    
    def inference(self):
        canvas_image = self.ui.image_label.pixmap().toImage()
        tensor_image = self.qimage_to_tensor(canvas_image)
        
        self.model.load_state_dict(torch.load('vgg19.pth', map_location=torch.device('cpu')))

        test_transforms = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        self.model.eval()
        input_image = tensor_image
        input_image = test_transforms(input_image).unsqueeze(0) # (1, 3, height, width)
        output = self.model(input_image) 
        
        classes = self.classes
        _, predicted = torch.max(output, 1)
        predicted_class = classes[predicted]
        self.ui.result_label.setText('Predicted = ' + str(predicted_class))
        print('Predicted = ', predicted_class)
        
        # Bar chart
        softmax = nn.Softmax(dim=1)
        output = softmax(output).squeeze()
        output = output.detach().numpy()
        plt.bar(classes, output)
        plt.title('Probability of each class')
        plt.xticks(classes, fontsize=8)
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.show()
    
    # Q2 Training a DCGAN on CIFAR10
    def show_training_images(self):
        image_size = 64
        batch_size = 128
        ngpu = 1
        workers = 2
        
        # os.makedirs('Q2_images', exist_ok=True)
        # with zipfile.ZipFile('Dataset_CvDl_Hw2/Q2_images.zip', 'r') as zip_ref:
        #     zip_ref.extractall('Q2_images')
        
        dataroot = "Q2_images"
        
        # Dataset without augmentation
        dataset_original = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
        
        # Dataset with augmentation
        dataset_augmented = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

        # Create dataloaders
        dataloader_original = torch.utils.data.DataLoader(dataset_original, batch_size=batch_size, shuffle=True, num_workers=workers)
        dataloader_augmented = torch.utils.data.DataLoader(dataset_augmented, batch_size=batch_size, shuffle=True, num_workers=workers)

        # Decide which device we want to run on
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        # Get a batch of images
        self.real_batch_original = next(iter(dataloader_original))
        self.real_batch_augmented = next(iter(dataloader_augmented))

        # Plot the images
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Training Images (Original)")
        plt.imshow(np.transpose(vutils.make_grid(self.real_batch_original[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Training Images (Augmented)")
        plt.imshow(np.transpose(vutils.make_grid(self.real_batch_augmented[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.show()
    
    def show_model_structure_dc_gan(self):
        # Number of channels in the training images
        nc = 3

        # Size of z latent vector (i.e. size of generator input)
        nz = 100

        # Size of feature maps in generator
        ngf = 64
        
        # Size of feature maps in discriminator
        ndf = 64
        
        # Generator Model
        class Generator(nn.Module):
            def __init__(self, ngpu):
                super(Generator, self).__init__()
                self.ngpu = ngpu
                self.main = nn.Sequential(
                    nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
                    nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(True),
                    nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
                    nn.Tanh()
                )

            def forward(self, input):
                return self.main(input)

        # Discriminator Model
        class Discriminator(nn.Module):
            def __init__(self, ngpu):
                super(Discriminator, self).__init__()
                self.ngpu = ngpu
                self.main = nn.Sequential(
                    nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                    nn.Sigmoid()
                )

            def forward(self, input):
                return self.main(input)
            
        # Initialize models
        ngpu = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.netG = Generator(ngpu).to(device)
        self.netD = Discriminator(ngpu).to(device)
        # Initialize weights
        def weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        # Print model structures
        print("Generator Model:\n", self.netG)
        print("Discriminator Model:\n", self.netD)
    
    def show_training_loss(self):
        img_gan_loss = cv2.imread('G_D_loss.png')
        cv2.imshow('Generator and Discriminator Loss During Training', img_gan_loss)
    
    def show_generated_images(self):
        # Load the trained Generator model
        self.netG.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))
        self.netG.eval()

        # Define parameters
        nz = 100  # Size of the latent vector
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Generate random noise as input for the generator
        noise = torch.randn(64, nz, 1, 1, device=device)

        # Generate fake images
        with torch.no_grad():
            fake_images = self.netG(noise).detach().cpu()

        # Visualize the generated images
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(self.real_batch_augmented[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
        
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(vutils.make_grid(fake_images, padding=2, normalize=True), (1, 2, 0)))
        plt.show()
    
    def qimage_to_tensor(self, qimage):
        width, height = qimage.width(), qimage.height()
        byte_format = qimage.format()

        if byte_format == QImage.Format_RGB32:
            buffer = bytes(qimage.bits().asstring(4 * width * height))
            image_array = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
            image_array = image_array[:, :, :3]
        elif byte_format == QImage.Format_RGB888:  
            buffer = bytes(qimage.bits().asstring(3 * width * height))
            image_array = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
        else:
            raise ValueError("Unsupported image format")

        return image_array