import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision import models


checkpointd_dir = "checkpoints/"
model_name = "vanila_vgg"
TRAIN = 'train'
VAL = 'val'
TEST = 'test'
'''# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transforms = {
    TRAIN: transforms.Compose([
        # Data augmentation is a good practice for the train set
        # Here, we randomly crop the image to 224x224 and
        # randomly flip it horizontally.
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    VAL: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
    TEST: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
}'''
transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])


class SenicDataset(Dataset):
	"""
		Custom Dataset
		Parameters:
			root_dir - directory including category folders with images

		Example:
		images/
			1000001859/
				26_0.jpg
				26_1.jpg
				...
			1000004141/
				...
			...
	"""
	
	def __init__(self, root_dir, transform=None):
		self.data = pd.read_csv('scenicOrNot.tsv', sep='\t').dropna()
		self.data['cat'] = np.ceil(self.data['Average'])
		self.data['name'] = self.data['Geograph URI'].str.split("/")
		self.data['name'] = self.data['name'].apply(lambda s: s[-1])
		self.root_dir = root_dir
		self.categories = set(self.data['cat'])
		# self.cat2idx = dict(zip(self.categories, range(len(self.categories))))
		# self.idx2cat = dict(zip(self.cat2idx.values(), self.cat2idx.keys()))
		self.files = []
		for (dirpath, dirnames, filenames) in os.walk(self.root_dir):
			for f in filenames:
				if f.endswith('.jpg'):
					row = self.data[self.data['name'].str.contains(f.lstrip("0").split(".")[0])]
					
					o = dict()
					o['img_path'] = dirpath + '/' + f
					o['category'] = row['cat']
					self.files.append(o)
		self.transform = transform
	
	def __len__(self):
		return len(self.files)
	
	def __getitem__(self, idx):
		img_path = self.files[idx]['img_path']
		category = self.files[idx]['category']
		print(img_path)
		print(str(category.iloc[0]))
		image = cv2.imread(img_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		if self.transform:
			image = self.transform(image)
		
		return {'image': image, 'category': category.iloc[0]}


train_set = SenicDataset('train_dir', transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)

valid_set = SenicDataset('valid_dir', transform=transform)
valid_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)

test_set = SenicDataset('test_dir', transform=transform)
test_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)

print('######### Dataset class created #########')
print('Number of images: ', len(train_set))
print('Number of categories: ', len(train_set.categories))
print('Sample image shape: ', train_set[0]['image'].shape, end='\n\n')



### Define your network below
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 4)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
	
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


class ScenicModel:
	def __init__(self):
		self.ckpoint_path = checkpointd_dir + "/" + model_name
		self.epochs = 40
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		# model = Net()
		squeezenet = models.vgg11_bn(pretrained=False)
		self.model = squeezenet  # .double()
		self.model = self.model.to(self.device)
		
		if self.device == 'cuda':
			print("Let's use", torch.cuda.device_count(), "GPUs!")
			self.model = torch.nn.DataParallel(self.model)
			cudnn.benchmark = True
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
		
		print('######### Network created #########')
		print('Architecture:\n', self.model)
	
	def train(self):
		for epoch in range(self.epochs):
			running_loss = 0.0
			examples = 0
			for i, data in enumerate(train_loader, 0):
				# Get the inputs
				inputs, labels = data['image'], data['category']
				
				# Wrap them in Variable
				inputs, labels = Variable(inputs.float()).to(self.device), Variable(labels.long()).to(self.device)
				
				# Zero the parameter gradients
				self.optimizer.zero_grad()
				
				# forward + backward + optimize
				outputs = self.model(inputs)
				loss = self.criterion(outputs, labels)
				loss.backward()
				self.optimizer.step()
				
				# Print statistics
				running_loss += loss.data[0]
				examples += 4
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / examples))
		self.save_model()
		print('Finished Training')
	
	def save_model(self):
		torch.save(self.model.state_dict(), self.ckpoint_path)
		print('Saved to={}'.format(self.ckpoint_path))
		return
	
	def test(self):
		self.model.load_state_dict(torch.load(self.ckpoint_path))
		self.model.eval()
		for i, (input, _) in enumerate(test_loader):
		if cuda:
			input = input.cuda(async=True)
		input_var = torch.autograd.Variable(input, volatile=True)
		
		# compute output
		output = model(input_var)
		# Take last layer output
		if isinstance(output, tuple):
			output = output[len(output)-1]
		
		# print (output.data.max(1, keepdim=True)[1])
		lab = classes[numpy.asscalar(output.data.max(1, keepdim=True)[1].cpu().numpy())]
		print ("Images: " + next(names) + ", Classified as: " + lab)

model = ScenicModel()
model.train()
