# -*- coding: utf-8 -*-
# !/usr/local/bin/python3

import os
from urllib.request import urlretrieve

import pandas as pd
import urllib3
from bs4 import BeautifulSoup


data = pd.read_csv('scenicOrNot.tsv', sep='\t')
data.head()

image_URL = data['Geograph URI'][0]
print(image_URL)

import multiprocessing


def parse_data(data_file):
	data = pd.read_csv(data_file, sep='\t')
	image_URL = data['Geograph URI']
	return image_URL


def download_image(url):
	out_dir = "dataset"
	
	try:
		http = urllib3.PoolManager(100)
		response = http.request('GET', url)
		soup = BeautifulSoup(response.data.decode('utf-8'), features="lxml")
		link = soup.find("img")
		img = link.get("src")
		filename = os.path.join(out_dir, img.split("/")[-1])
		filename = str(filename.split("_")[0]) + ".jpg"
		# print(filename)
		# print(img)
		if os.path.exists(filename):
			print('Image {} already exists. Skipping download.'.format(filename))
			return
		
		urlretrieve(img, filename)
		print("{} downloaded".format(filename))
		return
	except Exception as e:
		print('Warning: Could not download image from {}, {}'.format(url, e))
		return


def loader():
	out_dir = "datasest"
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)
	
	key_url_list = parse_data('scenicOrNot.tsv')
	#     print (key_url_list.tolist()[10:])
	download_image(key_url_list.tolist())
	pool = multiprocessing.Pool(processes=100)  # Num of CPUs
	key_url_list = key_url_list.tolist()
	pool.map(download_image, key_url_list)
	pool.close()
	pool.terminate()


def split(dir="datasest"):
	import os
	import pathlib
	source1 = dir
	train_set = "train_dir/"
	test_set = "test_dir/"
	valid_set = "valid_dir/"
	pathlib.Path(train_set).mkdir(parents=True, exist_ok=True)
	pathlib.Path(test_set).mkdir(parents=True, exist_ok=True)
	pathlib.Path(valid_set).mkdir(parents=True, exist_ok=True)
	
	files = os.listdir(source1)
	import shutil
	import numpy as np
	for f in files:
		rand = np.random.rand(1)
		if rand < 0.2:
			shutil.copy(source1 + '/' + f, test_set + '/' + f)
		elif 0.2 < rand < 0.9:
			shutil.copy(source1 + '/' + f, train_set + '/' + f)
		else:
			shutil.copy(source1 + '/' + f, valid_set + '/' + f)


# arg1 : data_file.csv
# arg2 : output_dir
if __name__ == '__main__':
	loader()
	split()
