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
	out_dir = "results"
	
	try:
		http = urllib3.PoolManager(1)
		response = http.request('GET', url)
		soup = BeautifulSoup(response.data.decode('utf-8'), features="lxml")
		link = soup.find("img")
		img = link.get("src")
		filename = os.path.join(out_dir, img.split("/")[-1])
		filename = filename.split("_")[0] + ".jpg"
		print(filename)
		print(img)
		if os.path.exists(filename):
			print('Image {} already exists. Skipping download.'.format(filename))
			return
		
		urlretrieve(img, filename)
		print("{} downloaded".format(filename))
		return
	except:
		print('Warning: Could not download image from {}'.format(url))
		return


# try:  # pil_image = Image.open(BytesIO(image_data))  # print("SAVED")  # except:  # 	print('Warning: Failed to parse image')  # 	return


# try:
# 	pil_image_rgb = pil_image.convert('RGB')
# 	print("SAVED")
# except:
# 	print('Warning: Failed to convert image  to RGB')
# 	return
#
# try:
# 	pil_image_rgb.save(filename, format='JPEG', quality=90)
# 	print("SAVED")
# except:
# 	print('Warning: Failed to save image {}'.format(filename))
# 	return


def loader():
	out_dir = "results"
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)
	
	key_url_list = parse_data('scenicOrNot.tsv')
	#     print (key_url_list.tolist()[10:])
	download_image(key_url_list.tolist())
	pool = multiprocessing.Pool(processes=10)  # Num of CPUs
	key_url_list = key_url_list.tolist()
	pool.map(download_image, key_url_list)
	pool.close()
	pool.terminate()


# arg1 : data_file.csv
# arg2 : output_dir
if __name__ == '__main__':
	loader()
