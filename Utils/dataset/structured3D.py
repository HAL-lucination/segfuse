import torch
import cv2
import glob
import os
import numpy as np
from random import Random
from torch.utils.data import Dataset
import OpenEXR, Imath, array
import math
import torch.utils.data
from skimage import io
import os.path as osp


# class S3D(Dataset):
#     def __init__(self, root_path, width, height, subset = None, seed = 1313):
#         super().__init__()
#         self._root_path = root_path
#         self._width = width
#         self._height = height
#         self._paths = glob.glob(f"{root_path}\\*\\*\\*\\*")
#         self._seed = seed
#         self._rng = Random(seed)
#         self._room_lightings = ["rawlight", "coldlight", "warmlight"]
#         if subset is not None:
#             self._paths = self._paths[:int(len(self._paths) * subset)]
#
#
#         self._resize = (width != 1024) or (height != 512) #S3D
#
#     def __getitem__(self, i):
#         light_type = self._rng.choice(self._room_lightings)
#         path = os.path.join(self._paths[i], "full")
#         rgb = cv2.imread(os.path.join(path, f"rgb_{light_type}.png"))
#         depth = cv2.imread(os.path.join(path, "depth.png"), -1).astype(np.float)
#         if self._resize:
#             rgb = cv2.resize(rgb, (self._width, self._height), cv2.INTER_CUBIC)
#             depth = cv2.resize(depth, (self._width, self._height), cv2.INTER_NEAREST)
#         return torch.from_numpy(rgb).float().permute(2,0,1) / 255, torch.from_numpy(depth).float().unsqueeze(0) / 1000
#
#     def __len__(self):
#         return len(self._paths)
# class S3D(Dataset):
#     def __init__(self, root_path, width, height, subset = None, seed = 1313):
#         super().__init__()
#         self._root_path = root_path
#         self._width = width
#         self._height = height
#         self.path_to_img_list = 'C:/Users/Hal/OmniPerson/OmniDepth/batch_test/train_list.txt'
# 		self.image_list = np.loadtxt(path_to_img_list, dtype=str)
#
# 		# Max depth for GT
# 		self.max_depth = 8.0
#
#     def __getitem__(self, i):
#         self._resize = (width != 1024) or (height != 512) #S3D
# 		relative_paths = self.image_list[idx]
# 		relative_basename = osp.splitext((relative_paths[0]))[0]
# 		basename = osp.splitext(osp.basename(relative_paths[0]))[0]
# 		rgb = self.readRGBPano(osp.join(self.root_path, relative_paths[0]))
# 		depth = self.readDepthPano(osp.join(self.root_path, relative_paths[1]))
#         if self._resize:
#             rgb = cv2.resize(rgb, (self._width, self._height), cv2.INTER_CUBIC)
#             depth = cv2.resize(depth, (self._width, self._height), cv2.INTER_NEAREST)
#         return torch.from_numpy(rgb).float().permute(2,0,1) / 255, torch.from_numpy(depth).float().unsqueeze(0) / 1000
#
# 	def __len__(self):
# 		'''Return the size of this dataset'''
# 		return len(self.image_list)
#
# 	def readRGBPano(self, path):
# 		'''Read RGB and normalize to [0,1].'''
# 		rgb = cv2.imread(path).astype(np.float32) / 255.
# 		# change io.imread to cv2.imread solve the channel problem!!!
# 		return rgb
#
#
# 	def readDepthPano(self, path):
# 		return self.read_exr(path)[...,0].astype(np.float32)
#
#
# 	def read_exr(self, image_fpath):
# 		f = OpenEXR.InputFile( image_fpath )
# 		dw = f.header()['dataWindow']
# 		w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
# 		im = np.empty( (h, w, 3) )
#
# 		# Read in the EXR
# 		FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
# 		channels = f.channels( ["R", "G", "B"], FLOAT )
# 		for i, channel in enumerate( channels ):
# 			im[:,:,i] = np.reshape( array.array( 'f', channel ), (h, w) )
# 		return im
#
#
# if __name__ == "__main__":
#     d = S3D(r"D:\VCL\Users\vlad\Datasets\Structure3D\Structure3D\Structured3D_splited\train", 512,256)
#     rgb, dpeth = d[0]
#     z = True


class OmniDepthDataset(torch.utils.data.Dataset):
	'''PyTorch dataset module for effiicient loading'''

	def __init__(self,
		root_path,
		path_to_img_list):

		# Set up a reader to load the panos
		self.root_path = root_path

		# Create tuples of inputs/GT
		self.image_list = np.loadtxt(path_to_img_list, dtype=str)

		# Max depth for GT
		self.max_depth = 20.0


	def __getitem__(self, idx):
		'''Load the data'''

		# Select the panos to load
		relative_paths = self.image_list[idx]

		# Load the panos
		relative_basename = osp.splitext((relative_paths[0]))[0]
		basename = osp.splitext(osp.basename(relative_paths[0]))[0]
		rgb = self.readRGBPano(osp.join(self.root_path, relative_paths[0]))
		depth = self.readDepthPano(osp.join(self.root_path, relative_paths[1]))
		depth_mask = ((depth <= self.max_depth) & (depth > 0.)).astype(np.uint8)
		# depth_mask = (depth > 0.).astype(np.uint8)

		# Threshold depths
		depth *= depth_mask

		# Make a list of loaded data
		pano_data = [rgb, depth, depth_mask, basename]

		# Convert to torch format
		pano_data[0] = torch.from_numpy(pano_data[0].transpose(2,0,1)).float()
		pano_data[1] = np.flip(pano_data[1],axis=0).copy()
		pano_data[1] = torch.from_numpy(pano_data[1][None,...]).float()
		pano_data[2] = torch.from_numpy(pano_data[2][None,...]).float()

		# Return the set of pano data
		return pano_data

	def __len__(self):
		'''Return the size of this dataset'''
		return len(self.image_list)

	def readRGBPano(self, path):
		'''Read RGB and normalize to [0,1].'''
		rgb = cv2.imread(path).astype(np.float32) / 255.
		# change io.imread to cv2.imread solve the channel problem!!!
		return rgb


	def readDepthPano(self, path):
		return self.read_exr(path)[...,0].astype(np.float32)


	def read_exr(self, image_fpath):
		f = OpenEXR.InputFile( image_fpath )
		dw = f.header()['dataWindow']
		w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
		im = np.empty( (h, w, 3) )

		# Read in the EXR
		FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
		channels = f.channels( ["R", "G", "B"], FLOAT )
		for i, channel in enumerate( channels ):
			im[:,:,i] = np.reshape( array.array( 'f', channel ), (h, w) )
		return im
