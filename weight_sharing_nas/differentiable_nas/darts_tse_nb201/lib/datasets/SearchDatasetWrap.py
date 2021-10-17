##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import torch, copy, random
import torch.utils.data as data


class SearchDataset(data.Dataset):

  def __init__(self, name, data, train_split, valid_split, direct_index=False, check=True, true_length=None, merge_train_val=False):
    self.datasetname = name
    self.direct_index = direct_index
    self.merge_train_val = merge_train_val
    if isinstance(data, (list, tuple)): # new type of SearchDataset
      assert len(data) == 2, 'invalid length: {:}'.format( len(data) )
      print("V2 SearchDataset")
      self.train_data  = data[0]
      self.valid_data  = data[1]
      self.train_split = train_split.copy()
      self.valid_split = valid_split.copy()
      self.mode_str    = 'V2' # new mode 
    else:
      print("V1 Search Dataset")
      self.mode_str    = 'V1' # old mode 
      self.data        = data
      self.train_split = train_split.copy()
      self.valid_split = valid_split.copy()
      if check:
        if len(train_split) != len(valid_split) and len(train_split) < 48000 and not merge_train_val:
          intersection = set(train_split).intersection(set(valid_split))
          assert len(intersection) == 0, 'the splitted train and validation sets should have no intersection'
        else:
          print(f"Skipping checking intersection because since len(train_split)={len(train_split)}, len(valid_split)={len(valid_split)}")
    self.length      = len(self.train_split) if true_length is None else true_length

  def __repr__(self):
    return ('{name}(name={datasetname}, train={tr_L}, valid={val_L}, version={ver})'.format(name=self.__class__.__name__, datasetname=self.datasetname, tr_L=len(self.train_split), val_L=len(self.valid_split), ver=self.mode_str))

  def __len__(self):
    return self.length

  def __getitem__(self, index):

    if self.direct_index:
      assert index in self.train_split and index not in self.valid_split
      train_index = index

    else:
      assert index >= 0 and index < self.length, 'invalid index = {:}'.format(index)
      train_index = self.train_split[index]  

    valid_index = random.choice( self.valid_split )
    if not self.merge_train_val:
      assert valid_index not in self.train_split or (self.datasetname in ["cifar100", "ImageNet16-120"] and not self.merge_train_val)
    if self.mode_str == 'V1':
      train_image, train_label = self.data[train_index]
      valid_image, valid_label = self.data[valid_index]
    elif self.mode_str == 'V2':
      train_image, train_label = self.train_data[train_index]
      valid_image, valid_label = self.valid_data[valid_index]
    else: raise ValueError('invalid mode : {:}'.format(self.mode_str))
    return train_image, train_label, valid_image, valid_label
