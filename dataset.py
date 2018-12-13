import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
from doa_math import to_class

class CustomDataset(Dataset):
  def __init__(self, data_entries, config):
    self.len = len(data_entries)
    self.internal_data = data_entries
    self.config = config

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    doa_classes = self.doa_classes
    try:
      data = np.load(self.internal_data[index][0])
    except Exception as e:
      print(str(e))
      print("Error loading: " + str(index))
    data = np.moveaxis(data, -1, 0)

    label = np.array(self.internal_data[index][1:4]).astype("float32")[0]
    if doa_classes:
      label = to_class(label,doa_classes)
    if output_has_all_frames(self.config):
      label = np.array([label]*25)

    return data, label

def read_data_entries(labelpath):
  # initialize dataset
  if not os.path.exists(labelpath):
    print("file does not exist")
    return 100
    
  csvfile = open(labelpath, 'r')
  csv_reader = csv.reader(csvfile, delimiter=',')
  next(csv_reader, None)
  data_entries = []
  for line in csv_reader:
    npypath = os.path.join(data_folder, line[0])
    if os.path.exists(npypath):
      data_entries.append([npypath, [float(x) for x in line[1:4]]])
    # if len(dataset)>1000:
    #     break
  return data_entries

def generate_loader(config,data_entries,shuffle=False):
  dataset = CustomDataset(data_entries,config)
  return DataLoader(dataset=train_dataset,batch_size=config.batch_size,\
                    shuffle=shuffle,num_workers=0)

def generate_loaders(config):
  train_labels_path = os.path.join(config.data_folder,'train_labels.csv')
  test_labels_path = os.path.join(config.data_folder,'test_labels.csv')

  train_data_entries = read_data_entries(train_labels_path)
  train_data_entries, val_data_entries = train_test_split(data_entries,\
                            test_size=config.test_to_all_ratio, random_state=11)
  test_data_entries = read_data_entries(test_labels_path)
  
  train_loader = generate_loader(config,train_data_entries,shuffle=True)
  val_loader = generate_loader(config,val_data_entries)
  test_loader = generate_loader(test_data_entries)
  
  return train_loader,val_loader,test_loader

