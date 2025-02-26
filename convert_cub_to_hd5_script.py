import os
import numpy as np
import h5py
from glob import glob
import torchfile
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# Paths from config
images_path = config['birds_images_path']
embedding_path = config['birds_embedding_path']
text_path = config['birds_text_path']
datasetDir = config['birds_dataset_path']

# Read class lists
val_classes = open(config['val_split_path']).read().splitlines()
train_classes = open(config['train_split_path']).read().splitlines()
test_classes = open(config['test_split_path']).read().splitlines()

# Create HDF5 file
f = h5py.File(datasetDir, 'w')
train = f.create_group('train')
valid = f.create_group('valid')
test = f.create_group('test')

# Iterate over classes
i = 0
for _class in sorted(os.listdir(embedding_path)):
    # Assign to correct split
    if _class in train_classes:
        split_group = train  
    elif _class in val_classes:
        split_group = valid
    elif _class in test_classes:
        split_group = test
    else:
        continue  # Skip unknown classes

    data_path = os.path.join(embedding_path, _class)
    txt_path = os.path.join(text_path, _class)


    
    # Process embeddings & text files
    for example, txt_file in zip(sorted(glob(data_path + "/*.t7")), sorted(glob(txt_path + "/*.txt"))):

        # Load .t7 file
        example_data = torchfile.load(example)

        # Convert keys from bytes to strings
        example_data = {key.decode('utf-8') if isinstance(key, bytes) else key: value for key, value in example_data.items()}

        # Extract image path
        img_path = example_data['img']
        if isinstance(img_path, bytes):
            img_path = img_path.decode('utf-8')


        # Extract embeddings
        embeddings = example_data['txt']
        if isinstance(embeddings, bytes):
            embeddings = embeddings.decode('utf-8').numpy()


        # Extract example name
        example_name = img_path.split('/')[-1][:-4]

        # Read text captions
        with open(txt_file, "r") as f:
            txt = f.readlines()


        # Resolve full image path
        img_path = os.path.join(images_path, img_path)

        # Read image file
        with open(img_path, 'rb') as img_file:
            img = img_file.read()

        # Select random 5 captions
        txt_choice = np.random.choice(range(10), 5, replace=False)

        embeddings = embeddings[txt_choice]
        txt = np.array(txt)[txt_choice]
        dt = h5py.special_dtype(vlen=str)

        # Store data in HDF5
        for c, e in enumerate(embeddings):
            ex = split_group.create_group(f"{example_name}_{c}")
            ex.create_dataset('name', data=example_name)
            ex.create_dataset('img', data=np.void(img))
            ex.create_dataset('embeddings', data=e)
            ex.create_dataset('class', data=_class)
            ex.create_dataset('txt', data=txt[c].astype(object), dtype=dt)

        print(f"Stored {example_name}")

# Close HDF5 file
f.close()
print("HDF5 Dataset Created Successfully!")



