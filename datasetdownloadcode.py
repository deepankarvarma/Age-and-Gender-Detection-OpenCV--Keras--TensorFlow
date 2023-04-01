import os
import gdown
import tarfile

# Define the URL for the UTKFace dataset
url = 'https://drive.google.com/u/0/uc?id=0BxYys69jI14kYVM3aVhKS1VhRUk&export=download'

# Define the directory to store the dataset
data_dir = './data'

# Create the directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Download the tar file
filename = 'UTKFace.tar.gz'
filepath = os.path.join(data_dir, filename)
gdown.download(url, filepath, quiet=False)

# Extract the tar file
tar = tarfile.open(filepath, 'r:gz')
tar.extractall(data_dir)
tar.close()
