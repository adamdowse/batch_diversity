import tensorflow_datasets as tfds

#     #Load dataset
#/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/datasets/
ds = tfds.load('mnist', data_dir='/com.docker.devenvironments.code/datasets/',shuffle_files=False, as_supervised=True,with_info=True,download=True)

