import tensorflow_datasets as tfds

#     #Load dataset
ds = tfds.load('cifar10', data_dir='/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ad00878/datasets/'shuffle_files=False, as_supervised=True,with_info=True)

