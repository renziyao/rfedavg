from torchtext.experimental.datasets.text_classification import AmazonReviewPolarity
from src.data.utils import *

root_dir = './data/'
vectors_cache = './data/vector_cache/'

(train_dataset, test_dataset) = AmazonReviewPolarity(root=root_dir)
vocab = train_dataset.get_vocab()
vocab.__init__(
    vocab.freqs, 
    max_size=2998, 
    vectors='glove.6B.300d',
    vectors_cache=vectors_cache,
)
train_dataset = Subset(train_dataset, list(range(1000)))
test_dataset = Subset(test_dataset, list(range(1000)))

def non_iid_percent(params):
    num_user = params['Trainer']['n_clients']
    s = params['Dataset']['s']
    dataset_split = split_dataset_by_percent(train_dataset, test_dataset, s, num_user, lambda x: x[0])
    for item in dataset_split: item['vocab'] = vocab
    testset_dict = {
        'train': None,
        'test': test_dataset,
        'vocab': vocab,
    }
    return dataset_split, testset_dict