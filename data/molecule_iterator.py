import re
import pickle
import random
from torchtext.legacy.data import Field, Example, Dataset, Iterator, BucketIterator
pattern = "(]|\[|Br?|Cl?|As?|Mg?|Se?|Te?|Hg?|Li?|Pr?|A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|R|S|T|U|V|W|X|Y|Z|a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z|\(|\)|\.|=|#|-|\+|\/|\\\\|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


# Specific Tokenizer(Using in Field Processing)
def smi_tokenizer(smi, regex = re.compile(pattern)):
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens), 'smi:' + smi + '--tokens:' + ''.join(tokens)
    return tokens


class SmileBucketIterator(object):
    def __init__(self, vocab_file, batch_size, train_data_file, test_data_file=None, valid_data_file=None, seed_random=0,
                 random_test=True, file_read=True, load_vocab=True, training=False):
        self.batch_size = batch_size
        self.load_vocab = load_vocab
        self.training_model = training
        # Field Instantiation
        smi_field = Field(init_token='<sos>', eos_token=' ', tokenize=smi_tokenizer, pad_token=' ',
                          include_lengths=True, batch_first=True)
        fields = [('smile', smi_field)]
        # load smile data & Erasing space
        if self.training_model:
            if file_read:
                with open(train_data_file, 'r') as f:
                    train_mol_strs = f.read().strip().split('\n')
                    train_mol_strs = [mol.replace(' ', '') for mol in train_mol_strs]
                if not random_test:
                    with open(test_data_file, 'r') as f:
                        test_mol_strs = f.read().strip().split('\n')
                        test_mol_strs = [mol.replace(' ', '') for mol in test_mol_strs]
            else:
                train_mol_strs = train_data_file
                test_mol_strs = test_data_file if not random_test else None
            # Smiles Preprocessing(tokenizing)
            train_mol_strs = [smi_field.preprocess(mol) for mol in train_mol_strs]  # list: tokenized smiles_vocab lists
            test_mol_strs = [smi_field.preprocess(mol) for mol in test_mol_strs] if not random_test else None
        else:
            if file_read:
                with open(valid_data_file, 'r') as f:
                     mol_strs = f.read().strip().split('\n')
                     mol_strs = [mol.replace(' ', '') for mol in mol_strs]
            else:
                mol_strs = valid_data_file
            # Smiles Preprocessing(tokenizing)
            mol_strs = [smi_field.preprocess(mol) for mol in mol_strs]

        # Building Smiles Examples
        if self.training_model:
            if random_test:
            # Random select test smiles
                if seed_random != 0:
                    random.seed(seed_random)
                test_mol_strs = random.sample(train_mol_strs, 10000)
                for element in test_mol_strs:
                    train_mol_strs.remove(element)
            train_smi_examples = []
            for mol in train_mol_strs:
                ex = Example.fromlist([mol, None], fields)
                train_smi_examples.append(ex)
            test_smi_examples = []
            for mol in test_mol_strs:
                ex = Example.fromlist([mol, None], fields)
                test_smi_examples.append(ex)
        else:
            smi_examples = []
            for mol in mol_strs:
                ex = Example.fromlist([mol, None], fields)
                smi_examples.append(ex)
        # Load or build vocab
        if self.load_vocab:
            # print('Loading vocab from:', vocab_file)
            smi_field.vocab = pickle.load(open(vocab_file, 'rb'))
        else:
            print('build and save vocab file:', vocab_file)
            smi_field.build_vocab(mol_strs)
            pickle.dump(smi_field.vocab, open(vocab_file, 'wb'), protocol=2)
        # Vocab Properties
        self.vocab = smi_field.vocab
        self.vocab_size = len(smi_field.vocab.itos)
        self.padding_idx = smi_field.vocab.stoi[smi_field.pad_token]
        self.sos_idx = smi_field.vocab.stoi[smi_field.init_token]
        self.eos_idx = smi_field.vocab.stoi[smi_field.eos_token]
        self.unk_idx = smi_field.vocab.stoi[smi_field.unk_token]
        # Dataset Building
        if self.training_model:
            self.train_smi = Dataset(train_smi_examples, fields=fields)
            self.test_smi = Dataset(test_smi_examples, fields=fields)
        else:
            self.valid_smi = Dataset(smi_examples, fields=fields)

    def train_bucket_iter(self, batch_size=None):
        bsize = self.batch_size if batch_size is None else batch_size
        return BucketIterator(self.train_smi, batch_size=bsize, train=True,
                              sort_within_batch=True, repeat=False, sort_key=lambda x: len(x.smile))

    def test_bucket_iter(self, batch_size=None):
        bsize = self.batch_size if batch_size is None else batch_size
        return BucketIterator(self.test_smi, batch_size=bsize, train=False,
                              sort_within_batch=True, repeat=False, sort_key=lambda x: len(x.smile))

    def valid_bucket_iter(self, batch_size=None):
        bsize = self.batch_size if batch_size is None else batch_size
        return BucketIterator(self.valid_smi, batch_size=bsize, train=False,
                              sort_within_batch=True, repeat=False, sort_key=lambda x: len(x.smile))

    def test_bucket_iter_nosort(self, batch_size=None):
        bsize = self.batch_size if batch_size is None else batch_size
        return BucketIterator(self.test_smi, batch_size=bsize, train=False, sort=False,
                              sort_within_batch=False, repeat=False, sort_key=None)

    def valid_bucket_iter_nosort(self, batch_size=None):
        bsize = self.batch_size if batch_size is None else batch_size
        return BucketIterator(self.valid_smi, batch_size=bsize, train=False, sort=False,
                              sort_within_batch=False, repeat=False, sort_key=None)

    def get_vocab(self):
        return self.vocab

class SmileSingleIterator(object):
    def __init__(self, smiles, vocab_file, device='cpu', vocab_direct_read=False):
        self.device = device
        # Field Instantiation
        smi_field = Field(init_token='<sos>', eos_token=' ', tokenize=smi_tokenizer, pad_token=' ',
                          include_lengths=True, batch_first=True)
        fields = [('smile', smi_field)]
        # load smile & Erasing space
        mol_strs = smiles.replace(' ', '')
        # Smiles Preprocessing(tokenizing)
        mol_strs = smi_field.preprocess(mol_strs)  # tokenized smiles_vocab lists
        # Building Smiles Examples
        smi_example = [Example.fromlist([mol_strs, None], fields)]
        # Load vocab
        if vocab_direct_read:
            smi_field.vocab = vocab_file
        else:
            smi_field.vocab = pickle.load(open(vocab_file, 'rb'))
        # Vocab Properties
        self.vocab = smi_field.vocab
        self.vocab_size = len(smi_field.vocab.itos)
        self.padding_idx = smi_field.vocab.stoi[smi_field.pad_token]
        self.sos_idx = smi_field.vocab.stoi[smi_field.init_token]
        self.eos_idx = smi_field.vocab.stoi[smi_field.eos_token]
        self.unk_idx = smi_field.vocab.stoi[smi_field.unk_token]
        # Dataset Building
        self.dataset_smi = Dataset(smi_example, fields=fields)
        self.smi = self.dataset_smi

    def return_iter(self):
        return Iterator(self.smi, batch_size=1, train=False, device=self.device, sort=None, sort_within_batch=None)

    def get_vocab(self):
        return self.vocab