from torchtext import data
from torchtext import datasets

import io
import os
import string

class TranslationDataset(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))
        # not sure what this does...
    def __init__(self, path, exts, field, **kwargs):
        """ 
        Arguments:
            path: common prefix of path to the data files for both languages. 
            exts: tuple of path extensions for each language
            fields: tuple of fields used for data in each language
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1]), ('idx', data.LabelField(use_vocab=False))]

        src_path, trg_path = tuple(os.path.expanduser(path+x) for x in exts)

        examples = []
        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
            for i, (src_line, trg_line) in enumerate(zip(src_file, trg_file)):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line, i], fields)

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, exts, fields, path=None, root='.data', train='train', validation='val', test='test', **kwargs):
        """
        Arguments:
            exts: A tuple containing the path extension for each language.
            fields: tuple of fields used for data in each language
            path (str): common prefix of splits' file paths, or None to use 
                the result of cls.download(root)
            root: Root dataset storage directory. Default is '.data'
            train: the prefix of the train data. Default: 'train'
            val: prefix of val. Default: 'val'
            test: prefix of test. Default: 'test'
        """
        if path is None:
            path = cls.download(root)

            train_data = None if train is None else cls(
                os.path.join(path, train), exts, fields, **kwargs)
            val_data = None if validation is None else cls(
                os.path.join(path, validation), exts, fields, **kwargs)
            test_data = None if test is None else cls(
                os.path.join(path, test), exts, fields, **kwargs)

            return tuple(d for d in (train_data, val_data, test_data)
                if d is not None)

            
    def load_data(args):







        






    

