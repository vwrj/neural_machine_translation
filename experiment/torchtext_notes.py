data.Example.fromlist

# Our code: 
# fields = [('src', fields[0]), ('trg', fields[1]), ('idx', data.LabelField(use_vocab=False))]
# examples = []
# for i, (src_line, trg_line):
#                 if src_line != '' and trg_line != '':
#                     examples.append(data.Example.fromlist(
#                         [src_line, trg_line, i], fields))


# torchtext -- example.py
# ex = Example()
# for each name, field, data: 
#     ex.name = field.preprocess(data)

# At the end, we'll have:
# ex.src = `some data tokenized`
# ex.trg = `some data tokenized`

# This will simply tokenize each src_line and trg_line according to your tokenize func. 
# And at the end, you end up with a list of Example objects. 



super(TranslationDataset, self).__init__(examples, fields, **kwargs)

# Dataset():
# self.examples = examples
# self.fields = dict(fields)

# Each Dataset (train, val, test) has examples and access to same Field objects.
# When you type `train.src`, it gives you a generator function (yield x.src for x in self.examples) from your list of dataset Examples. 



SRC.build_vocab(train.src, min_freq=args.min_freq, max_size=args.max_vocab_size)

# simply builds the Vocab object, using data from our train dataset's SRC column. Vocab object contains a Counter for each word encountered. 
# This Vocab object is an attr of this Field. 




train_iter = data.BucketIterator(
        dataset=train_data, 
        batch_size=args.batch_size,
        repeat=False,
        sort_key=lambda x: len(x.src),
        sort_within_batch=True,
        device=device,
        train=True
    )

# When we actually build the batches, that's where padding and numericalizing come into play. 
# In the Batch class, there's this line during __init__: 
# setattr(self, name, field.process(batch, device=device))

# So that's where each Batch will look at each Field and ask it to pad/numericalize its data according to the Field's Vocab object. 
