# Machine translation using Transformers

### Prepare data

```
python prepare_data.py --output-path <output-path>
```

### Train tokenizer

```
python train_tokenizer.py --train-file <train-file> --vocab-size <vocab-size> --save-dir <save-dir> --filename <save-filename>
```

### Train

```
python train.py <args>
```

### Test

```
python test.py <args>
```

See `train.py` and `test.py` for arguments.