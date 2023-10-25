# Machine translation using Transformers

### Prepare data (mtet or iwslt15)

```
python prepare_data_mtet.py --output-path <output-path>
```

### Train tokenizer

```
python train_tokenizer.py --train-file <train-file> --vocab-size <vocab-size> --save-dir <save-dir> --filename <save-filename>
```

### Train

```
python train.py <args>
```

### Evaluation

```
python eval.py <args>
```

See `train.py` and `eval.py` for arguments.