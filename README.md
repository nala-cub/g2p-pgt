# SIGMORPHON 2020 Shared Tasks

## Task 1

This repository is a fork of the vanilla transformer implementation of shijie-wu, providing the extra features of Multi-tasking for G2P (Grapheme-to-Phoneme Conversion) and P2G (Phoneme-to-Grapheme Conversion), and a pointer-generator copy-mechanism.


## Dependencies

- python 3
- pytorch==1.4
- numpy
- tqdm
- fire


## Install

```bash
make
```


## Train Example

```bash
python src/train.py --dataset g2p --train data/kor_train.tsv --test data/kor_test.tsv --dev data/kor_dev.tsv --arch transformer --model model/kor --opt_pickles model/kor-opt-pickle --align data/kor_alignment.pkl --use_copy True --seed 12
```

### Flags
```
--use_copy [True|False]   use pointer-generator mechanism
--opt_pickles [path]    path to pickle file storing performance results including WER and PER
--align [path]    path to alignment file (which we generated using GIZA++)
--freq_eval_after [INT]    Epoch after which evaluation starts
--freq_eval_every [INT]    Frequency of evaluation after freq_eval_after
```


## License

MIT
