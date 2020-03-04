# SIGMORPHON 2020 Shared Tasks

## Task 0 Baseline

First download and augment [(Anastasopoulos and Neubig, 2019)](https://arxiv.org/abs/1908.05838) the data

```bash
git clone https://github.com/sigmorphon2020/task0-data.git
bash example/sigmorphon2020-shared-tasks/augment.sh
python example/sigmorphon2020-shared-tasks/task0-build-dataset.py all
```

Run hard monotonic attention [(Wu and Cotterell, 2019)](https://arxiv.org/abs/1905.06319) and the transformer [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), both one model per language and one model per language family.
```bash
bash example/sigmorphon2020-shared-tasks/task0-launch.sh
```


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


## License

MIT
