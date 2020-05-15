#!/bin/bash
for lng in mlg ceb hil tgl mao; do
python example/sigmorphon2020-shared-tasks/augment.py task0-data/austronesian $lng --examples 10000
done
for lng in dan isl nob swe nld eng deu gmh frr ang; do
python example/sigmorphon2020-shared-tasks/augment.py task0-data/germanic $lng --examples 10000
done
for lng in nya kon lin lug sot swa zul aka gaa; do
python example/sigmorphon2020-shared-tasks/augment.py task0-data/niger-congo $lng --examples 10000
done
for lng in cpa azg xty zpv ctp czn cly otm ote pei; do
python example/sigmorphon2020-shared-tasks/augment.py task0-data/oto-manguean $lng --examples 10000
done
for lng in est fin izh krl liv vep vot mhr myv mdf sme; do
python example/sigmorphon2020-shared-tasks/augment.py task0-data/uralic $lng --examples 10000
done