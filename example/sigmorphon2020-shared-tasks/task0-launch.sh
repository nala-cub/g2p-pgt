run=bash

for lang in aka ang azg ceb cly cpa ctp czn dan deu eng est fin frr gaa gmh hil isl izh kon krl lin liv lug mao mdf mhr mlg myv nld nob nya ote otm pei sme sot swa swe tgl vep vot xty zpv zul; do
# regular data
echo trm $lang
$run example/sigmorphon2020-shared-tasks/task0-trm.sh $lang
echo mono $lang
$run example/sigmorphon2020-shared-tasks/task0-mono.sh $lang

# augmented data
echo trm hall $lang
$run example/sigmorphon2020-shared-tasks/task0-hall-trm.sh $lang
echo mono hall $lang
$run example/sigmorphon2020-shared-tasks/task0-hall-mono.sh $lang
done

for lang in austronesian germanic niger-congo oto-manguean uralic; do
# regular data
echo trm $lang
$run example/sigmorphon2020-shared-tasks/task0-trm.sh $lang
echo mono $lang
$run example/sigmorphon2020-shared-tasks/task0-mono.sh $lang

# augmented data
echo trm hall $lang
$run example/sigmorphon2020-shared-tasks/task0-hall-trm.sh $lang
echo mono hall $lang
$run example/sigmorphon2020-shared-tasks/task0-hall-mono.sh $lang
done