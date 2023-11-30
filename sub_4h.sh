gpu=$1
config=$2
checkpoint=$3
echo $config
echo $checkpoint


for region in boxi_0015 boxi_0034 boxi_0076 roxi_0004 roxi_0005 roxi_0006 roxi_0007; do
  for year in 2019 2020; do
    python train.py --gpus $gpu --mode predict --config_path $config --name U-NET-126 --checkpoint $checkpoint --test_region ${region} --test_year ${year}
  done
done

for region in boxi_0015 boxi_0034 boxi_0076 roxi_0004 roxi_0005 roxi_0006 roxi_0007; do
  for year in 2019 2020; do
    rm submission_4h/${year}/${region}.pred.h5.gz
    gzip submission_4h/${year}/${region}.pred.h5
  done
done

cd submission_4h/
zip -r submission_4h.zip 2019/ 2020/
cd -
