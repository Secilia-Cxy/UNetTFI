# UNetTFI

## Overview
This repository contains the code and UNetTFI models used in [Weather4cast](https://www.weather4cast.org) competition. It includes trained models, their weights, configurations, and scripts to reproduce our results.

## Introduction
The aim of the 2023 edition of the Weather4cast competition is to predict **quantitatively** future high resolution rainfall events from lower resolution satellite radiances. 

## Repository Structure
- `checkpoints/`: Contains the trained model weights.
- `models/`: Contains model architecture files.
  - `configurations/`: Contains YAML configuration files for model training and inference.
- `utils/`: Utility scripts including data loaders and evaluation scripts.
- `sub_4h.sh`, `sub_4h.trans.sh`, `sub_8h.sh`: Scripts to generate submissions using the models.
- `train.py`: The main training script for the models.
- `UNetTFI.yaml`: The conda envirnment.
- `COPYING`: The file containing the copyright information.
- `LICENSE`: The license file for the project.
- `README.md`: This file, explaining the project and setup.

## Environment Setup
To create an environment with the required dependencies, run:
```bash
conda env create -f w4cNew.yaml
```

Activate the environment with:
```bash
conda activate w4cNew
```

## Generate Submissions
To generate submissions, execute the following scripts in the repository's root directory, ensuring that the correct GPU index, configuration file, and model checkpoint path are provided. Model weights can be found in releases.

For the UNetTFI 4-hour prediction on nowcasting dataset:
```bash
sh sub_4h.sh [gpu] models/configurations/UNetTFI_4h.yaml "checkpoints/UNetTFI_4h.ckpt"
```

For the UNetTFI 4-hour prediction on transfer dataset:
```bash
sh sub_4h.trans.sh [gpu] models/configurations/UNetTFI_4h_trans.yaml "checkpoints/UNetTFI_4h.ckpt"
```

For the UNetTFI 8-hour prediction on core challenge dataset:
```bash
sh sub_8h.sh [gpu] models/configurations/UNetTFI_8h.yaml "checkpoints/UNetTFI_8h.ckpt"
```


## Citation

```
@InProceedings{pmlr-v220-gruca22a,
  title = 	 {Weather4cast at NeurIPS 2022: Super-Resolution Rain Movie Prediction under Spatio-temporal Shifts},
  author =       {Gruca, Aleksandra and Serva, Federico and Lliso, Lloren\c{c} and R\'ipodas, Pilar and Calbet, Xavier and Herruzo, Pedro and Pihrt, Ji\v{r}\'{\i} and Raevskyi, Rudolf and \v{S}im\'{a}nek, Petr and Choma, Matej and Li, Yang and Dong, Haiyu and Belousov, Yury and Polezhaev, Sergey and Pulfer, Brian and Seo, Minseok and Kim, Doyi and Shin, Seungheon and Kim, Eunbin and Ahn, Sewoong and Choi, Yeji and Park, Jinyoung and Son, Minseok and Cho, Seungju and Lee, Inyoung and Kim, Changick and Kim, Taehyeon and Kang, Shinhwan and Shin, Hyeonjeong and Yoon, Deukryeol and Eom, Seongha and Shin, Kijung and Yun, Se-Young and {Le Saux}, Bertrand and Kopp, Michael K and Hochreiter, Sepp and Kreil, David P},
  booktitle = 	 {Proceedings of the NeurIPS 2022 Competitions Track},
  pages = 	 {292--313},
  year = 	 {2022},
  editor = 	 {Ciccone, Marco and Stolovitzky, Gustavo and Albrecht, Jacob},
  volume = 	 {220},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {28 Nov--09 Dec},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v220/gruca22a.html},
}


@INPROCEEDINGS{9672063,  
author={Herruzo, Pedro and Gruca, Aleksandra and Lliso, Llorenç and Calbet, Xavier and Rípodas, Pilar and Hochreiter, Sepp and Kopp, Michael and Kreil, David P.},  
booktitle={2021 IEEE International Conference on Big Data (Big Data)},   
title={High-resolution multi-channel weather forecasting – First insights on transfer learning from the Weather4cast Competitions 2021},   
year={2021},  
volume={},  
number={},  
pages={5750-5757},  
doi={10.1109/BigData52589.2021.9672063}
}

@inbook{10.1145/3459637.3482044,
author = {Gruca, Aleksandra and Herruzo, Pedro and R\'{\i}podas, Pilar and Kucik, Andrzej and Briese, Christian and Kopp, Michael K. and Hochreiter, Sepp and Ghamisi, Pedram and Kreil, David P.},
title = {CDCEO'21 - First Workshop on Complex Data Challenges in Earth Observation},
year = {2021},
isbn = {9781450384469},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3459637.3482044},
booktitle = {Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management},
pages = {4878–4879},
numpages = {2}
}
```

