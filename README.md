# PuzzleFusion
This includes an original implementation ofour paper:
[Puzzlefusion: Unleashing the Power of Diffusion Models for Spatial Puzzle Solving](https://arxiv.org/pdf/2211.13785.pdf), NeurIPS (Spotlight) 2023.



![Model dataset](imgs/dataset2.png)

## Installation
Our implementation builds upon the publicly available  [guided-diffusion](https://github.com/openai/guided-diffusion) repository by OpenAI. To begin, first clone our repository, PuzzleFusion, to your local machine. Following this, you can install the necessary dependencies by executing the commands provided below
```
pip install -r requirements.txt
pip install -e .
```

## Crosscut puzzles

The Cross Cut dataset can be accessed via this [crosscut-data](https://drive.google.com/file/d/1kRRI9V6ro1MK0f-rNbw0hg5jw_WVwlzw/view?usp=share_link), we utilized the original code from the [Crossing cut puzzle Paper code ](https://openaccess.thecvf.com/content/CVPR2021/papers/Harel_Crossing_Cuts_Polygonal_Puzzles_Models_and_Solvers_CVPR_2021_paper.pdf) to generate the data. After downloading the data, please place it within a folder named 'datasets'.  

***For training you can run:***
```
cd scripts
bash script.sh
```
***For testing you can run:***
```
cd scripts
bash script_test.sh
```
We also have provided checkpoint for easier testing [here](https://drive.google.com/file/d/1jdqZFikSXTVDyOBErL0tn373RCcQKV1f/view?usp=share_link), you can download that and move it to ./scripts/ckpts/preds 


## Voronoi puzzles
The Voronoi dataset can be accessed via this [Voronoi-data](https://drive.google.com/file/d/1baKbS7zwA2envoIPfpQYuIxSQVqfW_eO/view?usp=share_link), we utilized the om the [Vornoi puzzle  Generator ](https://github.com/sepidsh/PuzzleFussion/blob/main/Voronoi_samples/vor_dataset_maker.py) to generate the data. After downloading the data, please place it within a folder named 'datasets'.

Samples will be saved in ./scripts/outputs and  model checkpoints will saved in to ./scripts/ckpts. Scripts for Vornoi dataset can be found in ./scripts/voronoi_scripts. You can move the file inside there to main ./scripts file. Similarry Although codes are almost identical we provided puzzle fusion voronoi version code under the fulder puzzle_fusion/puzzle_fusion_voronoi you can replaze the files there with files inside puzzle_fusion. Voronoi data reader is also there. 

After moving files. 
***For training you can run:***
```
cd scripts
bash script.sh
```
***For testing you can run:***
```
cd scripts
bash script_test.sh
```
We also have provided checkpoint for easier testing [here](https://drive.google.com/file/d/1VB_7M6Uodb6eK2DDMprAuZUdT9UaCA4O/view?usp=share_link), you can download that and move it to ./scripts/ckpts/preds



## MagicPlan 

You can download MagicPlan dataset from  [here](https://drive.google.com/file/d/1cu4HQHhCMyezzgUbpFn_5ymqT46xXQdz/view?usp=sharing), dataset follows same license as code.

## Citation

```
@inproceedings{
hosseini2023puzzlefusion,
title={Puzzlefusion: Unleashing the Power of Diffusion Models for Spatial Puzzle Solving},
author={Sepidehsadat Hosseini and Mohammad Amin Shabani and Saghar Irandoust and Yasutaka Furukawa},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=Z764QxwETf}
}
```





***Further details for magicplan dataset/rplan , instructions on code and along with datasets will be published shortly.***
