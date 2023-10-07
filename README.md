# PuzzleFussion
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

The Cross Cut dataset can be accessed via this [crosscut-data](https://drive.google.com/file/d/1kRRI9V6ro1MK0f-rNbw0hg5jw_WVwlzw/view?usp=share_link),e utilized the original code from the [Crossing cut puzzle Paper code ](https://openaccess.thecvf.com/content/CVPR2021/papers/Harel_Crossing_Cuts_Polygonal_Puzzles_Models_and_Solvers_CVPR_2021_paper.pdf) to generate the data. After downloading the data, please place it within a folder named 'datasets'.  

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
We also have provided checkpoint for easier testing [here](https://drive.google.com/file/d/1jdqZFikSXTVDyOBErL0tn373RCcQKV1f/view?usp=share_link)

Samples will be saved in ./scripts/outputs and  model checkpoints will saved in to ./scripts/ckpts.






***Further details for magicplan dataset/rplan and voronoi, instructions on code and along with datasets will be published shortly.***
