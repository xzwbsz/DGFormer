# DGFormer
The source code of the SOTA Dynamic Graph Reformer (DGFormer) model for weather station prediction.

DGFormer is the update of Dynamic Climate Pattern Graph Recurrent Network (DCPGRN) for meteorological forecasting
This is a official pyTorch implementation of DGFormer

![image](https://user-images.githubusercontent.com/44642002/236671358-363ebabf-0477-4258-97b8-39fd6904a005.png)

![image](https://user-images.githubusercontent.com/44642002/236467448-15e556f8-d9b8-4407-8bb0-8c5373b827eb.png)

![image](https://github.com/xzwbsz/DGFormer/assets/44642002/952995c0-8784-4766-bc2b-7ecbc35b0e23)



## Basic Requirements
torch>=1.7.0

torch-geometric-temporal

Dependency can be installed using the following command:

pip install -r requirements.txt

## Data Preparaion
You can download our data from [google drive](https://blog.csdn.net/zhn11/article/details/128899461?spm=1001.2014.3001.5502), and put it into ./data dir.

Download the dataset and copy it into data/ dir. And Unzip them.

To execute the baseline experiment, you can change the "graph_model", 

## Training the Model
The configuration is set in config.yaml file for training process. Run the following commands to train the target model.

python train.py

We are further developing the distributed version for a larger scale GNN model.

## Experiment Results
We compared our model with STGCN, DCRNN, AGCRN, WGC, Dysat, TGAT, GWN and CLCRN, also with ECMWF (NWP method). The reslut shows that our model outperform others especially in temperature prediction
![image](https://github.com/xzwbsz/DGFormer/assets/44642002/42dfd789-9b2a-4bd5-bf1d-a87159f15950)



## Acknowledgement
The project is developed based on [PM2.5-GNN](https://github.com/shuowang-ai/PM2.5-GNN), [Reformer](https://github.com/google/trax/blob/master/trax/models/reformer/reformer.py) and [IDGL](https://github.com/hugochan/IDGL) for dynamic graph.


