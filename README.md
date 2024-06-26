# Exploring the Chemical Space of Ionic Liquids for CO2 Dissolution through Generative Machine Learning Models
Xiuxian Chen, Guzhong Chen, Kunchi Xie, Jie Cheng, Jiahui Chen, Zhen Song*, and Zhiwen Qi*

State Key laboratory of Chemical Engineering, School of Chemical Engineering, East China University of Science and Technology, 130 Meilong Road, Shanghai 200237, China

Corresponding authors: songz@ecust.edu.cn (Z. S.), zwqi@ecust.edu.cn (Z. Q.)

This is the official implementation of "Exploring the Chemical Space of Ionic Liquids for CO2 Dissolution through Generative Machine Learning Models". In this work, a generative framework combining re-balanced variational autoencoder (VAE), artificial neural network (ANN), and particle swarm optimization (PSO) is developed based on a comprehensive experimental solubility database from literature.

# Getting Started

## Dataset
For the pre-training SMILES database, you can download the original Pubchem coumpound database at ftp://p.ncbi.nlm.nih.gov/pubchem/Compound/. the CO2 solubility dataset is in the Modeldata/CO2ModelData.xlsx

## Training
The RB-VAE can be pretrained, fine tuned and saved in RB-VAE/__main__.py by changing the corresponding arguments. After training VAE, the FNN can be trained and tested in FNN&PSO/MLPTraining.py. 

## Screening & Designing
All the screening and designing functions within the FNN&PSO/MLPTraining.py can be processed and customly modified. For calculating the SA score, please uses the RDKit-based implementation in https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score by Peter Ertl and Greg Landrum.
