# COVID-19-Playground


## Table of contents
1. [Table of contents](#table-of-contents)
2. [Setup](#setup)
3. [Approach](#approach)
    1. [Country nearest neighbour](#country-nearest-neighbour)  
    2. [Reccurent predictor](#reccurent-predictor)
 
4. [Results](#results)
5. [Disclaimer](#disclaimer)

## Setup
To run this project you will hace to install the 
## Approach
### Country nearest neighbour

Rather than training a model for every country, it is more suited to train a model for each individual one, using only the nearest neighbours countries in terms of growth. Please check the [this](notebooks/Covid_19_Country_growth_similarity.ipynb) notebook for more details. By doing this, we improve the predictions for the majority of countries. 

#### Romania - average disease spread
![romania](assets/images/romania_growth.png)

#### Germany - evolved disease spread
![germany](assets/images/germany_growth.png)

### Reccurent predictor

![RNN_predictor](assets/images/rnn.gif)
A naive model based of reccurent cells us implied. For the exact implementation of the model, please consider taking a look at [basic_recurrent_predictor.py](core/networks/basic_recurrent_predictor.py).

The predictor was only trained on the neareast neighbours. To limit the growth, an unsupervised loss is used for smoothing out the long term prediction.

## Results
### Italy 
 
<img src="assets/images/italy_deaths.png"></img> <img src="assets/images/italy_confirmed.png" ></img>

### Romania

<img src="assets/images/romania_deaths.png"></img> <img src="assets/images/romania_confirmed.png" ></img>

## Disclaimer

