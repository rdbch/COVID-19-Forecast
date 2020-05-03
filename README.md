# ********* WORK IN PROGRESS *****************
# COVID-19-Playground
This repository implements Recurrent Neural Network, which predicts the spread of COVID-19 accross the world. To accomplish this, a model is trained for each country, taking into account its nearest neighbors in terms of growth (later explained).

## Table of contents
1. [Table of contents](#table-of-contents)
2. [Getting started](#setup)
    1. [Installation](#installation)
    
3. [Approach](#approach)
    1. [Country nearest neighbour](#country-nearest-neighbour)  
    2. [Reccurent predictor](#reccurent-predictor)
 
4. [Results](#results)
5. [Disclaimer](#disclaimer)

##Getting Started

### Installation
Clone this repo:
```
$ git clone https://github.com/rdbch/COVID-19-Forecast/
$ cd COVID-19-Forecast 
```

Install PyTorch and 1.0+ and other dependencies (pandas, numpy, seaborn, jupyter, etc).
- pip
```
$ pip install -r requirements.txt 
```

- conda

**Note:** development was done using GPU accelerated version of Pytorch, so they have CUDA dependencies included.  
``` 
$ conda create --name COVID-19-Forecast --file req.txt 
```


## Approach
### Country nearest neighbour

Rather than training a model for every country, it is more suited to train a model for each individual one, using only the nearest neighbours countries in terms of growth. Please check the [this](notebooks/Covid_19_Country_growth_similarity.ipynb) notebook for more details. By doing this, we improve the predictions for the majority of countries. 

Below it is explained how the nearest neighbors of a source country **S** are obtained:
First, we discard the entries(days) which are below a specified threshold (have less than a specified number of cases), for every country(**S** included).
Then, we take a candidate country **C<sub>n</sub>**. A candidate must be more evolved than Romania, meaning it reached the treshold earlier.
We then create a sliding window. Its size is equal to the number of days which are above the treshold in Romania. The window starts sliding over the data from the candidate country, 
beginning with the first day it reached the treshold. For each such iteration, an error is computed. The smallest error will be the error associated with the candidate country.
Next, we choose another country as candidate and repeat the above steps.
After computing the error for each country, the nearest neighbors of Romania will be those countries which have the smallest associated error.

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

