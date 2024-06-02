# Description
Code for ICML 2024 paper 

**Learning from Streaming Data when Users Choose**

Jinyan Su, Sarah Dean

# Instructions

Create directory ```./dataset``` and ```results```.

## Data preprocess

First, download MovieLens 10M Dataset into to the ```./dataset``` directory and unzip it. 

Then, run 
```
python movieLens_data_preparation.py
``` 
to preprocess the MovieLens 10M dataset


## MovieLens 10M dataset result

We use the  $n=3$(number of service providers), and $\eta_c=1$ and $\zeta=0, 0.2, 0.5, 0.8, 1$, run the following python file to store the result in ```./result``` file
```
python movieLens.py
```


## Census Data result
```
python census.py
```

After storing the results in the ```./results``` directory, the plot of the convergence of the loss and the iterates are shown in ```plot.ipynb```.
