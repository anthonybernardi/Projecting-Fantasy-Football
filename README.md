# Projecting-Fantasy-Football
project fantasy football points using machine learning

# Overview    
Language Used: Python 3

Packages Used: Pandas, BeautifulSoup, Numpy, Tensorflow, Matplotlib


- Scrape.py scrapes football stats from https://www.pro-football-reference.com and places the data into CSV's

- Analyze.py uses this data to make a neural network
    - Uses a variety of features to train the network
    - Can choose which years to train and test on
    - Plots different graphs as seen below
    - Prints MAPE with different filters as well


# Graph Output

The graphs below show a pretty solid correlation.

Mean Average Error (MAE) of 40 fantasy points sounds like a lot, but that's only 2.5 points per game, which isn't too bad.

Mean Average Percent Error (MAPE) averages around 45%, however it is around 25% when a player's actual points >100

![alt text](https://i.imgur.com/E7L1cmX.png) ![alt text](https://i.imgur.com/y8BdSVe.png) ![alt text](https://i.imgur.com/Tv6tTtb.png)
