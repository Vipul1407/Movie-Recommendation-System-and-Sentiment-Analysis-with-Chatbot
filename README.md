## Prerequisites
The following open source packages are used in this project:
* Numpy
* Pandas
* Matplotlib
* Flask
* IMDbPY API

## Project Structure 
```
project
│   README.md
│    
└───Data
│   │   Data_README.txt
│
└───Images
|   |   img1.png
|   |   img1.png
|   |   ...
|
└───Model
│   │   Exploratory data analysis with PostgreSQL.ipynb
│   │   Hybrid recommendation algorithm.ipynb
│   │   PostgreSQL_Database_wrapper.py
│   │   Movies_Datase.pkl
│   │   Movies_Learned_Features.pkl
│   │
│   └───images
│       │   img1.png
│       │   img2.png
│       │   ...
│   
└───Server
      │   README.txt
      └───templates
      │     │    home.html
      │     │    layout.html
      │     │    result.html
      └───app.py
      └───Movies_Datase.pkl
      └───Movies_Learned_Features.pkl
      └───requirements.txt
      └───templates
```
## Dataset 
The dataset is provided by GroupLens and can be downloaded from here it contains the following files(links.csv, movies.csv, ratings.csv, and tags.csv)

> "This dataset (ml-latest-small) describes 5-star rating and free-text tagging activity from MovieLens, a movie recommendation service. It contains 100836 ratings and      3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996, and September 24, 2018. This dataset was generated on September    26, 2018."

* This [Dataset](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip) of 1 MB is the one used in all Notebooks 
* This [Dataset](https://files.grouplens.org/datasets/movielens/ml-latest.zip) of 265 MB is the one used for training deployed model. 

## Usage
Setup to run on localhost<br/>
1. Clone the repository
```
git clone https://github.com/Vipul1407/Movie-Recommendation-System-with-Chatbot-and-Sentiment-analysis-.git
cd ~/Hybrid-recommendation-system-web-application
```
2. Create a Python 3.7 environment.
```
  virtualenv .
```
3. Activate Your Virtual Environment for Linux
```
source bin/activate
```
3. for Windows
```
cd Scripts
then
activate
```
4. Install dependencies.
```
   pip install -r requirements.txt
```
5. Run following command from the directory you have stored the file
```
python app.py
```