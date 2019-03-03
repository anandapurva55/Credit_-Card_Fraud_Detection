# Credit Card Fraud Detection

This is a Machine Learning based model developed in Python for the detection of fraudulent credit card's transactions.

## Dataset :-

Source :- https://www.kaggle.com/mlg-ulb/creditcardfraud

The dataset contains 31 parameters and each parameter is having more than 2 lakhs recorded values. The dataset was loaded using the Pandas library.

Following are the parameters :-

    a) Time : Time of transaction
    b) V1-V28 : V1 to V28 are the results of PCA dimensionality reduction that was used to protect sensitive information in this dataset.Like location or name of the user.
    c) Amount : Amount for which transaction was done.
    d) Class : Class 0 represents a fraudaulent transaction and Class 1 represents a genuine transaction.
    
### Description of the parameters 
![screen shot 2019-02-27 at 11 38 51 am](https://user-images.githubusercontent.com/31860248/53469893-b730c980-3a85-11e9-8af6-4269d417bc84.png)
![screen shot 2019-02-27 at 11 50 41 am](https://user-images.githubusercontent.com/31860248/53469968-ff4fec00-3a85-11e9-8378-c89e3bcb30e2.png)

    For saving computational power and time, we take 10% of the entire dataset rather than the whole.

### Histogram of each parameter
    Plotted using Matplotlib.
![screen shot 2019-02-27 at 11 53 33 am](https://user-images.githubusercontent.com/31860248/53470111-a3d22e00-3a86-11e9-9963-93a4892e98c4.png)
![screen shot 2019-02-27 at 11 54 11 am](https://user-images.githubusercontent.com/31860248/53470156-bfd5cf80-3a86-11e9-834b-f3105b726567.png)

### Correlation Matrix

    It tells the linear relationships between attributes so that unwanted ones can be filtered out and was plotted using Pandas, Matplotlib and Seaborn modules.
![screen shot 2019-02-27 at 11 54 45 am](https://user-images.githubusercontent.com/31860248/53470178-d1b77280-3a86-11e9-8d58-1ef40e1d69c4.png)

## Algorithms

    The algorithms used are Local Outlier Factor and Random Forests. Former one achieved efficiency of 99.75% and the later one of 99.65%. They were imported from the Sklearn module.
    
### Final Result

![screen shot 2019-02-27 at 11 55 06 am](https://user-images.githubusercontent.com/31860248/53470179-d419cc80-3a86-11e9-90c4-3c070d39342e.png)

