
# CS 4641 Machine Learning Final Report

Alfredo Gateno, Alon Baruch, Matthew Grant, Rishit Mohan Ahuja, Shaun Sharma

# Video Link: https://youtu.be/0Veir3Y800A


# Intro/Background

Airbnb is a service through which property owners can rent out their property to anyone looking to stay there. Most properties are rented for a short time such as a few nights and property prices are usually charged per night. Our project will be to build a model which can predict the price per night of an Airbnb listing using different datasets described in Airbnb Data Table Schema.md


# Problem Definition

The problem we are trying to solve is: given a set of features for an Airbnb listing, such as location, number of bedrooms & bathrooms, etc., how should a property owner price their listing? Using our model, property owners would be able to input these features and get an output price that aligns with past Airbnb listings with similar features without having to go through all of the research to find them. This problem is important since real estate owners struggle to get their listing prices compared to their competitors, based on similarities such as location, number of bedrooms/bathrooms, amenities, etc. With our model, we are giving an opportunity to the owner to base its pricing strategies and manage its listings, while being accurate enough to the market demands.


# Data Collection

We obtained the data from Citadel’s Datathon Summer 2020. We planned to primarily use three datasets: listings.csv, demographics.csv and real_estate.csv for predicting the price. Please look at the ‘Airbnb Data Table Schema.md’ to know all these datasets’ features. Listings.csv has 50321 rows and 29 columns, demographics.csv has 33120 rows and 27 columns and real_estate.csv has 29111 rows and 262 columns. Furthermore, there are three other datasets we had access to: venues.csv, econ_state.csv, and calendar.csv. As mentioned above, full details of these data sets are available at ‘Airbnb Data Table Schema.md’.


# Methods  



1. Data manipulation & cleaning
2. EDA 
3. Feature selection and engineering 
4. Model Creation
5. Evaluation 


## Data Manipulation & Cleaning


### Merging Datasets

Our initial plans were to merge listings, demographics and real-estate into one dataframe using the zip code feature. However we discovered that almost all the features in real_estate had about 55% null values. We could have filled the data with some standard techniques such as mean, median, etc. or by using regression but there were two main problems in that approach: (a) We would have had to do it for 262 features. (b) Adding so many data points can significantly bias the data and lead to inaccurate predictions. Therefore, we removed real_estate.csv and merged only listings.csv and demographics.csv.

We checked null values for demographics and listings as well and found that demographics had no null values and whereas listings had several features with null values. Below is a visual showing which features in listings.csv had the most null values.



![Picture1](https://user-images.githubusercontent.com/68388723/145147827-417c0b4e-e723-4fd5-8ba8-9f07c3c38b01.png)


 Specifically we saw that:



* bathrooms 0.29 % missing values
* bedrooms 0.08 % missing values
* beds 0.17 % missing values
* has_availability 100.0 % missing values
* name 0.02 % missing values
* review_scores_checkin 25.27 % missing values
* review_scores_cleanliness 25.17 % missing values
* review_scores_communication 25.17 % missing values
* review_scores_location 25.27 % missing values
* review_scores_rating 25.06 % missing values
* review_scores_value 25.3 % missing values
* weekly_price 77.01 % missing values

Since has_availability and weekly_price are largely null, we remove these features. Roughly 25% of review_scores_checkin, review_scores_cleanliness, review_scores_communication, review_scores_location, review_scores_rating and review_scores_value has null values. Since 25% of the data in these features is null, we had to choose between the two options: (a) removing these features (b) removing the rows corresponding to which there are null values in these columns.

We chose the (b) option due to the nature of the problem we are solving. These features are very important for predicting the price of the Airbnb listing and we should prefer keeping these features even if that means that we have to remove a lot of rows particularly because we have a lot of data. After removing the rows containing null, we ended up with 36972 rows.


### Working with Different Data Types

Now that we have removed the majority of the null values from our data set we can go through and ensure that all the features are in the correct data type. 

Some features that should be of the numerical type but were as string type, maybe because of the way they were inputted were: ‘price’, ‘$10,000-$14,999’, ‘$15,000-$24,999’, ‘$25,000-$34,999’, ‘$35,000-$49,999’, ‘$50,000-$64,999’, ‘$65,000-$74,999’, ‘$75,000-$99,999’, ‘$100,000_or_more’, ‘median_household_income’, ‘Mean_household_income’.

These were important features and therefore could not be removed. We converted each of them to numerical form. Some of the features had some random characters like ‘-‘, ‘:’ in them which led to errors while converting to the numeric type. Therefore, we used

df[num_cols].apply(pd.to_numeric, errors=’coerce’)

and individually removed the Nan values present returned by the above code. We had to remove only a few rows in the dataset but the conversion of these features to numeric type added a lot of value to our dataset

Now we had all data that should be numeric as numeric, our only problem was converting data that was categorical. 


### Dealing with Categorical Data

The first thing we did was to remove identifying features from the dataset. These included: Host ID, Listing ID, Listing Name.  Then we removed features that are naturally redundant such as Latitude and Longitude, which are unnecessary since we already have a ‘city’ feature as well as features that are not relevant to future predictions such as ‘Availability 30’ 

We also removed features that would be impractical to categorize such as ‘Amenities’ since there are too many amenities and many of them can be inputted custom to a listing. After making those changes to the dataset, we used panda’s get_dummies method to one-hot encode the remaining categorical features. This took our dataset from 46 features to 100 features. The dataset up to this point will be called our “Baseline” dataset. We will discuss further dealings with categorical data in the 
Engineering section of the report.  


## EDA

Our baseline dataset has 65 features and 36,971 rows. The average nightly price for a listing in our dataset is $168.27 with a median price of $110 and a maximum price of $10,000 a night. Due to this high range in prices, our price feature for listings has a standard deviation of $233.25. 

Our baseline dataset is centered around 5 metropolitan areas: Asheville, Austin, Nashville, New Orleans, and Los Angeles. Below is a map showing the geographical distribution of Listings in each area



![Picture2](https://user-images.githubusercontent.com/68388723/145147830-fb9d2731-08fc-4a79-8097-90950dda4eff.png)


![Picture3](https://user-images.githubusercontent.com/68388723/145147831-03a6687c-c7cd-46dc-bb3d-c7e8902241b9.png)


![Picture4](https://user-images.githubusercontent.com/68388723/145147834-964c95b5-9fff-4a5f-b870-cf06e1aa3c06.png)


![Picture5](https://user-images.githubusercontent.com/68388723/145147837-bbc2209b-7085-4aa5-a5a7-04cdea4ae520.png)


![Picture6](https://user-images.githubusercontent.com/68388723/145147838-4b0212b0-ae00-4ea9-8397-361f7a2505c0.png)


One of the other datasets we had, calendar, contains over 18,383,956 rows where each row contains an Airbnb listing ID, the date, whether the listing was available on that date, the price, and the metro_area the listing was located at. We wanted to see if we could use the average availability of a listing as one of our features so we plotted the average availability versus the average price of a listing, but found little correlation.

![Picture7](https://user-images.githubusercontent.com/68388723/145147839-619f4925-0ad4-4f2d-b91e-d5484ce437a0.png)


We checked how many people usually live in an Airbnb accommodation and it turned out that most of the accommodations are two people. Therefore, we based our following analysis on only two-people accommodations.

![Picture8](https://user-images.githubusercontent.com/68388723/145147841-e3b640d6-2381-4d5f-9c3b-fc5e49574880.png)


Next, we plotted the average price (which is basically our main target) vs the metropolitan areas for 2-people accommodations. This shows that the average price of a 2 person rental is consistent across all the metropolitan areas.

![Picture9](https://user-images.githubusercontent.com/68388723/145147842-f074982c-74ee-4fa2-ae43-3dbd85c81e0c.png)


## Feature Selection & Engineering

Once we had our dataset completely numeric by one-hot encoding the categories, we started determining which features would help improve our model the best. First we checked the variance of all the features from our encoded dataset. Here is a pie chart showing which feature made up the largest section of the 50 lowest variance columns.


![Picture10](https://user-images.githubusercontent.com/68388723/145147843-c367c91c-6708-4aaf-a9bd-87aea382c2fb.png)

### Combining Categorical Data

From this we can see that Property Type columns make up 68% of the bottom 50 features by variance. This is not incredibly surprising seeing as there are 36 different types of properties leading to 36 different columns being added to our data set. One way to address the low variance in property types is to remove all or just some of the property type columns from the dataset however we believed that property type would be relevant for our regression and we did not think it was wise to remove some property types while keeping others. So instead, we combined the 36 types of properties into 6 types detailed below



* The group House added the groups: ‘Villa', 'Townhouse', 'Bungalow', 'Cabin', 'Chalet', 'Vacation home'
* The group Apartment added the groups: 'Serviced apartment', 'Condominium', 'Loft', 'Dorm', 'Entire Floor'
* The group Tent added the groups: 'Tipi', 'Yurt', 'Hut'
* The group Guest House added the groups: 'Guesthouse', 'Guest suite', 'In-law'
* The group Hotel added the groups: 'Bed & Breakfast', 'Boutique hotel', 'Hostel'
* All other property types were combined under the grouping “Other”

Next up we saw that the Cancellation Policy had 6 categories: Flexible, Moderate, Strict, No Refunds, Super Strict 30, and Super Strict 60. Since the last 3 of these categories were all in the low variance list and had very few properties that fall under these categories, we decided to combine No Refunds, Super Strict 30, and Super Strict 60 into a new category “Super Strict”. 


### Removing Features with Low Variance

Another thing we noticed is that all 5 of the bed type categories were in the low variance list. All the bed types other than ‘Real Bed’ had a variance of below 0.01 and ‘Real Bed’ had a variance of 0.02 so we decided to remove Bed Type as a feature from our dataset. 


### Removing Inaccurate Data

After addressing features with low variance we began to examine the data provided by demographics.csv. Unfortunately what we discovered was that the data for income distributions for each zip code was inaccurate. The data was supposed to show what percentage of households in each zip code fall under which income group: $9,999_or_less', '$10,000-$14,999', '$15,000-$24,999', '$25,000-$34,999', '$35,000-$49,999', '$50,000-$64,999', '$65,000-$74,999', '$75,000-$99,999', and '$100,000_or_more'. However some of these columns ended up being percentages while other ended up being integers representing the number of households, not the percentage and the sum of percentages were greater than 100% for most zip codes. For these reasons we decided that the income distribution data was inaccurate and we removed those features from our dataset, keeping only mean and median household income for each zip code. 


### Adding Custom Features

While performing EDA on the calendar.csv dataset, we were able to calculate average availability for each listing. We believed this may be useful for our regression model so we added average availability as a feature to each item in our dataset. 


### Using Correlation to Remove Redundant Values

We know mapped the correlation of our remaining features:


![Picture11](https://user-images.githubusercontent.com/68388723/145147844-b6ca39ea-bb55-41dc-b21b-89fc23bc2100.png)



As can be seen by the heat map above, the population groupings have high correlation with one another and with the overall population of the town. The reason for this is because each feature holds the number of individuals in that age group so zip codes with large populations will have large values for each of these features. To address this we combined the 13 age groups into 4 age groups: >= 19, 20-44, 45-64, 65+ and converted the groups from integer values indicating overall population of that group to percentages relative to overall population of the zip code. 

We can see the impact this change had on our data below:

![Picture12](https://user-images.githubusercontent.com/68388723/145147845-8cfbf7c8-17b3-41e5-a391-da2304275e50.png)


We also see that each state has 100% correlation to its metropolitain and that is due to the nature of the data, we only have information about listings in those specific cities. For this reason we will remove the State features from our dataset.

We also see that ‘instant_bookable_t’ and ‘instant_bookable_f’ have an exact correlation. This is because instant bookable is a boolean value that was encoded as ‘t’ or ‘f’. We will remove ‘instant_bookable_f’ so that we can use ‘instant_bookable_t’ as our indicator for that feature.

The next step we took was to grab the features with the highest pairwise correlations. They are listed below


<table>
  <tr>
   <td>median_household_income
   </td>
   <td>mean_household_income 
   </td>
   <td>0.945605
   </td>
  </tr>
  <tr>
   <td>room_type_Entire home/apt
   </td>
   <td>room_type_Private room
   </td>
   <td>0.931779 
   </td>
  </tr>
  <tr>
   <td>property_type_Apartment
   </td>
   <td>property_type_House
   </td>
   <td>0.908343
   </td>
  </tr>
  <tr>
   <td>accommodates
   </td>
   <td>beds
   </td>
   <td>0.819863
   </td>
  </tr>
  <tr>
   <td>accommodates
   </td>
   <td>bedrooms
   </td>
   <td>0.778329
   </td>
  </tr>
  <tr>
   <td>20-44_years
   </td>
   <td>45-64_years
   </td>
   <td>0.750004
   </td>
  </tr>
  <tr>
   <td>review_scores_rating
   </td>
   <td>review_scores_value
   </td>
   <td>0.746757
   </td>
  </tr>
  <tr>
   <td>bedrooms
   </td>
   <td>beds
   </td>
   <td>0.723534
   </td>
  </tr>
  <tr>
   <td>20-44_years
   </td>
   <td>64_or_more
   </td>
   <td>0.719935
   </td>
  </tr>
  <tr>
   <td>review_scores_cleanliness
   </td>
   <td>review_scores_rating
   </td>
   <td>0.718286
   </td>
  </tr>
  <tr>
   <td>bathrooms
   </td>
   <td>bedrooms
   </td>
   <td>0.689374
   </td>
  </tr>
  <tr>
   <td>review_scores_checkin
   </td>
   <td>review_scores_communication
   </td>
   <td>0.687180
   </td>
  </tr>
  <tr>
   <td>review_scores_communication
   </td>
   <td>review_scores_rating
   </td>
   <td>0.637545
   </td>
  </tr>
  <tr>
   <td>cancellation_policy_moderate
   </td>
   <td>cancellation_policy_strict
   </td>
   <td>0.627802
   </td>
  </tr>
  <tr>
   <td>population
   </td>
   <td>households
   </td>
   <td>0.623831
   </td>
  </tr>
  <tr>
   <td>review_scores_cleanliness
   </td>
   <td>review_scores_value
   </td>
   <td>0.620583
   </td>
  </tr>
  <tr>
   <td>bathrooms
   </td>
   <td>beds
   </td>
   <td>0.599738
   </td>
  </tr>
  <tr>
   <td>review_scores_checkin
   </td>
   <td>review_scores_rating
   </td>
   <td>0.599580
   </td>
  </tr>
  <tr>
   <td>accommodates
   </td>
   <td>bathrooms
   </td>
   <td>0.597033
   </td>
  </tr>
</table>


We notice several features that can be removed due to high Multicollinearity. The first pair to examine is mean household income and median household income. We will remove mean income in favor of median income since we know that the income distribution values are skewed towards high income. Next we will remove the beds feature since it is highly correlated to both bedrooms and accommodates. We will also remove the different categories of the review scores (cleanliness, checkin, communication, value, location) in favor of the compiled review scores rating. Lastly we will remove ‘room_type_Private room’ in favor of ‘room_type_Entire home/apt’ since these two categories have a very high correlation, we see that if a room is not an entire home/ apartment it is likely to be a private room. 

That is all of the feature selection we have completed. At the end we transformed our 52 feature dataset into a 32 feature dataset which we will further reduce by conducting PCA as part of our model pipeline. We refer to this new dataset as the ‘improved’ dataset in the following sections.


## Model Creation


### We trained the models on two datasets: 



1. The baseline dataset: in this dataset all the features are converted to numeric type using some of the methods discussed in the data cleaning and data manipulation section. \

2. The improved dataset: besides the preprocessing done on the baseline dataset, we manually selected some of the features (discussed in the feature selection section) and over that we used PCA to further reduce the number of features.


### We used two models to for predictions:



1. Sklearn’s Linear Regression Model that conducts an ordinary least squares regression
2. XGBoost Regressor, an ensemble model that uses gradient boosting to fit a regression model

Why did we use XGBoost?  \
 \
It is a boosting technique that tries to significantly decrease the bias while slightly increasing the variance in prediction. 

Some of the important hyperparameters for XGBoost are the following:

_n_estimators_: the number of models used in our ensemble model. Test values: [1 - 21]

_max_depth:_ it is the maximum depth of the decision trees that we will be using. Test values: [3, 5, 7, 9]

_learning_rate:_ this hyperparameter decides how much weight should be given to the new decision tree whose prediction we are going to use. Unlike random forest, we manually assign a value to the learning rate for XGBoost. Test Values: [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]

_minimum_child_weight:_ minimum sum of weights of all observations required in a child. Test Values: [1, 3, 5]

We will use Sklearn’s GridSearchCV method to test the ranges for each of these hyperparameters. We will discuss the results of our models and tuning of our hyperparameters in the Results and Discussion section.


# Results/Evaluation & Discussion


## Evaluation Metrics


### Normalized RMSE

For the purpose of evaluation, we made a new evaluation matrix called normalized_RMSE that is basically RMSE/(y_mean)(we were motivated by one of the homework problems to implement this idea).

Why did we choose the normalized_RMSE matrix?

RMSE does not give a good estimate of how well the model predicted. If we are working on a dataset that has very small values for all the features then the RMSE will also be very small even if the model made inaccurate predictions. Therefore, to take into account the nature of the dataset, we divide RMSE by the mean of the y-values. The closer our normalized_RMSE gets to 0, the better our model is performing


### R-Squared

R-Squared is a measure of the variation in the dependent variable that is predictable from the independent variables. The closer R-Squared gets to 1, the better our model is performing.

Results table:


<table>
  <tr>
   <td>
   </td>
   <td><strong>normalized_RMSE</strong>
   </td>
   <td><strong>r-squared</strong>
   </td>
  </tr>
  <tr>
   <td><strong>XGBoost on baseline data</strong>
   </td>
   <td>0.023
   </td>
   <td>0.43
   </td>
  </tr>
  <tr>
   <td><strong>XGBoost on improved data</strong>
   </td>
   <td>0.022
   </td>
   <td>0.36
   </td>
  </tr>
  <tr>
   <td><strong>Linear Regression on baseline data</strong>
   </td>
   <td>0.024
   </td>
   <td>0.39
   </td>
  </tr>
  <tr>
   <td><strong>Linear Regression on improved data</strong>
   </td>
   <td>0.024
   </td>
   <td>0.37
   </td>
  </tr>
</table>



## Hyperparameter Tuning

We performed cross-validation using sklearn’s gridsearch and the following testing ranges for each hyperparameter in the **baseline dataset**:



* [1,21] for the number of estimators
* [3,5,7,9] for the max depth
* [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3] for the learning rate
* [1, 3, 5] for the minimum child weight

The best results for the baseline dataset were achieved by a learning rate of 0.2, a max depth of 7, a minimum child weight of 5, and 19 estimators.

We also performed cross-validation using sklearn’s gridsearch and the following testing ranges for each hyperparameter in the **improved data** set:



* [1,21] for the number of estimators
* [3,5,7,9] for the max depth
* [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3] for the learning rate
* [1, 3, 5] for the minimum child weight

The best results for the improved dataset were achieved by a learning rate of 0.2, a max depth of 5, a minimum child weight of 3, and 15 estimators.

After XGBoost Hyperparameter Tuning:


<table>
  <tr>
   <td>
   </td>
   <td><strong>normalized_RMSE</strong>
   </td>
   <td><strong>r-squared</strong>
   </td>
  </tr>
  <tr>
   <td><strong>XGBoost on baseline data</strong>
   </td>
   <td>0.022
   </td>
   <td>0.50
   </td>
  </tr>
  <tr>
   <td><strong>XGBoost on improved data</strong>
   </td>
   <td>0.023
   </td>
   <td>0.37
   </td>
  </tr>
</table>



## 

![Picture13](https://user-images.githubusercontent.com/68388723/145147847-1dc18b8d-5f96-41f1-9b13-6f4a77d6e18a.png)


## Reasons for Error

One of the first reasons we can think of for error is that about 1.5% of our listings have outlier prices (> $868 per night). While 1.5% might not sound like a lot, it is significantly higher than the 0.3% of outliers we can expect from a standard Gaussian Distribution. Since the Sklearn Linear Regression model uses an ordinary least squares approach it is disproportionately affected by outliers. Furthermore, ordinary least squares assume that all features are linearly independent. As was seen during feature selection and EDA, there is strong collinearity between a few of the remaining features in the ‘improved’ dataset and these features are even more numerous in the ‘baseline’ dataset. 

Another reason that our models performed better with the ‘baseline’ dataset than with the ‘improved’ dataset could be due to our feature selection process. One specific part to mention could be the removal of ‘‘room_type_Private room’ since it was so highly correlated to ‘‘room_type_Entire home/apt’, however there is a third option for room types which is ‘shared room’. We removed one category of room types and not the other two which most likely had an effect on our model. Lastly we were more selective with which features get removed for high correlation and which do not. An example of features with high correlation to each other that stayed in the dataset include: ‘Accommodates’, ‘Bedrooms’ & ‘Bathrooms’. This problem is compounded by the fact that all these features are not only highly correlated with each other but are also highly correlated with price which is the feature we are trying to predict. It is possible that listings with high values for ‘Accommodates’ naturally have a high value for ‘Bedrooms’ and for ‘Bathrooms’ and all of these have an effect on price so listings with high values for ‘Accommodates’ will lead to an estimated price that is larger than their real price. 

Sources for error in the XGBoost Regressor involve the hyperparameters that were chosen and the type of data that was inputted. First, the range of testing for our hyperparameters was not large so it is possible we have found a local minimum instead of a more optimal combination. One example is that the best minimum child weight returned to us for the baseline dataset was 5 which is at the top of the range we tested. It is possible that a minimum child weight of 7 would be more optimal however we did not check those values. A second source of error for our XGBoost regressor has to do with the encoding of our categorical data. XGBoost may assume that there is an ordinal relationship between our integer values however since we have many categorical features there is no such relationship between features like: ‘property_type_house’, ‘instant_bookable_t’, and ‘metropolitain_austin’. 


# Conclusion

Throughout this project, we learned how to apply a breath of EDA and supervised learning methods to real word data. Our results allow for the average AirBnB lister to find a suitable price for their listing that is neither too high or too low given a set of features and showcase the power of regression models. While this was not a newly explored problem, we believe our model to be more robust compared to other price prediction models due to the amount of features and data we included.

While we were able to take our initial given dataset and use it to create functioning models, our process involved an extensive combination of machine learning techniques to format data and tune predictive models to achieve a result. As is the case in many machine learning projects, our initial data set had to be cleaned, encoded, and modified to better fit our purposes as well as be understood by the machine learning model. We additionally spent time researching which kinds of models would potentially provide the best accuracy given the type of data we were using. Additionally we spent time hyper-parameter tuning to overcome issues with overfitting and underfitting to increase the accuracy of our predictions. Through the use of these techniques, we were successfully able to apply concepts from throughout the semester to produce a useful model in predicting Airbnb listing prices. 


# References



* [https://arxiv.org/pdf/1907.12665.pdf](https://arxiv.org/pdf/1907.12665.pdf) - AirBnB prediction using Machine Learning and Sentiment Analysis. (K-Means clustering and Linear Regression)
* [http://cs229.stanford.edu/proj2016/report/WuYu_HousingPrice_report.pdf](http://cs229.stanford.edu/proj2016/report/WuYu_HousingPrice_report.pdf) - Real Estate Price Precition with Regression and Classification
* [http://rl.cs.mcgill.ca/comp598/fall2014/comp598_submission_99.pdf](http://rl.cs.mcgill.ca/comp598/fall2014/comp598_submission_99.pdf) - Prediction of real estate property prices in Montreal
* [https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f](https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f) - Hyperparameter Tuning of XGBoost Regressor
