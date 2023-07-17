We can break the work done on that project into 3 main pieces:

- Dataset.<br>
- Centralized Models.<br>
- Federated models.<be>

<h2>Dataset</h2>:
  
Any good machine learning work always starts with a hunt for the perfect dataset.

Choosing a dataset for such a project is a tough task because there are already very few available datasets publicly, and most of those datasets have their features anonymized or PCA applied to them. 

In a regular machine learning project not knowing what the features are is kinda a small problem or even no problem at all, but In this project where the model will be part of an application, We have to know what the features we’re going to feed that model After all, I can’t just tell the backend to send my model an API that contains information that neither of us knows where it did come from.

We found that one of the solutions for such a problem is using a synthesised dataset.

synthesised datasets are not produced by a  real-world application, instead, this type of data is artificially generated, and most of the time. 
There are not so many synthesised datasets that suit our application though, we are trying to build a bank so we have to pick a dataset that looks like it came from a bank, or at least alter the one that we get to look like it’s a dataset that came from a bank.

We first picked the PaySim dataset, It’s kinda famous, and even though it got 6 million records which is way too heavy on our computational power, we were ready to go with it, but then we discovered a data leakage on the dataset: the transactions which are detected as fraud are cancelled, so for fraud detection, 2 columns must not be used, was that warning written on the Data card for this dataset? No, It was buried in the discussion that started by a random user discovering that rule, most people weren’t aware of that information, and a lot of the papers that used this dataset didn’t mention the data pre-processing details or the names of the columns they used, so we wouldn't be able to trust their results on this dataset, or able to compare our work to them fairly, But sadly we discovered that after doing some work on this dataset, so even though it’s not a perfect dataset, we still gonna take you in an adventure in it. 

dataset link: https://www.kaggle.com/datasets/ealaxi/paysim1

The next one that we found was the Credit Card Transactions from IBM, it’s also a synthesised dataset, but this time instead of 6 million records, it’s 24 million records!

6 million records were already too heavy for our computational power so you can guess by now that a 24-million-records dataset is not a good choice.

Here’s a link for this dataset: https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions

But we decided to make it good though, we chose only the online transactions and decided to focus on them, that dataset contained only 2 million online transactions, and those 2 million transactions had 60% of the entire fraudulent transactions of the IBM dataset, so it was kinda convenient for us.

In the case of the online transaction, the columns:(Merchant City, Merchant State, Zip, Use Chip) are always the same value all over the column, which isn’t a thing that any machine learning model can learn from, so we removed them to get a lighter dataset by 4 whole columns, you may think now that we probably got worst results than those who used the original dataset got if that’s your guess then you’re wrong!

I was also curious to answer some question about this dataset which: does the fraud number grows over the years? also if that’s the case does that relationship differ if we only looked at the online transactions? And I figured out that the fraud doesn’t increase each year, instead, it got a weird relationship with the year advancement as it goes up and down, also which happens in the online transaction-only dataset, just in case you thought that choosing to use the online transaction only messed up that lovely dataset!

Also in the online-only transaction dataset, there was no fraud before 2002 or after 2018! So I decided to chop those years off, other years in the middle got no fraud but I didn’t chop them off, only trimmed the edges!
      
With those chops and cuts, we got a final dataset that .7% of it is fraud which means less than 1% but if you think that’s highly imbalanced, then you should know that the original dataset got only .12% of its full transactions as fraud, also you can wait till you hear of our next monster.

But you know, there is always something bad with a dataset, perfect datasets only exist in heaven, and the only wrong thing that we could discover with this dataset was the fact that it’s not very popular, so we didn’t have a big bool to compare with, and no one - as far as we know- have applied federated learning to this dataset, so we decided to use another dataset beside that one: European credit card transactions, that dataset is fully anonymised except for 2 features of 30 features. here's a link for this dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

It got a cute size compared to the monsters that we mentioned above with only 284,807 transactions and only 492 of those transactions are fraud which means only 0.172% of all transactions are fraud.
It didn’t need any kind of feature engineering though, only removed the time column as it is useless, and normalized the dataset. 

<h2>Centralized Models</h2>

We have experimented with multiple networks and models through our work on this project, our goal was to catch as much fraud as we could while at the same time avoiding false fraud reports as it can be really annoying for customers.<br>

The main obstacle that we faced is the highly imbalanced nature of the publicly available datasets that we can use in our project, there are multiple ways to fight against the imbalanced dataset issues and the way that we chose was through choosing the right algorithms.

We chose mainly 3 metrics to evaluate the model: ROC_AUC, Accuracy and Recall, it's not common to use accuracy as a metric when it comes to dealing with imbalanced datasets but we found that some models can give very high Recall -almost 99%- and bad accuracy, those models are just extremely biased to the fraud class, it's like that the model learned that saying fraud is always the right answer instead of learning how to actually differentiate between the fraud and no-fraud!
