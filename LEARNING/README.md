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

But you know, there is always something bad with a dataset, perfect datasets only exist in heaven, and the only wrong thing that we could discover with this dataset was the fact that it’s not very popular, so we didn’t have a big bool to compare with, and no one - as far as we know- have applied federated learning to this dataset, so we decided to use another dataset beside that one: European credit card transactions (ECC dataset), that dataset is fully anonymised except for 2 features of 30 features. here's a link for this dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

It got a cute size compared to the monsters that we mentioned above with only 284,807 transactions and only 492 of those transactions are fraud which means only 0.172% of all transactions are fraud.
It didn’t need any kind of feature engineering though, only removed the time column as it is useless, and normalized the dataset. 

<h2>Centralized Models</h2>

We have experimented with multiple networks and models through our work on this project, our goal was to catch as much fraud as we could while at the same time avoiding false fraud reports as it can be really annoying for customers.<br>

The main obstacle that we faced was the highly imbalanced nature of the publicly available datasets that we can use in our project, there are multiple ways to fight against the imbalanced dataset issues and the way that we chose was through choosing the right algorithms.

We experimented with three main methods:<br>
1- Few-Shot Learning.<br>
2- Feed Forward Network with different loss functions.<br>
3- Tree-based models.<br>

<h3>1- Few-Shot Learning:</h3>

Few-Shot Learning is considered a kind of meta-learning, FSL was first introduced to solve problems in the domain of image classification, using only a few images from each class, it can construct a classifier that’s able to classify those classes from each other.<br>
The paper that inspired us to follow this method - in other words the paper that started the curse - is Federated Meta-Learning for Fraudulent Credit Card Detection by Wenbo Zheng et al, that paper got the most impressive results that we have ever seen through our search, But it also 2 problems:
The paper lacked a lot of details about the modelling and the hyper-parameters for their models were never mentioned of course, they didn’t share their code, so we were never able to re-create their results even after spending months trying to.<br>
The general structure of the model that this paper described was really computationally expensive, enough to say that they used ResNet 34 with a k-tuplet loss for 1000 epochs just to get the representation of the data! Given our computational power situation, trying to re-create their results on the gigantic datasets that we got was a suicidal mission.<br>
We tried to communicate with the authors of this paper to get a lead on the hyper-parameters that they used in their model, or how they managed to code their loss function in an efficient way but we got no responses at all.
The best that we managed to do using few-shot learning was with the PaySim dataset we later cancelled using it, but we see that the effort that we did building that model worth mentioning anyway.<br>

The model architecture that we used to build the few-shot model for the PaySim dataset was constant of mainly 2 components: Prototypical Networks and a pre-trained ResNet18, Prototypical Networks is an algorithm that was introduced by Snell et al. in 2017 in their paper that’s called “Prototypical Networks for Few-shot Learning”.<br>
The training settings that we used to train our network is called episodic training, it simply depends on feeding the model a batch that got n numbers of samples of each class.<br>
In our specific case, we got only 2 classes, and we chose to feed our model a support set that consisted of 20 samples of frauds and 20 samples of non-frauds, and another query set that consisted of 20 samples of frauds and 20 samples of non-frauds. 
Both The query samples and support samples get passed through ResNet18.<br>
The prototype of the fraud class is produced by taking the mean of the outputs that result from passing the fraud samples in the support set through ResNet18, Same thing is for the non-fraud prototype.<br>
Right after that, we compute the Euclidean distance between the query samples and the support samples and based on the resulting score the model decides whether that query sample is fraud or not.<br>
We used Cross Entropy as a loss function for this model and we used Adam as an optimizer with a learning rate of .01. 
We run our model for only 100 epochs.<br>
That model tends to actually be biased in favour of the fraud class!  we can see that clearly in the results below as we got  F1_score = 66%, ROC_AUC = %93.7, average_precision = %46.6, Precision = %53 and Recall = %87.7, and the figure below shows the confusion matrix of that model:<br>

<img src="https://github.com/Aml-Hassan-Abd-El-hamid/Fraud_Detection_using_Federated_Learning_AND_bank_application/blob/main/LEARNING/readme%20images/Screenshot%20from%202023-07-29%2002-58-43.png" width="340" height="340" >

<h3>2- Feed Forward Network with different loss functions.</h3>

For our second adventure, I decided to go for a simpler and lighter solution, I decided to go with a simple feed-forward neural network, that network alone wouldn’t stand in the face of an imbalanced dataset and win the battle after all our network needs a weapon to stand in the face of the monster and weapon, I chose is an algorithm level weapon, I chose to experiment with the loss function.<br>
Our main network’s structure is constructed mainly of 6 linear layers, each one of those layers is followed by a Batch Normalization and a leaky_relu activation function, except for the last layer which is the output layer, that layer changes according to the loss function in use.<br>
we experimented with more than one loss function using mainly the ECC dataset as its size was very suitable for our computational power, you can find the code to experiment with the losses and the network on the train "FFnetwork using different losses.py" file.<br>
Here I share a little info about the losses functions that we used alongside some of the results of our experiments using each loss function:<br>

1- BCEWithLogits (binary cross-entropy with logits):

Cross entropy is one of the most famous loss functions in the history of deep learning, it’s the go-to function when dealing with classification tasks. 
But the main problem with highly imbalanced datasets is the fact that most machine learning algorithms are not actually designed to deal with such a monster, So we had to use an altered version of the classic cross entropy to get some respectful results.<br>
Binary cross-entropy with logits loss function from Pytorch combines a Sigmoid layer and the Binary cross-entropy loss in one single class, that small change is supposed to make the binary cross_entropy with logits more stable than a sigmoid activation function followed by a Binary cross-entropy loss.

This version of cross entropy is designed to deal only with binary classification and that’s where the Binary part comes in, another different thing that we did to make this function robust against the curse of our highly imbalanced dataset is that we had to add weight to the class in focus, which helps with directing the focus of our machine learning model to the class that needs extra attention, in our case that class is fraud transaction.

By setting the weight of the fraud transaction to 5, we trained our network for 50 epochs using an SGD optimizer with a learning rate of .001 and a momentum of .9. And we got F1_score = 80%, ROC_AUC = %90, average_precision = %66, Precision = %79 and Recall = %80 and the figure below shows the confusion matrix:<br>

<img src="https://github.com/Aml-Hassan-Abd-El-hamid/Fraud_Detection_using_Federated_Learning_AND_bank_application/blob/main/LEARNING/readme%20images/Screenshot%20from%202023-08-01%2003-59-30.png" width="340" height="340" >

2- Roc-star: 

That was one of the coolest loss functions that we ever heard of and we actually enjoyed learning about it a lot.
The story of the loss functions started in 2003 with the paper: Approximates the Area Under Curve score, using an approximation based on the Wilcoxon-Mann-Whitney U statistic. Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R.

The main problem with trying to come up with a ROC_AUC function is the fact that this function isn’t differentiable as ROC_AUC is actually a smooth curve, the fact that a certain function isn’t differentiable means that we won’t be able to get its gradient so we simply won’t be able to know how to steer the wheel of our neural network model.

The paper from 2003 introduced a way to make the ROC_AUC differentiable but there was only one problem with that function, it wasn’t working in practice! - as most papers do -<br>
But almost 10 years after that paper there was a nice engineer who altered that loss function to make it more practical and shared his code on the following GitHub repo:
https://github.com/iridiumblue/roc-star

that function gives us a nice ROC_AUC and an extremely high Recall score but the precision of the model was under the ground! As we got F1_score = 1.4%, ROC_AUC = %84, average_precision = %.6, Precision = %.71 and Recall = %92 and that’s our confusion matrix:<br>

<img src="https://github.com/Aml-Hassan-Abd-El-hamid/Fraud_Detection_using_Federated_Learning_AND_bank_application/blob/main/LEARNING/readme%20images/Screenshot%20from%202023-08-01%2004-03-34.png" width="340" height="340" >

3- LDAM (Label-Distribution-Aware Margin Loss):

That loss was first introduced in the paper: Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss
The authors of the paper mainly test that loss function on computer vision tasks and it is not actually very popular in the tabular data world, but we wanted to give it a try because after all the authors of that paper said that the target of their loss function is to help deep learning models with class imbalance. 

It’s based on the hinge-loss that is used in the famous SVM classifier, the usual loss that’s used in SVM encourages an equal margin for each class, but in that paper, it encourages a larger margin for the minority class.

Using the function with our network we managed to get  F1_score = 79%, ROC_AUC = %91, average_precision = %63, Precision = %77.7 and Recall = %82 
and here is our confusion matrix:<br>

<img src="https://github.com/Aml-Hassan-Abd-El-hamid/Fraud_Detection_using_Federated_Learning_AND_bank_application/blob/main/LEARNING/readme%20images/Screenshot%20from%202023-08-01%2004-21-20.png" width="340" height="340" >

4- Focal loss function:

That loss function was first introduced in the paper: Focal Loss for Dense Object Detection and it can be considered as an extension of the famous cross entropy function.

It was meant to deal with the not only imbalance in classes but also it’s supposed to be able to deal with hard-to-differentiate classes, which is what needed for fraud detection tasks as fraud transactions tend to disguise themselves as non-fraud so they are not only a minority but they are also hard to be differentiated, in the figure below, you can see a visual example of a minority class that is hard to differentiate, the example is for an image segmentation task actually as this function was originally made to deal with images segmentation tasks.<br>

<img src="https://github.com/Aml-Hassan-Abd-El-hamid/Fraud_Detection_using_Federated_Learning_AND_bank_application/blob/main/LEARNING/readme%20images/Screenshot%20from%202023-06-12%2013-57-27.png" width="440" height="340" > the source of this image is this video:https://www.youtube.com/watch?v=NqDBvUPD9jg

Focal loss focuses on the cases in which the model predicts wrongly rather than the cases in which predicts them confidently, which makes us ensure that predictions on hard examples improve over time, In other words, focal loss punishes the easy examples by down weighting them with a factor of  (1-  pt)γ.

You can see in the figure below how the added weight (1-  pt)γ can affect the performance of the model and also how the different values of γ can change the performance of the model:

<img src="https://github.com/Aml-Hassan-Abd-El-hamid/Fraud_Detection_using_Federated_Learning_AND_bank_application/blob/main/LEARNING/readme%20images/Screenshot%20from%202023-06-12%2014-23-41.png" width="460" height="340" > this image is from the paper

Also in the figure above, we can see clearly that the only difference between the cross entropy function and the focal loss function is the term: (1-  pt)γ, where γ≥0.

We didn’t use that original version of the focal loss exactly, we used a weighted one which give us the ability not only focus on the hard examples but also on the class that needs extra attention due to its high rarity in the dataset, 

That loss function gave us the highest balance results that we were able to get on the European credit card during our journey with it, we later also used that function when conducting federation on the European credit card dataset.

With a gamma = 3.75, an alph = [1,150] and an SGD optimizer we were able to get F1_score = 81.6%, ROC_AUC = %91, average_precision = %66.76, Precision = %81 and Recall = %82. And here is our confusion matrix:<br>

<img src="https://github.com/Aml-Hassan-Abd-El-hamid/Fraud_Detection_using_Federated_Learning_AND_bank_application/blob/main/LEARNING/readme%20images/Screenshot%20from%202023-06-12%2014-58-09.png" width="340" height="340" >


