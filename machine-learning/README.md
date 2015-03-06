Hands-on with machine learning
==============================

First of all, let me be clear about one thing: You're not going to "learn" machine learning in 60 minutes.

Instead, the goal of this session is to give you some sense of how to approach one type of machine learning in practice, specifically [http://en.wikipedia.org/wiki/Supervised_learning](supervised learning).

For this exercise, we'll be training a simple classifier that learns how to categorize bills from the California Legislature based only on their titles. Along the way, we'll focus on three steps critical to any supervised learning application: feature engineering, model building and evaluation.

To help us out, we'll be using a Python library called [http://scikit-learn.org/](scikit-learn), which is the easiest to understand machine learning library I've seen in any language.

That's a lot to pack in, so this session is going to move fast, and I'm going to assume you have a strong working knowledge of Python. Don't get caught up in the syntax. It's more important to understand the process.

Since we only have time to hit the very basics, I've also included some additional points you might find useful under the "What we're not covering" heading of each section below. There are also some resources at the bottom of this document that I hope will be helpful if you decide to learn more about this on your own.

Feature engineering
===================

Feature engineering can at once be one of the easiest and most difficult things to master in machine learning.

Features are how you represent data to your model. Most of the time, this involves two things: 1.) Selecting (or creating) elements of data that will be useful for the task at hand; and 2.) Representing them as numerical values in the form of a matrix.

The first part is an art. Creating features that you model finds useful can be a tedious exercise in trial and error, but fortunately it doesn't (necessarily) require any special knowledge of machine learning to get started. The second part requires some basic understanding of linear algebra and geometry, but nothing too intimidating.

For this example, we'll see how to extract features from just the words in our bill titles.

What we're covering
-------------------

- Features are the information we give our models. They are the only way models understand the world around them.

- Features are typically represented as a matrix composed of vectors. In CAR terms, you can think of this as being like a spreadsheet. In our example, each row is a document, each column represents a word contained in the whole set of documents, and each value is the number of times a given word appeared in a given document. This is also known as a term/document matrix.

- The number of columns also represents the dimensionality of our dataset in a geometric space known as the hyperplane. A matrix with two columns, for latitude and longitude, can be thought of as representing a vector in two-dimensional space (like a map). Our data follows a similar intuition but is represented in hundreds of dimensions.

What we're not covering
-----------------------

- In our example, features are simply counts of the words contained in our bill titles. But often it's helpful to define them more explicitly, depending on the task at hand. For example, if you're building a classifier to [identify quotes in text](https://github.com/cjdd3b/quotex-simple/), two useful features might be "paragraph contains quote marks" and "paragraph ends with quote marks". Often I like to define each of these as [functions](https://github.com/cjdd3b/quotex-simple/blob/master/features/features.py), which are combined later into a feature vector.

- Features can be either discrete or continuous, but different models handle those in different ways.

- Typically you'll want to scale your features so their values fall within a defined range (-1 to 1, for example). A number of simple [normalization techniques](http://en.wikipedia.org/wiki/Feature_scaling) can be applied during preprocessing to solve this.

- As a rule of thumb, keep the dimensionality of your feature space as low as possible. Your intuition of how points relate in two or three dimensions [doesn't apply in higher dimensional spaces](http://en.wikipedia.org/wiki/Curse_of_dimensionality). Dimensionality reduction techniques such as [principal component analysis](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) can be useful here.

- In some models, such as [Random Forests](http://scikit-learn.org/stable/modules/ensemble.html#feature-importance-evaluation), it's useful to understand how much each feature contributes to the model's predictive power. You can also do this to pare out features that add very little to the model. Likewise, it's also good to keep an eye out for features that correlate with each other.

Model building
==============

There are dozens of commonly used models for classification, all of which have certain advantages and disadvantages. But no matter which model we use, the process for working with it is largely the same.

First, we have to "train" it on a set of pre-labeled data, represented by the set of features we created in the previous step. Once that's done, we'll evaluate its performance, tune it, and, once we're satisfied, use it to classify new data.

Scikit-learn makes it easy to substitute one model for another, so we'll try two: Multinomial Naive Bayes and Random Forests.

What we're covering
-------------------

- Classification models essentially apply weights to features. They do this by looking at a pre-labeled set of data. After the model has been "trained," it can evaluate unseen data using the information it learned from the training set.

- A model is only as good as its features.

- Journalists have long used models -- particularly linear and logistic regression -- to explain variation within datasets. We're doing something similar but taking it one step further and using the model to predict something about unseen data.

- In the newsroom, I like to choose models that are relatively transparent and easy to explain: linear and logistic regression, Naive Bayes, k-nearest neighbors, decision trees and Random Forests are good examples. Things like Support Vector Machines and neural networks are more black-box, and so I only use those when there's a good reason.

What we're not covering
------------------------

- Overfitting happens when you create a classifier that performs well on training data but doesn't generalize well to unseen data. It's a major problem to beware of. Here's a great [visual explanation](http://www.quora.com/What-is-an-intuitive-explanation-of-overfitting) of what it looks like.

- Regularization is one good way to prevent overfitting, and it's worth [learning about](https://www.youtube.com/watch?v=Ms7QkS-uKiI).

- Models have different types of parameters that can be tuned to optimize their performance. Fine-tuning parameters can get you a bit more accuracy, but tuning them poorly can blow up your model. So be careful. Scikit-learn provides some useful tools to auto-tune parameters via methods like [Grid Search](http://scikit-learn.org/stable/modules/grid_search.html).

- Once a model is trained, you can persist it using either Python's pickle module or the slightly fancier [joblib](http://scikit-learn.org/stable/modules/model_persistence.html), which is packaged with scikit-learn. This is helpful when you have a model that takes a long time to train, and you don't want to retrain it every time you run your program.

- It's generally a good idea to keep the code that creates your model separate from the code that runs it. Use persistance to dump and load the model as needed.

Evaluation
==============

Knowing how to properly evaluate your models is perhaps the single most important thing in practical machine learning.

Having some intuition into how your model performs -- and specifically how it fails -- in some ways obivates the need to fully understand many of the complex mathematical theory that underpins machine learning in general, at least when you're first starting out.

Here we'll go over some techniques for seeing how our model performs, and how we can use that information to improve it.

What we're covering
-------------------

- Distill your model's performance into a single number. Then try to make that number go up.

- [K-fold cross-validation](http://scikit-learn.org/stable/modules/cross_validation.html) is generally a good way to evaluate classifier performance.

- Some, but not all, models can also produce confidence scores when they are applied to unseen data. In the case of scikit-learn's Random Forests, we can access these scores using the [predict_proba method](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict_proba).

What we're not covering
-----------------------

- In classification, be mindful of the tradeoff between [precision and recall](http://en.wikipedia.org/wiki/Precision_and_recall). Which one you optimize for depends on what you're trying to accomplish. The composite metric [F1](http://en.wikipedia.org/wiki/F1_score) captures some of this balance, but there are plenty of times it won't be the best metric for your task.

- Beware any model that performs too well.

- Take time to learn exactly what your model is getting right and what it's getting wrong. Looking at specific mistakes can guide you toward creating new features, tuning parameters and trying different types of models.

Other resources
===============

If you're interested in learning more about machine learning, I'd highly recommend Andrew Ng's [Machine Learning](https://www.coursera.org/course/ml) course on Coursera, and the book and online course [Mining of Massive Datasets](http://www.mmds.org/). Both provide a good theoretical grounding in many common techniques, as well as helpful practical advice.

O'Reilly also produces several books worth checking out: [Machine Learning for Hackers](http://shop.oreilly.com/product/0636920018483.do), [Programming Collective Intelligence](http://shop.oreilly.com/product/9780596529321.do) and [Data Analysis With Open Source Tools](http://shop.oreilly.com/product/9780596802363.do).

[Learning scikit-learn](http://www.amazon.com/Learning-scikit-learn-Machine-Python/dp/1783281936) and [Mastering Machine Learning with scikit-learn](https://www.packtpub.com/big-data-and-business-intelligence/mastering-machine-learning-scikit-learn) are also good for learning more about the scikit-learn library.

And you can always contact me if you have any questions: chase.davis@nytimes.com