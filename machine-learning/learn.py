from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

if __name__ == '__main__':

    ########## STEP 1: DATA IMPORT AND PREPROCESSING ##########

    # Here we're taking in the training data and splitting it into two lists: One with the text of
    # each bill title, and the second with each bill title's corresponding category. Order is important.
    # The first bill in list 1 should also be the first category in list 2.
    training = [line.strip().split('|') for line in open('data/training.txt', 'r').readlines()]
    text = [t[0] for t in training if len(t) > 1]
    labels = [t[1] for t in training if len(t) > 1]

    # A little bit of cleanup for scikit-learn's benefit. Scikit-learn models wants our categories to
    # be numbers, not strings. The LabelEncoder performs this transformation.
    encoder = preprocessing.LabelEncoder()
    correct_labels = encoder.fit_transform(labels)

    ########## STEP 2: FEATURE EXTRACTION ##########
    print 'Extracting features ...'

    # vectorizer = CountVectorizer(stop_words='english', min_df=2, lowercase=True, analyzer='word')

    # These two lines use scikit-learn helpers to transform our training data into a document/term matrix.
    vectorizer = CountVectorizer()
    data = vectorizer.fit_transform(text).todense()

    # print data
    # print data.shape

    ########## STEP 3: MODEL BUILDING ##########
    print 'Training ...'

    # model = RandomForestClassifier(n_estimators=10, random_state=0)

    model = MultinomialNB()
    fit_model = model.fit(data, correct_labels)

    ########## STEP 4: EVALUATION ##########
    print 'Evaluating ...'

    # Evaluate our model with 10-fold cross-validation
    scores = cross_validation.cross_val_score(model, data, correct_labels, cv=10)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

    ########## STEP 5: APPLYING THE MODEL ##########
    print 'Classifying ...'

    docs_new = ["Public postsecondary education: executive officer compensation.",
                "An act to add Section 236.3 to the Education code, related to the pricing of college textbooks.",
                "Political Reform Act of 1974: campaign disclosures.",
                "An act to add Section 236.3 to the Penal Code, relating to human trafficking."
            ]

    test_data = vectorizer.transform(docs_new)

    for i in xrange(len(docs_new)):
        print '%s -> %s' % (docs_new[i], encoder.classes_[model.predict(test_data.toarray()[i])])