

def GaussianNaiveBayes (df):
    """
    Function to fit a Gaussian Naive Bayes model to the given features X and Labels y 
    df = [X|y]
    """
    import numpy as np

    pos_counts = df.disease.value_counts()['positive']
    neg_counts = df.disease.value_counts()['negative']
    
    pos_prior = pos_counts / df.shape[0]
    neg_prior = neg_counts / df.shape[0]
    
    values = [(pos_prior,neg_prior)]
    
    for i, column in enumerate(df.columns[:-1]):
        if df.dtypes[column] == np.int64:
            pos_vals = []
            neg_vals = []
            for j, val in enumerate(df[column]):
                if df.disease[j] == 'positive':
                    pos_vals.append(val)
                else:
                    neg_vals.append(val)
#             calculating the mean and the sample standard deviation 
            pos_mean = np.mean(pos_vals)
            pos_stdv = np.std(pos_vals, ddof= 1)
            neg_mean = np.mean(neg_vals)
            neg_stdv = np.std(neg_vals, ddof= 1)
            res = {'pos': (pos_mean,pos_stdv), 'neg': (neg_mean,neg_stdv)}
            values.append(res)
        else:
            res = {}
            unique = df[column].unique()
            for k in range(len(unique)):
                res [str(unique[k])] = [((1/len(unique))/(pos_counts+1)),
                                        ((1/len(unique))/(neg_counts+1))]
            for j, val in enumerate(df[column]):
                if df.disease[j] == 'positive':
                    res [str(val)][0] += 1 / ( pos_counts + 1)
                else:
                    res [str(val)][1] += 1 / ( neg_counts + 1)
            values.append(res)
    return values

def PrD (x, m, std):
    """
    Function to calculate the Gaussian probabilistic density of x with given mean and stdev
    """
    import math

    power = -(((x-m)**2)/(2*(std**2)))
    num = math.exp(power)
    denom = std*math.sqrt(2*math.pi)
    res = num / denom
    return res

def Predict (model,X_test):
    """
    Function to predict the disease using the GaussianNaiveBayes model from the given X_test
    """
    import numpy as np

    y = []
    yes_prob = []
    no_prob = []
    
    for x in X_test:
        (likelihood_of_yes, likelihood_of_no) = model[0]
        for j, val in enumerate(x, start =1):
            if not val == '_':
                if j in [1,3,6]:
                    likelihood_of_yes *= PrD(val, model[j]['pos'][0], model[j]['pos'][1])
                    likelihood_of_no *= PrD(val, model[j]['neg'][0], model[j]['neg'][1])
                else:
                    likelihood_of_yes *= model[j][str(val)][0]
                    likelihood_of_no *= model[j][str(val)][1]
        posteriori_pos = likelihood_of_yes / (likelihood_of_yes + likelihood_of_no)
        posteriori_neg = likelihood_of_no / (likelihood_of_yes + likelihood_of_no)
        yes_prob.append(posteriori_pos)
        no_prob.append(posteriori_neg)
        if posteriori_pos > posteriori_neg:
            y.append('positive')
        else:
            y.append('negative')
    return y, np.stack((yes_prob,no_prob), axis=-1)