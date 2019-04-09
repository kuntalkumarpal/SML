import numpy as np
import math
import random
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def dataPreperation(fileName):
    """ Parses the data into lists from file"""

    p = [.01 ,.02, .05 ,.1 ,.625 ,1]
    print([math.ceil(i*1372) for i in p])
    with open(fileName, 'r') as f:
        data = f.read().split("\n")
    print ("Size of data:",len(data))
    dataSize = len(data)

    parsedData = [each.split(",") for each in data]
    print(parsedData)
    intParsedData = []
    for each in parsedData:
        intParsedData.append([float(val) for val in each ])
    #print (intParsedData)
    print ("Size of data:",len(intParsedData))

    intParsedData = np.array(intParsedData)
    print(intParsedData.size)
    #print(intParsedData)
    kfold = KFold(3, True, 1)
    inputData = []
    for train, test in kfold.split(intParsedData):
        #print(train,test)
        print('train: %s, test: %s' % (intParsedData[train], intParsedData[test]))
        print (intParsedData[train].shape, intParsedData[test].shape)
        inputData.append([intParsedData[train], intParsedData[test]])
        input("DATAPREP")
    #print(inputData,len(inputData),len(inputData[0]))

    return inputData
def calculateMeans(data):
    """ Calculate Means across each of the features for classes 0 and 1"""
    #[[m,m,m,m] [m,m,m,m]]

    means = []

    # For feature f1 both classes 0 and 1
    f1 = data[:, [0, 4]]
    f1c0 = f1[f1[:,1] == 0]
    f1c0 = f1c0[:,0].tolist()
    f1c1 = f1[f1[:,1] == 1]
    f1c1 = f1c1[:,0].tolist()
    means.append([sum(f1c0)/float(len(f1c0))])
    means.append([sum(f1c1)/float(len(f1c1))])


    # For feature f2 both classes 0 and 1
    f2 = data[: , [1,4]]
    f2c0 = f2[f2[:,1] == 0]
    f2c0 = f2c0[:,0].tolist()
    f2c1 = f2[f2[:,1] == 1]
    f2c1 = f2c1[:,0].tolist()
    means[0].append(sum(f2c0)/float(len(f2c0)))
    means[1].append(sum(f2c1)/float(len(f2c1)))

    # For feature f3 both classes 0 and 1
    f3 = data[: , [2,4]]
    f3c0 = f3[f3[:,1] == 0]
    f3c0 = f3c0[:,0].tolist()
    f3c1 = f3[f3[:,1] == 1]
    f3c1 = f3c1[:,0].tolist()
    means[0].append(sum(f3c0)/float(len(f3c0)))
    means[1].append(sum(f3c1)/float(len(f3c1)))

    # For feature f4 both classes 0 and 1
    f4 = data[: , [3,4]]
    f4c0 = f4[f4[:,1] == 0]
    f4c0 = f4c0[:,0].tolist()
    f4c1 = f4[f4[:,1] == 1]
    f4c1 = f4c1[:,0].tolist()
    means[0].append(sum(f4c0)/float(len(f4c0)))
    means[1].append(sum(f4c1)/float(len(f4c1)))

    return means


def calculateVariances(means, data):
    """ Calculate Variances across each of the features for classes 0 and 1"""
    #[[v,v,v,v] [v,v,v,v]]

    variances = []

    # For feature f1 both classes 0 and 1
    f1 = data[:, [0, 4]]
    f1c0 = f1[f1[:,1] == 0]
    f1c0 = f1c0[:,0].tolist()
    f1c1 = f1[f1[:,1] == 1]
    f1c1 = f1c1[:,0].tolist()
    sqDiffc0 = [pow((n-means[0][0]),2) for n in f1c0]
    sqDiffc1 = [pow((n-means[1][0]),2) for n in f1c1]
    variances.append([sum(sqDiffc0)/float(len(f1c0)-1)])
    variances.append([sum(sqDiffc1)/float(len(f1c1)-1)])


    # For feature f2 both classes 0 and 1
    f2 = data[: , [1,4]]
    f2c0 = f2[f2[:,1] == 0]
    f2c0 = f2c0[:,0].tolist()
    f2c1 = f2[f2[:,1] == 1]
    f2c1 = f2c1[:,0].tolist()
    sqDiffc0 = [pow((n-means[0][1]),2) for n in f2c0]
    sqDiffc1 = [pow((n-means[1][1]),2) for n in f2c1]
    variances[0].append(sum(sqDiffc0)/float(len(f2c0)-1))
    variances[1].append(sum(sqDiffc1)/float(len(f2c1)-1))

    # For feature f3 both classes 0 and 1
    f3 = data[: , [2,4]]
    f3c0 = f3[f3[:,1] == 0]
    f3c0 = f3c0[:,0].tolist()
    f3c1 = f3[f3[:,1] == 1]
    f3c1 = f3c1[:,0].tolist()
    sqDiffc0 = [pow((n-means[0][2]),2) for n in f3c0]
    sqDiffc1 = [pow((n-means[1][2]),2) for n in f3c1]
    variances[0].append(sum(sqDiffc0)/float(len(f3c0)-1))
    variances[1].append(sum(sqDiffc1)/float(len(f3c1)-1))

    # For feature f4 both classes 0 and 1
    f4 = data[: , [3,4]]
    f4c0 = f4[f4[:,1] == 0]
    f4c0 = f4c0[:,0].tolist()
    f4c1 = f4[f4[:,1] == 1]
    f4c1 = f4c1[:,0].tolist()
    sqDiffc0 = [pow((n-means[0][3]),2) for n in f4c0]
    sqDiffc1 = [pow((n-means[1][3]),2) for n in f4c1]
    variances[0].append(sum(sqDiffc0)/float(len(f4c0)-1))
    variances[1].append(sum(sqDiffc1)/float(len(f4c1)-1))

    return variances

def getPrior(data):
    """ Finds the Prior of the data"""
    prior = []
    dataLen = data.shape[0]
    #print (dataLen)
    numClass0 = len(data[data[:,4]==0])
    numClass1 = dataLen - numClass0
    #print (numClass0)
    #print (numClass1)

    return [numClass0/(1.0*dataLen),numClass1/(1.0*dataLen)]



def GNB(data):
    #print (data)
    means = calculateMeans(data)
    vars = calculateVariances(means,data)
    #print (means)
    #print (vars)
    return means,vars

def predictGNB(test,mean, variance, prior):
    """ Returns the predicted Classes for each of the test data"""
    ''' test : (noTest,5)
    '''
    predictions = []
    for eachData in test.tolist():
        probc0 = prior[0]
        probc1 = prior[1]
        for i in range(len(eachData)-1):
            probc0 *= (1.0/math.sqrt(2*math.pi*variance[0][i])) * math.exp(-math.pow((eachData[i]-mean[0][i]),2)/(2.0*variance[0][i]))
            probc1 *= (1.0/math.sqrt(2*math.pi*variance[1][i])) * math.exp(-math.pow((eachData[i]-mean[1][i]),2)/(2.0*variance[1][i]))
            #print ("Prob:", probc0, probc1)
        #print ("ProbF:", probc0, probc1)

        if probc0 > probc1:
            predictions.append(0)
        else:
            predictions.append(1)

    return predictions

def accuracyGNB(predicted, gold):
    noMatches = len([1 for i,j in zip(predicted,gold) if i==j])
    return noMatches/(1.0 * len(gold))

def gaussianNaiveBayes(trainData,testData):

    #print (testData)
    testLabel = testData[:,4].tolist()
    prior = getPrior(trainData)
    mean, variance = GNB(trainData)
    predicted = predictGNB(testData, mean, variance, prior)
    accuracy = accuracyGNB(predicted, testLabel)
    #print (accuracy)
    return accuracy

def logisticRegression(trainData,testData, learningRate, threshold, numIterations):

    trainLabel = trainData[:,4]
    trainData = trainData[:,[0,1,2,3]]
    testLabel = testData[:,4]
    testData = testData[:,[0,1,2,3]]
    numFeatures = trainData.shape[1]
    #print (numFeatures,trainData.shape)
    weights = np.random.zeros(numFeatures)
    #print(trainLabel.size)
    for eachIter in range(numIterations):
        weightedFeature = np.dot(trainData, weights)
        #print(weightedFeature.shape)
        likelihood = 1.0/(1.0+np.exp(-weightedFeature))
        gradient = np.dot(trainData.T,(likelihood - trainLabel)) / (1.0*trainLabel.size)
        weights -= learningRate * gradient

    prob = 1.0/(1.0+np.exp(-np.dot(testData, weights)))
    predicted = (prob >= threshold)
    accuracy = (predicted == testLabel).mean()
    #print (accuracy)
    return accuracy

def graphPlot(gAcc,lAcc,samples):

    '''fig = plt.figure(figsize=(11,8))
    ax1 = fig.add_subplot(111)
    ax1.plot(gAcc, samples, label='Component 4', color='b', marker='o')
    plt.xlabel('Builds')'''
    fig = plt.figure()
    ax = plt.axes()

    ax.plot(samples,gAcc,label='GaussianNaiveBayes')
    ax.plot(samples,lAcc,label='LogisticRegression')

    plt.xlim(0, samples[-1])
    plt.ylim(0.5,1);
    plt.title("Accuracy vs Training-set Size")
    plt.xlabel("Training-set Size")
    plt.ylabel("Accuracy");
    plt.legend()
    plt.show()

if __name__ == '__main__':

    # Data Preperation
    fileName = "data/data_banknote_authentication.txt"
    data = dataPreperation(fileName)
    #print (data)
    #input("WAIT")
    #[[train,test][train,test][train,test]]

    # Logistic Regression params Initialization
    learningRate = 0.01
    threshold = 0.5
    numIterations = 100

    # Gaussian Naive Bayes
    GaussianAccuracy = []
    LogisticAccuracy = []
    samples = []
    for i in data:
        #print (i)
        trainData = i[0]
        #print (trainData)
        testData = i[1]
        #print (testData)
        #input("WAIT")
        gaussianNaiveBayes(trainData,testData)

        #print (trainData)
        #trainData = data[0][0]
        #testData = data[0][1]
        #print (trainData.shape,testData.shape)
        logisticRegression(trainData,testData, learningRate, threshold, numIterations)

        trainSize = trainData.shape[0]
        fractions = [.01, .02, .05, .1, .625, 1]
        samples = [math.ceil(i*trainSize) for i in fractions]
        selIdx = [random.sample(range(0,trainSize),i) for i in samples]
        #print(trainSize)
        print(samples)
        #print(selIdx)
        #print ([len(set(i)) for i in selIdx])
        gAcc = []
        lAcc = []
        for j in selIdx:
            trainDataFiltered = trainData[j,:]
            #print (trainDataFiltered.shape)

            eachGAcc = []
            eachLAcc = []
            for k in range(5):
                eachGAcc.append(gaussianNaiveBayes(trainDataFiltered,testData))
                eachLAcc.append(logisticRegression(trainDataFiltered,testData, learningRate, threshold, numIterations))
            #print(eachGAcc)
            #print(eachLAcc)
            gAcc.append(sum(eachGAcc)/len(eachGAcc))
            lAcc.append(sum(eachLAcc)/len(eachLAcc))

        print (gAcc)
        print (lAcc)
        GaussianAccuracy.append(gAcc)
        LogisticAccuracy.append(lAcc)
        #print (trainData[[2,1,5,99],:])

    print (GaussianAccuracy)
    print (LogisticAccuracy)
    for x in range(3):

        graphPlot(GaussianAccuracy[x],LogisticAccuracy[x],samples)
