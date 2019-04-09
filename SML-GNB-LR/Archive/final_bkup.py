import numpy as np
import math
import random
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def dataPreperation(fileName):
    """ Parses the data into lists from file using 3-Fold"""

    with open(fileName, 'r') as f:
        data = f.read().split("\n")
    #print ("Size of data:",len(data))
    dataSize = len(data)

    parsedData = [each.split(",") for each in data]
    #print(parsedData)
    intParsedData = []
    for each in parsedData:
        intParsedData.append([float(val) for val in each ])
    #print (intParsedData)
    #print ("Size of data:",len(intParsedData))

    intParsedData = np.array(intParsedData)
    np.random.shuffle(intParsedData)
    #print (intParsedData)
    #print(intParsedData.shape)
    #print(intParsedData)
    kfold = KFold(3, True, 1)
    inputData = []
    for train, test in kfold.split(intParsedData):
        #print(train,test)
        ##print('train: %s, test: %s' % (intParsedData[train], intParsedData[test]))
        #print (intParsedData[train].shape, intParsedData[test].shape)
        inputData.append([intParsedData[train], intParsedData[test]])
        #input("DATAPREP")
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
    #print (f1c1,len(f1c1))
    #print ("%%%%%")
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
    #print ("sqDiffc0:",sqDiffc0)
    #print ("sqDiffc1:",sqDiffc1)
    #print ("len(f1c0:",len(f1c0))
    #print ("len(f1c1:",len(f1c1))

    variances.append([sum(sqDiffc0)/float(len(f1c0))])
    variances.append([sum(sqDiffc1)/float(len(f1c1))])


    # For feature f2 both classes 0 and 1
    f2 = data[: , [1,4]]
    f2c0 = f2[f2[:,1] == 0]
    f2c0 = f2c0[:,0].tolist()
    f2c1 = f2[f2[:,1] == 1]
    f2c1 = f2c1[:,0].tolist()
    sqDiffc0 = [pow((n-means[0][1]),2) for n in f2c0]
    sqDiffc1 = [pow((n-means[1][1]),2) for n in f2c1]
    variances[0].append(sum(sqDiffc0)/float(len(f2c0)))
    variances[1].append(sum(sqDiffc1)/float(len(f2c1)))

    # For feature f3 both classes 0 and 1
    f3 = data[: , [2,4]]
    f3c0 = f3[f3[:,1] == 0]
    f3c0 = f3c0[:,0].tolist()
    f3c1 = f3[f3[:,1] == 1]
    f3c1 = f3c1[:,0].tolist()
    sqDiffc0 = [pow((n-means[0][2]),2) for n in f3c0]
    sqDiffc1 = [pow((n-means[1][2]),2) for n in f3c1]
    variances[0].append(sum(sqDiffc0)/float(len(f3c0)))
    variances[1].append(sum(sqDiffc1)/float(len(f3c1)))

    # For feature f4 both classes 0 and 1
    f4 = data[: , [3,4]]
    f4c0 = f4[f4[:,1] == 0]
    f4c0 = f4c0[:,0].tolist()
    f4c1 = f4[f4[:,1] == 1]
    f4c1 = f4c1[:,0].tolist()
    sqDiffc0 = [pow((n-means[0][3]),2) for n in f4c0]
    sqDiffc1 = [pow((n-means[1][3]),2) for n in f4c1]
    variances[0].append(sum(sqDiffc0)/float(len(f4c0)))
    variances[1].append(sum(sqDiffc1)/float(len(f4c1)))

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
    #print ("*********")
    #print (means)
    #print (vars)
    #print ("*********")
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
    weights = np.zeros(numFeatures)
    #print(trainLabel.size)
    for eachIter in range(numIterations):
        #print (trainData.shape,weights.shape)
        weightedFeature = np.dot(trainData, weights)
        #print (weightedFeature.shape)
        #print(weightedFeature.shape)
        ####likelihood = 1.0/(1.0+np.exp(-weightedFeature))
        likelihood = 1.0/(1.0+np.exp(-weightedFeature))
        #print (likelihood.shape)
        gradient = np.dot(trainData.T,(likelihood -trainLabel))/(1.0*trainLabel.size)
        weights -= learningRate * gradient
        #print(gradient.shape)
        #input()
    prob = 1.0/(1.0+np.exp(-np.dot(testData, weights)))
    predicted = (prob >= threshold)
    accuracy = (predicted == testLabel)
    accuracy = accuracy.mean()
    #print (accuracy)
    #print(predicted)
    #input("KKK")
    return accuracy

def graphPlot(gAcc,lAcc,samples, fold=4):

    '''fig = plt.figure(figsize=(11,8))
    ax1 = fig.add_subplot(111)
    ax1.plot(gAcc, samples, label='Component 4', color='b', marker='o')
    plt.xlabel('Builds')'''
    fig = plt.figure()
    ax = plt.axes()

    ax.plot(samples,gAcc,label='GaussianNaiveBayes')
    ax.plot(samples,lAcc,label='LogisticRegression')

    plt.xlim(0, samples[-1])
    plt.ylim(0.7,1);
    plt.title("Accuracy vs Training-set Size")
    plt.xlabel("Training-set Size")
    plt.ylabel("Accuracy");
    plt.legend()
    plt.savefig("Graph_"+str(fold)+".png")

def checkIsOK(newTrainData):

    z = {}
    z['0']=0
    z['1']=0
    x = newTrainData[:,4].tolist()
    for i in x:
        z[str(int(i))] += 1
    # if there is only one data of each sample then  variance becomes zero
    if ( (z['0']<2) or (z['1']< 2) ):
        #input("GOTCHA")
        return False
    #print (z['0'],z['1'])
    #input("CHECKERR")
    return True

if __name__ == '__main__':

    # Data Preperation
    fileName = "data/data_banknote_authentication.txt"
    data = dataPreperation(fileName)

    #print (data)
    isShufflingOK = False
    while(not isShufflingOK):
        for j in data:
            x = len(set(j[0][:,4].tolist()))
            y = len(set(j[1][:,4].tolist()))

            #print(x,y)
            if (x == 1 or y == 1):
                input("Reshuffle")
                data = dataPreperation(fileName)
        isShufflingOK = True


    #input("WAIT")
    #[[train,test][train,test][train,test]]

    # Logistic Regression params Initialization
    learningRate = 0.01
    threshold = 0.5
    numIterations = 1000
    fractions = [.01, .02, .05, .1, .625, 1]
    numFeatures = 4

    # Gaussian Naive Bayes
    #GaussianAccuracy = []
    #LogisticAccuracy = []
    GNB_Accuracy = []
    LR_Accuracy = []
    samples = []
    fold = 0
    for i in data:
        #print (i)
        fold +=1
        trainData = i[0]
        #print (trainData.shape)
        testData = i[1]
        #print (testData.shape)
        #input("WAIT")
        #print(trainData.shape)
        dataSize = trainData.shape[0]
        #print(dataSize)
        samples = [math.ceil(i*dataSize) for i in fractions]
        #print(samples)
        GaussianAccuracy = []
        LogisticAccuracy = []
        for eachSample in samples:
            gAccuEachSample = []
            lAccuEachSample = []
            for noIter in range(5):
                selIdxTrain = random.sample(range(0,(dataSize)),eachSample)
                #print (selIdxTrain)
                newTrainData = trainData[selIdxTrain,:]
                while(not checkIsOK(newTrainData)):
                    selIdxTrain = random.sample(range(0,(dataSize)),eachSample)
                    newTrainData = trainData[selIdxTrain,:]
                #print (np.random.choice(trainData,(0.01*dataSize)))
                #input("RRRRRR")
                #print (newTrainData.shape)
                #print (newTrainData)

                gac = gaussianNaiveBayes(newTrainData,testData)
                gAccuEachSample.append(gac)
                lac = logisticRegression(newTrainData,testData,learningRate, threshold, numIterations)
                lAccuEachSample.append(lac)

            GaussianAccuracy.append(sum(gAccuEachSample)/len(gAccuEachSample))
            LogisticAccuracy.append(sum(lAccuEachSample)/len(lAccuEachSample))
            #input("NN")
            #print (GaussianAccuracy,len(GaussianAccuracy))
            #print (LogisticAccuracy,len(LogisticAccuracy))
            #input("TT")
        GNB_Accuracy.append(GaussianAccuracy)
        LR_Accuracy.append(LogisticAccuracy)
        #print ("Gaussian Accuracy for Fold - ",fold," is :", GaussianAccuracy)
        #print ("Logistic Accuracy for Fold - ",fold," is :", LogisticAccuracy)

        graphPlot(GaussianAccuracy, LogisticAccuracy, samples, fold)
        #input("RT")

    G = []
    L = []
    for i in range(len(fractions)):
        G.append( (GNB_Accuracy[0][i]+GNB_Accuracy[1][i]+GNB_Accuracy[2][i]) / 3.0 )
        L.append( (LR_Accuracy[0][i]+LR_Accuracy[1][i]+LR_Accuracy[2][i]) / 3.0 )

    print ("The Final Gaussian Naive Bayes Accuracy :",G)
    print ("The Final Logistic Regression Accuracy :",L)
    print ("\n")

    graphPlot(G, L, samples)


    #PART 5.2-C : Power of Generative model
    # Used the approach 1 of the two proposed approaches from TA
    # Here to create the part c separate, Using the training data for the the
    # previous folds created to learn the parameters and taken those for y=1
    # Using the parameters to generate the new 400 samples

    for eachFold in data:
        trainData = eachFold[0]
        testData = eachFold[1]

        means = calculateMeans(trainData)
        variances = calculateVariances(means,trainData)
        meansY1 = means[1]
        variancesY1 = variances[1]
        print ("Means Across each Features :",meansY1)
        print ("Variances Across each Features :",variancesY1)
        print ("\n")

        # For each Features finding means and variances
        for t in range(numFeatures):
            genData = np.random.normal(meansY1[t],math.sqrt(variancesY1[t]),400)
            #print (genData, genData.shape)
            print ("Generated Mean:",np.mean(genData),"Actual Mean:",meansY1[t])
            print ("Generated Variance:",np.var(genData),"Actual Variance:",variancesY1[t])
            print ("\n")
            #input()

