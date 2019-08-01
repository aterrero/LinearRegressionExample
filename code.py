import pandas
import numpy 
import random

#Creating the 3 additional files
train1000_100=pandas.read_csv("train-1000-100.csv")
train1000_100.head(50).to_csv('train-50(1000)-100',index = False)
train1000_100.head(100).to_csv('train-100(1000)-100',index = False)
train1000_100.head(150).to_csv('train-150(1000)-100',index = False)

class LR:
    def __init__(self,filename):
        self.data = pandas.read_csv(filename)
        self.data.insert(loc=0,column='x0',value=[1]*len(self.data))
        self.X = numpy.matrix(self.data.drop('y',axis = 1))
        self.Y = numpy.matrix(self.data['y']).transpose()
        self.N = len(self.X)
        self.lambdas=numpy.linspace(0,150,151,dtype = int)
        self.XT = self.X.transpose()
        self.XTX = self.XT*self.X
        self.MSEarray = [0]*len(self.lambdas)
        
    def trainsetfunction(self):
        self.Warray = [0]*len(self.lambdas)
        for i in range(len(self.lambdas)):
            self.M = self.XTX + self.lambdas[i]*numpy.identity(len(self.XT))
            self.MI = numpy.linalg.inv(self.M)
            self.W = self.MI*self.XT*self.Y
            self.Warray[i] = self.W
            self.predictions = self.X*self.W
            self.MSEarray[i] = (numpy.linalg.norm(self.predictions-self.Y)**2)/self.N
        self.resultdf = pandas.DataFrame(self.lambdas)
        self.resultdf.columns = ['lambda']
        self.resultdf['MSE'] = self.MSEarray
    
    def testsetfunction(self,traindataset):
        for i in range(len(self.lambdas)):
            self.predictions = self.X*traindataset.Warray[i]
            self.MSEarray[i] = (numpy.linalg.norm(self.predictions-self.Y)**2)/self.N
        self.resultdf = pandas.DataFrame(self.lambdas)
        self.resultdf.columns = ['lambda']
        self.resultdf['MSE'] = self.MSEarray
        
    def CVfunction(self,k):
        self.foldlist = [0]*k
        for i in range(k):
            self.foldlist[i] = (self.data[i*int(self.N/k):(i+1)*int(self.N/k)])
        self.lambdaMSE = [0]*len(self.lambdas)
        self.foldsMSE = [0]*len(self.foldlist)
        for i in range(len(self.lambdas)):
            for j in range(len(self.foldlist)):
                indexes = list(range(len(self.foldlist)))
                testfoldX = numpy.matrix(self.foldlist[j].drop('y',axis=1))
                testfoldY = numpy.matrix(self.foldlist[j]['y']).transpose()
                testfoldN = len(testfoldX)
                trainindexes = indexes[0:j]+indexes[j+1:len(indexes)]
                trainfolds = [0]*len(trainindexes)
                for k in range(len(trainfolds)):
                    trainfolds[k] = self.foldlist[trainindexes[k]]    
                trainfoldX = numpy.matrix(pandas.concat(trainfolds).drop('y',axis=1))
                trainfoldY = numpy.matrix(pandas.concat(trainfolds)['y']).transpose()
                trainfoldXT = trainfoldX.transpose()
                trainfoldXTX = trainfoldXT*trainfoldX
                trainfoldM = trainfoldXTX + self.lambdas[i]*numpy.identity(len(trainfoldXT))
                trainfoldMI = numpy.linalg.inv(trainfoldM)
                trainfoldW = trainfoldMI*trainfoldXT*trainfoldY
                testfoldpredictions = testfoldX*trainfoldW
                testfoldMSE = (numpy.linalg.norm(testfoldpredictions-testfoldY)**2)/testfoldN
                self.foldsMSE[j] = testfoldMSE
            self.lambdaMSE[i] = sum(self.foldsMSE)/len(self.foldsMSE)
        self.lambdaMSEdf = pandas.DataFrame(self.lambdaMSE)
        
    def LCfunction(self,trainingset,iterations):
        self.lambdas2=[1,25,150]
        self.trainsetsizes=numpy.linspace(10,int(len(self.data)/2),int((len(self.data)/2)/10),dtype=int)
        self.trainseterror = [0]*len(self.trainsetsizes)
        self.lcerrorlist = [[[0]*len(self.trainsetsizes) for count in range(2)] for count in range(3)]
        lcresultlist = [0]*iterations
        
        for j in range(len(lcresultlist)):
            for x in range(len(self.trainsetsizes)):
                lctrainset = trainingset.data.iloc[random.sample(list(trainingset.data.index), self.trainsetsizes[x])]
                lcX = numpy.matrix(lctrainset.drop('y',axis=1))
                lcN = len(lcX)
                lcY = numpy.matrix(lctrainset['y']).transpose()
                lcXT = lcX.transpose()
                lcXTX = lcXT*lcX
                for y in range(len(self.lambdas2)):
                    lcM = lcXTX + self.lambdas2[y]*numpy.identity(len(lcXT))
                    lcMI = numpy.linalg.inv(lcM)
                    lcW = lcMI*lcXT*lcY
                    lctrainpredictions = lcX*lcW
                    lctestpredictions = self.X*lcW
                    lctrainerror = (numpy.linalg.norm(lctrainpredictions-lcY)**2)/lcN
                    lctesterror = (numpy.linalg.norm(lctestpredictions-self.Y)**2)/self.N
                    self.lcerrorlist[y][0][x] = lctrainerror
                    self.lcerrorlist[y][1][x] = lctesterror
            lcresults = pandas.DataFrame(self.lcerrorlist[0][0])
            lcresults.columns = ['lambda = 1 train error']
            lcresults['lambda = 1 test error'] = self.lcerrorlist[0][1]
            lcresults['lambda = 25 train error'] = self.lcerrorlist[1][0]    
            lcresults['lambda = 25 test error'] = self.lcerrorlist[1][1]
            lcresults['lambda = 150 train error'] = self.lcerrorlist[2][0]    
            lcresults['lambda = 150 test error'] = self.lcerrorlist[2][1]
            lcresultlist[j] = lcresults
        
        lambda1trainerrs = [0]*len(lcresultlist)
        lambda1testerrs = [0]*len(lcresultlist)
        lambda25trainerrs = [0]*len(lcresultlist)
        lambda25testerrs = [0]*len(lcresultlist)
        lambda150trainerrs = [0]*len(lcresultlist)
        lambda150testerrs = [0]*len(lcresultlist)
        self.finalLCdf = lcresults.copy(deep=True)
        
        for x in range(len(self.trainsetsizes)):
            for y in range(len(lcresultlist)):
                lambda1trainerrs[y] = lcresultlist[y]['lambda = 1 train error'][x]
                lambda1testerrs[y] = lcresultlist[y]['lambda = 1 test error'][x]
                lambda25trainerrs[y] = lcresultlist[y]['lambda = 25 train error'][x]
                lambda25testerrs[y] = lcresultlist[y]['lambda = 25 test error'][x]
                lambda150trainerrs[y] = lcresultlist[y]['lambda = 150 train error'][x]
                lambda150testerrs[y] = lcresultlist[y]['lambda = 150 test error'][x]
            self.finalLCdf['lambda = 1 train error'][x] = sum(lambda1trainerrs)/len(lambda1trainerrs)
            self.finalLCdf['lambda = 1 test error'][x] = sum(lambda1testerrs)/len(lambda1testerrs)
            self.finalLCdf['lambda = 25 train error'][x] = sum(lambda25trainerrs)/len(lambda25trainerrs)
            self.finalLCdf['lambda = 25 test error'][x] = sum(lambda25testerrs)/len(lambda25testerrs)
            self.finalLCdf['lambda = 150 train error'][x] = sum(lambda150trainerrs)/len(lambda150trainerrs)
            self.finalLCdf['lambda = 150 test error'][x] = sum(lambda150testerrs)/len(lambda150testerrs)
        
        
        
#Test datasets
test10010="test-100-10.csv"
test100100="test-100-100.csv"
test1000100="test-1000-100.csv"

#Train datasets
train10010="train-100-10.csv"
train100100="train-100-100.csv"
train1000100="train-1000-100.csv"

train501000100="train-50(1000)-100"
train1001000100="train-100(1000)-100"
train1501000100="train-150(1000)-100"

## Calling class methods

#Train/Test 100-10:
train_100_10=LR(train10010)
train_100_10.trainsetfunction()
test_100_10=LR(test10010)
test_100_10.testsetfunction(train_100_10)
train_100_10.CVfunction(10)
 
#Train/Test 100-100:
train_100_100=LR(train100100)
train_100_100.trainsetfunction()
test_100_100=LR(test100100)
test_100_100.testsetfunction(train_100_100)
train_100_100.CVfunction(10)

#Train/Test 1000-100:
train_1000_100=LR(train1000100)
train_1000_100.trainsetfunction()
test_1000_100=LR(test1000100)
test_1000_100.testsetfunction(train_1000_100)
train_1000_100.CVfunction(10)
train_1000_100.LCfunction(train_1000_100,1000)

#Train/Test 50(1000)-100:
train_50_1000_100=LR(train501000100)
train_50_1000_100.trainsetfunction()
test_50_1000_100=LR(test1000100)
test_50_1000_100.testsetfunction(train_50_1000_100)
train_50_1000_100.CVfunction(10)

#Train/Test 100(1000)-100:
train_100_1000_100=LR(train1001000100)
train_100_1000_100.trainsetfunction()
test_100_1000_100=LR(test1000100)
test_100_1000_100.testsetfunction(train_100_1000_100)
train_100_1000_100.CVfunction(10)

#Train/Test 50(1000)-100:
train_150_1000_100=LR(train1501000100)
train_150_1000_100.trainsetfunction()
test_150_1000_100=LR(test1000100)
test_150_1000_100.testsetfunction(train_150_1000_100)
train_150_1000_100.CVfunction(10)

jointMSEdf=pandas.DataFrame(train_100_10.resultdf['MSE'])
jointMSEdf.columns=['Train 100-10 MSE']
jointMSEdf['Train 100-100 MSE']=train_100_100.resultdf['MSE']
jointMSEdf['Train 1000-100 MSE']=train_1000_100.resultdf['MSE']
jointMSEdf['Train 50(1000)-100 MSE']=train_50_1000_100.resultdf['MSE']
jointMSEdf['Train 100(1000)-100 MSE']=train_100_1000_100.resultdf['MSE']
jointMSEdf['Train 150(1000)-100 MSE']=train_150_1000_100.resultdf['MSE']
jointMSEdf['Test 100-10 MSE']=test_100_10.resultdf['MSE']
jointMSEdf['Test 100-100 MSE']=test_100_100.resultdf['MSE']
jointMSEdf['Test 1000-100 MSE']=test_1000_100.resultdf['MSE']
jointMSEdf['Test 50(1000)-100 MSE']=test_50_1000_100.resultdf['MSE']
jointMSEdf['Test 100(1000)-100 MSE']=test_100_1000_100.resultdf['MSE']
jointMSEdf['Test 150(1000)-100 MSE']=test_150_1000_100.resultdf['MSE']
#jointMSEdf.plot()

############################ - Question 2 - ############################

jointMSEdf[['Train 100-10 MSE','Test 100-10 MSE']].plot()
jointMSEdf[['Train 100-100 MSE','Test 100-100 MSE']].plot()
jointMSEdf[['Train 1000-100 MSE','Test 1000-100 MSE']].plot()
jointMSEdf[['Train 50(1000)-100 MSE','Test 50(1000)-100 MSE']].plot()
jointMSEdf[['Train 100(1000)-100 MSE','Test 100(1000)-100 MSE']].plot()
jointMSEdf[['Train 150(1000)-100 MSE','Test 150(1000)-100 MSE']].plot()

#a)
a=jointMSEdf[['Test 100-10 MSE']].sort_values(by='Test 100-10 MSE').index.tolist()[0]
b=jointMSEdf[['Test 100-10 MSE']].sort_values(by='Test 100-10 MSE').iloc[0,0]
print('For Test 100-10 the best lambda value is %d ' %a + 'and MSE is %f\n' %b)

a=jointMSEdf[['Test 100-100 MSE']].sort_values(by='Test 100-100 MSE').index.tolist()[0]
b=jointMSEdf[['Test 100-100 MSE']].sort_values(by='Test 100-100 MSE').iloc[0,0]
print('For Test 100-100 the best lambda value is %d ' %a + 'and MSE is %f\n' %b)

a=jointMSEdf[['Test 1000-100 MSE']].sort_values(by='Test 1000-100 MSE').index.tolist()[0]
b=jointMSEdf[['Test 1000-100 MSE']].sort_values(by='Test 1000-100 MSE').iloc[0,0]
print('For Test 1000-100 the best lambda value is %d ' %a + 'and MSE is %f\n' %b)

a=jointMSEdf[['Test 50(1000)-100 MSE']].sort_values(by='Test 50(1000)-100 MSE').index.tolist()[0]
b=jointMSEdf[['Test 50(1000)-100 MSE']].sort_values(by='Test 50(1000)-100 MSE').iloc[0,0]
print('For Test 50(1000)-100 the best lambda value is %d ' %a + 'and MSE is %f\n' %b)

a=jointMSEdf[['Test 100(1000)-100 MSE']].sort_values(by='Test 100(1000)-100 MSE').index.tolist()[0]
b=jointMSEdf[['Test 100(1000)-100 MSE']].sort_values(by='Test 100(1000)-100 MSE').iloc[0,0]
print('For Test 100(1000)-100 the best lambda value is %d ' %a + 'and MSE is %f\n' %b)

a=jointMSEdf[['Test 150(1000)-100 MSE']].sort_values(by='Test 150(1000)-100 MSE').index.tolist()[0]
b=jointMSEdf[['Test 150(1000)-100 MSE']].sort_values(by='Test 150(1000)-100 MSE').iloc[0,0]
print('For Test 150(1000)-100 the best lambda value is %d ' %a + 'and MSE is %f\n' %b)

#b)
jointMSEdf[['Train 100-100 MSE','Test 100-100 MSE']].iloc[1:].plot()
jointMSEdf[['Train 50(1000)-100 MSE','Test 50(1000)-100 MSE']].iloc[1:].plot()
jointMSEdf[['Train 100(1000)-100 MSE','Test 100(1000)-100 MSE']].iloc[1:].plot()

#c)
print('MSE is abnormally large for these three datasets because of Overfitting.')

############################ - Question 3 - ############################

#a)
#train_100_10.lambdaMSEdf.plot()
a=train_100_10.lambdaMSEdf.sort_values(by=0).index.tolist()[0]
b=train_100_10.lambdaMSEdf.sort_values(by=0).iloc[0,0]
print('For Train 100-10 the best lambda value according to CV is %d ' %a + 'and MSE is %f\n' %b)

a=train_100_100.lambdaMSEdf.sort_values(by=0).index.tolist()[0]
b=train_100_100.lambdaMSEdf.sort_values(by=0).iloc[0,0]
print('For Train 100-100 the best lambda value according to CV is %d ' %a + 'and MSE is %f\n' %b)

a=train_1000_100.lambdaMSEdf.sort_values(by=0).index.tolist()[0]
b=train_1000_100.lambdaMSEdf.sort_values(by=0).iloc[0,0]
print('For Train 1000-100 the best lambda value according to CV is %d ' %a + 'and MSE is %f\n' %b)

a=train_50_1000_100.lambdaMSEdf.sort_values(by=0).index.tolist()[0]
b=train_50_1000_100.lambdaMSEdf.sort_values(by=0).iloc[0,0]
print('For Train 50(1000)-100 the best lambda value according to CV is %d ' %a + 'and MSE is %f\n' %b)

a=train_100_1000_100.lambdaMSEdf.sort_values(by=0).index.tolist()[0]
b=train_100_1000_100.lambdaMSEdf.sort_values(by=0).iloc[0,0]
print('For Train 100(1000)-100 the best lambda value according to CV is %d ' %a + 'and MSE is %f\n' %b)

a=train_150_1000_100.lambdaMSEdf.sort_values(by=0).index.tolist()[0]
b=train_150_1000_100.lambdaMSEdf.sort_values(by=0).iloc[0,0]
print('For Train 150(1000)-100 the best lambda value according to CV is %d ' %a + 'and MSE is %f\n' %b)

#b)
print('Compared to the results of question 2a), the suggested values for lambda are reasonably close\n')

#c)
print('Cross Validation is quite expensive in terms of computing power and running time\n')

#d)
print('Factors affecting performance of CV are the size of the dataset itself, \nthe range of lambdas and the amount of folds chosen.\n')
 
############################ - Question 4 - ############################

train_1000_100.finalLCdf[['lambda = 1 test error','lambda = 1 train error']].plot()
train_1000_100.finalLCdf[['lambda = 25 test error','lambda = 25 train error']].plot()
train_1000_100.finalLCdf[['lambda = 150 test error','lambda = 150 train error']].plot()
