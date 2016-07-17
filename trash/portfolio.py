#!/usr/bin/python
import sys
import urllib2
import numpy
import scipy
import operator
import math
import itertools
import random

def grapDATA(file, year, mon, day):
    # The following path is used for a customized path
    #PATH = "C:\Users\Wei\Documents\Courses\Investment\stocks\\"
    PATH = "./"
    
    dict = []
    for tickers in open(file):
        ticker = tickers.strip()
        dict.append(ticker)
    #dict = {'BIDU','AAPL', 'YHOO', 'MS'}

    # Build a dictonary to save data
    DATA = {}
    Date = {}

    # Grasp historical data from Yahoo.finance
    for k, name in enumerate(dict):
        print k + 1, name
        # create 2-order dict with name
        DATA[name] = {}
        # Build target url
        s0 = "http://real-chart.finance.yahoo.com/table.csv?s="
        s1 = "&amp;a="
        s2 = "&amp;b="
        s3 = "&amp;c="
        s4 = "&amp;d=02&amp;e=23&amp;f=2016&amp;g=d&amp;ignore=.csv"
        
        s = s0 + name + s1 + mon + s2 + day + s3 + year + s4
        # Link to the url
        response = urllib2.urlopen(s)
        csv = response.read()
        csv = csv.split('\n')

        #  Send data of every company to a file
        
        f = open(PATH + 'data/' + name + '.csv', "w")
        for line in csv:
            f.write(line + "\n")
        f.close()
        
        #print csv
        for num, line in enumerate(csv):
            line = line.strip().split(',')
            if len(line) < 7 or num == 0:
                continue
            date = line[0]
            Date[date] = ""
            open_ = line[1]
            high_ = line[2]
            low_ = line[3]
            close_ = line[4]
            vol_ = line[5]
            adjClose_ = line[6]
            Date[date] = ""
            DATA[name][date] = adjClose_

    # Clean DATA, first del data in days that lots data miss
    tmp = []
    for date in Date:
        tag = 0
        for name in DATA:
            if date not in DATA[name]:
                DATA[name][date] = '-'
            if DATA[name][date] == '-':
                tag += 1
        if tag > len(DATA) / 2:
            tmp.append(date)
    for date in tmp:
        del Date[date]
    # 2nd Del data if a company misses data
    tmp = []
    for name in DATA:
        tag = 0
        for date in Date:
            if DATA[name][date] == '-':
                tag += 1
        if tag > 0:
            tmp.append(name)
    for name in tmp:
        print 'Del Company\t' + name
        del DATA[name]

    #  Send data to a file
    fout = open(PATH + "allDATA" + '.csv', "w")
    s = "Date"
    for name in DATA:
        s += '\t' + name
    fout.write(s + '\n')
    for date in sorted(Date):
        s = date
        for name in DATA:
            if date not in DATA[name]:
                DATA[name][date] = '-'
            s += '\t' + DATA[name][date]
        fout.write(s + '\n')
    f.close()
    return DATA, Date

def calMeanVar(DATA, Date, LIST):
    Mat = {}
    Return = {}
    # Get the return of every companies
    for name in LIST:
        if name not in DATA:
            continue
        Mat[name] = []
        Return[name] = 0
        for num, date in enumerate(sorted(Date)):
            if num != 0:
                f_return = float(DATA[name][date]) / last - 1
                Mat[name].append(f_return)
                Return[name] += f_return
            last = float(DATA[name][date])
        Return[name] /= (len(DATA[name]) - 1)
    
    # Calculate the mean and variance
    dim = len(Mat)
    Cov = [[0 for x in range(dim)] for x in range(dim)] 
    tag = 0
    for n1, i in enumerate(Mat):
        for n2, j in enumerate(Mat):
            cov = numpy.cov(Mat[i], Mat[j])[0][1]
            Cov[n1][n2] = cov
    return Return, Cov

def tangency(Return, Cov, Rf):
    invCov = numpy.linalg.inv(Cov)
    dim = len(Return)
    mu_ninus_rf = map(operator.sub, Return.values(), [Rf] * dim)
    top_Mat = numpy.dot(invCov, mu_ninus_rf)

    bot_Mat = numpy.dot(top_Mat, [1] * dim)
    weight = top_Mat / bot_Mat

    avg_Return = numpy.dot(Return.values(), weight)
    annualized_r = round(avg_Return * 252 * 100, 2)
    tmp = numpy.dot(weight, Cov)
    sig2 = numpy.dot(tmp, weight)
    sigma = math.sqrt(sig2)
    annualized_sigma = round(sigma * math.sqrt(252) * 100, 2)
    SP = round((avg_Return - Rf) / sigma * 100, 2)

    # Weight
    '''
    print "Tinker\t\tReturn\t\tWeight"
    for num, name in enumerate(Return):
        print name + '\t\t' + str(round(Return[name] * 252 * 100,2)) + '%\t\t' + str(round(weight[num] * 100,2)) + '%'
    '''
    '''
    # Sharpe's ratio
    SP = round((avg_Return - Rf) / sigma * 100, 2)
    print "Return\t\tVolatility\tSharpe Ratio"
    print str(round(annualized_r, 2)) + '%\t\t' + str(round(annualized_sigma, 2)) + '%\t\t' + str(round(SP,2)) + '%'
    '''
    return SP, annualized_r, annualized_sigma


if __name__ == "__main__":
    # you can also read a file to get a list with much more data
    file = sys.argv[1]
    #dict = {'BIDU','AAPL', 'YHOO', 'MS'}
    # Choose a date to start
    Rf = 0.005 / 252
    DATA, Date = grapDATA(file, '2011', '02','05')
    LIST = {'AAPL', 'M','AAL','KTOS','SBUX','MMM','GS','T','GM','FE','HRB','X','CDI','MAS','UNH'}

    ALL = DATA.keys()
    print "ALL\t", ALL
    Choose = 15
    Max_SP = 0
    for length in range(10000):
        Sets = []
        for times in range(Choose):
            random_num = random.randint(0,89)
            Sets.append(ALL[random_num])
        Unique_Set = set(Sets)
        #if len(Unique_Set) < Choose:
        #    continue
        Return, Cov = calMeanVar(DATA, Date, Unique_Set)
        Cur_SP, Cur_R, Cur_sig = tangency(Return, Cov, Rf)
        if Cur_SP > Max_SP:
            print "Searching " + str(length + 1) + " times, current best sharpe ratio: " + str(Cur_SP) + \
            '%, annulized return: ' + str(Cur_R) + '%, volatility: ' + str(Cur_sig) + '%'
            print "Portfolio: ", str(Unique_Set),'\n'
            Max_SP = Cur_SP



        




