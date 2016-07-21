#!/usr/bin/python

import re
import sys
import time
import urllib
import urllib2
import requests
import zipfile
import StringIO
import random
from math import log
# yahoo finance api, containing incomplete index
from yahoo_finance import Share 
from pprint import pprint


class ticker_info:
	# pars are customized to the data from NASDAQ website
	def __init__(self, ticker, pars, index, ex):
		
		self.ticker = ticker
		self.name = pars[0]
		self.lastSale = pars[1]
		self.marketCap = float(pars[2])
		self.IPOyear = pars[4]
		self.sector = pars[5]
		self.industry = pars[6]

		'''
		# Api from https://pypi.python.org/pypi/yahoo-finance/1.2.1
		# Since grabbing these information consumes much more time, 
		# a better way is to split the whole process into 2 seperate parts, 
		# one is to download basic informations, one to do calculation
		info = Share(ticker)
		self.EBITDA = info.get_ebitda()
		self.EPS = info.get_earnings_share()
		self.EPS_growth = info.get_price_earnings_growth_ratio()
		self.price_book = info.get_price_book()
		self.moving_avg_50 = info.get_50day_moving_avg()
		self.moving_avg_200 = info.get_200day_moving_avg()
		self.trailingPE = index['trailingPE']
		self.forwardPE = index['forwardPE']
		'''
		self.exchange = ex

class stock_price:
	def __init__(self, pars):
		self.Open = pars[0]
		self.High = pars[1]
		self.Low = pars[2]
		self.Close = pars[3]
		self.Volume = pars[4]
		self.adjClose = pars[5]

# Reference: A Five-Factor Asset Pricing Model by Eugene F. Fama and Kenneth R. French
# http://www8.gsb.columbia.edu/programs/sites/programs/files/finance/Finance%20Seminar/spring%202014/ken%20french.pdf
class fama_french_factor:
	def __init__(self, pars):
		tag = len(pars)
		self.Mkt_Rf = pars[0]  # Market return - risk-free rate
		self.SMB = pars[1]     # Small minus big
		self.HML = pars[2]     # High minus low (book to market ratio)
		if tag == 6:
			self.RMW = pars[3]
			self.CMA = pars[4]
		self.RF = pars[tag - 1]



class web_spider:
	# http://finance.yahoo.com/quote/BABA/key-statistics, incomplete_version
	def key_stat(self, ticker):
		url = "http://finance.yahoo.com/quote/" + ticker + "/key-statistics"
		index = {"trailingPE" : None, "forwardPE" : None}
		try:
			response = urllib2.urlopen(url)
			csv = response.read()
			for par in index:
				reg = "\\\"" + par + "\\\":{\\\"raw\\\":([\\d.]+)"
				m = re.search(reg, csv)
				if m:
					index[par] =  float(m.group(1))
		except:
			print "Http error in grapping PE"
		return index

	# Get all the tickers in NYSE, NASDAQ and AMEX from NASDAQ website
	# http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download
	def tickers(self):
		url1 = "http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange="
		url2 = "&render=download"
		tickerList = {}
		for exchange in ["NASDAQ", "NYSE", "AMEX"]:
			print "Download tickers from " + exchange
			response = urllib2.urlopen(url1 + exchange + url2)
			csv = response.read().split('\n')
			for num, line in enumerate(csv):
				line = line.strip().strip('"').split('","')
				if num == 0 or len(line) != 9:
					continue
				ticker = line[0]
				index = {"trailingPE" : None, "forwardPE" : None}
				#index = self.key_stat(ticker)
				tickerList[ticker] = ticker_info(ticker, line[1:], index, exchange)
		return tickerList

	

	# US Treasury N Year Yield from Bloomberg
	def treasury_yield(self):
		url = "http://www.bloomberg.com/markets/rates-bonds/government-bonds/us"
		# "GB6" : 6-month-treasury bond yield
		# "GT1" : 1-year-treasury bond yield
		# "GT10" : 10-year-treasury bond yield
		print "Download treasury yield"
		response = urllib2.urlopen(url)
		csv = response.read()
		pars = re.findall(r'id\":\"([^,"]+):GOV.*?yield\":([^,]*)', csv)
		treasuryList = {}
		for line in pars:
			types = line[0]
			yields = float(line[1])
			treasuryList[types] = yields / 100
		return treasuryList

	# Fama-French-five-factors from French's website
	# home: http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
	def FF_factor(self, tag):
		# build url to download related file
		url = "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
		if tag == 3:	url += "F-F_Research_Data_Factors_CSV.zip"
		elif tag == 5:	url += "F-F_Research_Data_5_Factors_2x3_CSV.zip"
		else:	sys.exit("Error in FF_factor, unknown number of factors")
		print "Download historical Fama French factors"
		# unzip file and parse data
		filehandle, _ = urllib.urlretrieve(url)
		zip_file_object = zipfile.ZipFile(filehandle, 'r')
		first_file = zip_file_object.namelist()[0]
		file = zip_file_object.open(first_file)
		content = file.read().split('\n')
		factorList = {}
		for num, line in enumerate(content):
			line = line.strip().split(',')
			if num < 10 or len(line) != tag + 2:
				continue
			date = line[0]
			if not date.isdigit():
				continue
			if int(date) < 198001:
				continue
			newDate = date[0:4] + "-" + date[4:]
			for par in range(1, len(line)):
				line[par] = float(line[par]) / 100
			factorList[newDate] = fama_french_factor(line[1:])
		return factorList

	# Return the historical returns for a single stock
	# Perioidically Stock Price from Yahoo.finance, monthly: m; weekly: w; daily: d
	# http://chart.finance.yahoo.com/table.csv?s=BIDU&a=5&b=12&c=2016&d=6&e=12&f=2016&g=m&ignore=.csv
	def yahoo_price(self, ticker, time_start, time_end, time_type):
		# split parameters
		start_y, start_m, start_d = time_start.split("-")
		if len(time_end.split("-")) == 3:
			end_y, end_m, end_d = time_end.split("-")
		else: # until now
			end_y = "2999"; end_m = "12"; end_d = "01"
		if time_type not in ['d', 'w', 'm']:
			sys.exit("Error, unknown type!")
		start_m, end_m = str(int(start_m) - 1), str(int(end_m) - 1)
		ticker_price = {}

		# Construct url
		url1 = "http://chart.finance.yahoo.com/table.csv?s=" + ticker
		url2 = "&a=" + start_m + "&b=" + start_d + "&c=" + start_y
		url3 = "&d=" + end_m + "&e=" + end_d + "&f=" + end_y
		url4 = "&g=" + time_type + "&ignore=.csv"
		#print url1 + url2 + url3 + url4

		# sleep for sometime to reduce the burden of the website
		time.sleep(random.uniform(0.1, 1))

		# parse url
		response = urllib2.urlopen(url1 + url2 + url3 + url4)
		csv = response.read().split('\n')
		# get the prices of a ticker
		for num, line in enumerate(csv):
			line = line.strip().split(',')
			if len(line) < 7 or num == 0:
			    continue
			date = line[0]
			if not re.search(r'^[12]\d{3}-[01]\d-[0123]\d$', date):
				continue
			if time_type == "m":
				date = date[0:7]
			for arg in range(1, len(line)):
				line[arg] = float(line[arg])
			ticker_price[date] = stock_price(line[1:])
		return ticker_price
			
	def yahoo_return(self, ticker, time_start, time_end, time_type):
		ticker_return = {}
		ticker_price = self.yahoo_price(ticker, time_start, time_end, time_type)
		# get the return of a ticker, sort a dict with 2-dim key by time
		# the hard part is that different company may have different IPO time
		for num, date in enumerate(sorted(ticker_price)):
			if num != 0:
				tmp = []
				tmp.append(ticker_price[date].Open / last.Open - 1)
				tmp.append(ticker_price[date].High / last.High - 1)
				tmp.append(ticker_price[date].Low / last.Low - 1)
				tmp.append(ticker_price[date].Close / last.Close - 1)
				tmp.append(ticker_price[date].Volume / last.Volume - 1)
				tmp.append(ticker_price[date].adjClose / last.adjClose - 1)
				ticker_return[date] = stock_price(tmp)
				#print date, ticker_price[date].adjClose, ticker_return[date].adjClose
			last = ticker_price[date]
		return ticker_return


def main():
	spider = web_spider()
	lists = spider.tickers()
	for ticker in lists:
		spider.key_index(ticker)
		#spider.key_stat(ticker)
	#spider.treasury_yield()
	#spider.FF_five_factor()
	#spider.yahoo_all_returns("2015-01-01", "-",'m')
	#ss = ticker_info([0]*10, "NYSE")
	#ss.key_index("BABA")
	
if __name__ == "__main__":
	sys.exit(main())
	
