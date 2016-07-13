#!/usr/bin/python

import re
import sys
import urllib
import urllib2
import requests
import zipfile
import StringIO
from math import log


class ticker_info:
	# pars are customized to the data from NASDAQ website
	def __init__(self, pars, ex):
		self.symbol = pars[0]
		self.name = pars[1]
		self.lastSale = pars[2]
		self.marketCap = pars[3]
		self.IPOyear = pars[5]
		self.sector = pars[6]
		self.industry = pars[7]
		self.exchange = ex

	# http://finance.yahoo.com/quote/BABA/key-statistics
	# incomplete_version
	def key_stat(self, ticker):
		url = "http://finance.yahoo.com/quote/" + ticker + "/key-statistics"
		response = urllib2.urlopen(url)
		csv = response.read()
		print csv
		parList = ["marketCap", "trailingPE", "forwardPE"]
		index = {}
		for par in parList:
			reg = "\\\"" + par + "\\\":{\\\"raw\\\":([\\d.]+)"
			m = re.search(reg, csv)
			if m:
				index[par] =  float(m.group(1))
				print par, float(m.group(1))
			else:
				index[par] = None

	def key_index(self, ticker):
		f = open('/home/wayne/Portfolio_Selection/tag')
		tag = ""
		Dict = {}
		for num, l in enumerate(f):
			l = l.strip().split("\t")
			if len(l) != 2:
				continue
			tag += l[0]
			Dict[num] = l[1]
		url = "http://download.finance.yahoo.com/d/quotes.csv?s=" + ticker + "&f=" + tag + "&e=.csv"
		response = urllib2.urlopen(url)
		csv = response.read().split(',')
		#for num in range(len(csv)):
		#	print num, Dict[num], csv[num]
		url = "http://finance.yahoo.com/q/ks?s=" + ticker + "+Key+Statistics"
		response = urllib2.urlopen(url)
		csv = response.read().split(',')
		print csv
		print url





# Reference: A Five-Factor Asset Pricing Model by Eugene F. Fama and Kenneth R. French
# http://www8.gsb.columbia.edu/programs/sites/programs/files/finance/Finance%20Seminar/spring%202014/ken%20french.pdf
class fama_french_factor:
	def __init__(self, pars):
		tag = len(pars)
		self.Mkt_Rf = pars[0]  # Market return - risk-free rate
		self.SMB = pars[1]     # Small minus big
		self.HML = pars[2]     # High minus low (book to market ratio)
		if tag == 4:
			self.RF = pars[3]
		if tag == 6:
			self.CMA = pars[4]
			self.RMW = pars[3]
			self.RF = pars[5]

class stock_price:
	def __init__(self, pars):
		self.Open = pars[0]
		self.High = pars[1]
		self.Low = pars[2]
		self.Close = pars[3]
		self.Volume = pars[4]
		self.adjClose = pars[5]

class web_spider:
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
				tickerList[ticker] = ticker_info(line, exchange)
		return tickerList

	# US Treasury N Year Yield from Bloomberg
	def treasury_yield(self):
		url = "http://www.bloomberg.com/markets/rates-bonds/government-bonds/us"
		# "GT10:GOV" : 10-year-treasury bond yield
		print "Download treasury yield"
		response = urllib2.urlopen(url)
		csv = response.read()
		pars = re.findall(r'id\":\"([^,"]+).*?yield\":([^,]*)', csv)
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
		print url
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
	def yahoo_return(self, time_start, time_end, ticker, type):
		# split parameters
		start_y, start_m, start_d = time_start.split("-")
		if len(time_end.split("-")) == 3:
			end_y, end_m, end_d = time_end.split("-")
		else: # until now
			end_y = "2999"; end_m = "12"; end_d = "01"
		if type not in ['d', 'w', 'm']:
			sys.exit("Error, unknown type!")

		ticker_price = {}
		ticker_return = {}

		# Construct url
		url1 = "http://chart.finance.yahoo.com/table.csv?s=" + ticker
		url2 = "&a=" + start_m + "&b=" + start_d + "&c=" + start_y
		url3 = "&d=" + end_m + "&e=" + end_d + "&f=" + end_y
		url4 = "&g=" + type + "&ignore=.csv"

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
			if type == "m":
				date = date[0:7]
			for arg in range(1, len(line)):
				line[arg] = float(line[arg])
			ticker_price[date] = stock_price(line[1:])
		# get the log return of a ticker, sort a dict with 2-dim key by time
		# the hard part is that different company may have different IPO time
		for num, key in enumerate(sorted(ticker_price)):
			if num != 0:
				tmp = []
				tmp.append(log(ticker_price[key].Open / last.Open))
				tmp.append(log(ticker_price[key].High / last.High)) 
				tmp.append(log(ticker_price[key].Low / last.Low))
				tmp.append(log(ticker_price[key].Close / last.Close))
				tmp.append(log(ticker_price[key].Volume / last.Volume))
				tmp.append(log(ticker_price[key].adjClose / last.adjClose))
				ticker_return[key] = stock_price(tmp)
				#print key, ticker_price[key].adjClose, ticker_return[key].adjClose
			last = ticker_price[key]
		return ticker_return
	# return the historical returns for all the tickers, format dict[ticker][date] = return
	def yahoo_all_returns(self, time_start, time_end, type):
		start_y, start_m, start_d = time_start.split("-")
		if len(time_end.split("-")) == 3:
			end_y, end_m, end_d = time_end.split("-")
		else: # until now
			end_y = "2999"; end_m = "12"; end_d = "01"
		if type not in ['d', 'w', 'm']:
			sys.exit("Error, unknown type!")

		tickers_of_all = self.tickers()
		ticker_return = {}
		# enumerate the tickers
		for kth, ticker in enumerate(tickers_of_all.keys()):
		#for kth, ticker in enumerate(['XCOM']):
			print kth, ticker
			if kth > 2:
				break
			ticker_return[ticker] = self.yahoo_return(time_start, time_end, ticker, type)
		return ticker_return		

	# return the historical returns for all the tickers, format: dict[date, ticker] = return
	def yahoo_returns(self, time_start, time_end, type):
		start_y, start_m, start_d = time_start.split("-")
		if len(time_end.split("-")) == 3:
			end_y, end_m, end_d = time_end.split("-")
		else: # until now
			end_y = "2999"; end_m = "12"; end_d = "01"
		if type not in ['d', 'w', 'm']:
			sys.exit("Error, unknown type!")

		tickers_of_all = self.tickers()
		ticker_price = {}
		ticker_return = {}
		# enumerate the tickers
		#for kth, ticker in enumerate(tickers_of_all.keys()):
		for kth, ticker in enumerate(['XCOM']):
			if kth > 2:
				break
			# Construct url
			url1 = "http://chart.finance.yahoo.com/table.csv?s=" + ticker
			url2 = "&a=" + start_m + "&b=" + start_d + "&c=" + start_y
			url3 = "&d=" + end_m + "&e=" + end_d + "&f=" + end_y
			url4 = "&g=m&ignore=.csv"
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
				if type == "m":
					date = date[0:7]
				for arg in range(1, len(line)):
					line[arg] = float(line[arg])
				ticker_price[date, ticker] = stock_price(line[1:])

		# get the log return of a ticker, sort a dict with 2-dim key by time
		# the hard part is that different company may have different IPO time
		tmp = {}; last = {}
		for num, key in enumerate(sorted(ticker_price, key = lambda x : x[0])):
			name = key[1]
			if name in last:
				tmp[name] = []
				tmp[name].append(log(ticker_price[key].Open / last[name].Open))
				tmp[name].append(log(ticker_price[key].High / last[name].High)) 
				tmp[name].append(log(ticker_price[key].Low / last[name].Low))
				tmp[name].append(log(ticker_price[key].Close / last[name].Close))
				tmp[name].append(log(ticker_price[key].Volume / last[name].Volume))
				tmp[name].append(log(ticker_price[key].adjClose / last[name].adjClose))
				ticker_return[key] = stock_price(tmp[name])
				print key, ticker_price[key].adjClose, ticker_return[key].adjClose
			last[name] = ticker_price[key]
		return ticker_return		


def main():
	spider = web_spider()
	#spider.tickers()
	#spider.treasury_yield()
	#spider.FF_five_factor()
	#spider.yahoo_all_returns("2015-01-01", "-",'m')
	#ss = ticker_info([0]*10, "NYSE")
	#ss.key_index("BABA")
	
if __name__ == "__main__":
	sys.exit(main())
	
