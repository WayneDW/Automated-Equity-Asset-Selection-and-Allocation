#!/usr/bin/python

import sys
import urllib2
import re
import os
import time
import urllib
import urllib2
'''
import requests
import zipfile
import StringIO
'''
import random
from math import log
# yahoo finance api, containing incomplete index
from yahoo_finance import Share 
from pprint import pprint




class stock_info_collector:
	# collect stock information, eg, price, volume, sector, ...
	def __init__(self, time_start, time_end, time_type):
		self.time_start, self.time_end, self.time_type = time_start, time_end, time_type
		# get ticker list from NASDAQ
		tickers = self.getTickers()
		self.Basic, self.Balance, self.Return = {}, {}, {}

		for num, ticker in enumerate(tickers):
			if num > 5:
				continue
			tag, times = 0, 1
			while not tag and times <= 5: # repeat download X times
				try:
					print num, ticker
					self.Basic[ticker] = tickers[ticker]
					self.Balance[ticker] = self.BalanceSheet(ticker)
					self.Return[ticker], tag = self.yahoo_return(ticker)
				except:
					print num, ticker, "Http error, wait some time!"
					time.sleep(times)
				times += 1
			time.sleep(times)


		#print tickers["SPP"].name, tickers["SPP"].BS.PE, tickers["SPP"].BS.moving_avg_50
	# format the data from the ticker list in NASDAQ
	class Basic():
		def __init__(self, name, lastSale, marketCap, IPO, sector, industry, ex):
			self.name, self.lastSale, self.marketCap,  = name, lastSale, float(marketCap)
			self.IPO, self.sector, self.industry, self.exchange = IPO, sector, industry, ex
			
	class Price:
		def __init__(self, pars):
			self.Open, self.High, self.Low, self.Close, self.Volume, self.adjClose = pars

	class BalanceSheet: # Balance Sheet Factors
		def __init__(self, ticker):
			# Api from https://pypi.python.org/pypi/yahoo-finance/1.2.1
			info = Share(ticker)
			self.EBITDA = str(info.get_ebitda())
			self.PE = str(info.get_earnings_share())
			self.PE_growth = str(info.get_price_earnings_growth_ratio())
			self.price_book = str(info.get_price_book())
			self.moving_avg_50 = str(info.get_50day_moving_avg())
			self.moving_avg_200 = str(info.get_200day_moving_avg())
			self.price_sales = str(info.get_price_sales())
			self.bookValue = str(info.get_book_value())
			self.marketCap = str(info.get_market_cap())
			self.dividend_share = str(info.get_dividend_share())
			self.dividend_yield = str(info.get_dividend_yield())
			self.short_ratio = str(info.get_short_ratio())

	# Get a complete ticker list belong to NYSE, NASDAQ and AMEX from NASDAQ website
	# http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download
	# Basic attributes: ticker, name, lastSale, MarketCap, IPOyear, sector, industry
	def getTickers(self):
		url1 = "http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange="
		url2 = "&render=download"
		tickerBasics = {}
		for exchange in ["NASDAQ", "NYSE", "AMEX"]:
			print "Download tickers from " + exchange
			response = urllib2.urlopen(url1 + exchange + url2)
			csv = response.read().split('\n')
			for num, line in enumerate(csv):
				line = line.strip().strip('"').split('","')
				if num == 0 or len(line) != 9: continue
				ticker, name, lastSale, MarketCap, IPOyear, sector, industry = line[0: 4] + line[5: 8]
				tickerBasics[ticker] = self.Basic(name, lastSale, MarketCap, IPOyear, sector, industry, exchange)
		return tickerBasics
				
	# Return the historical stock price, date type -> monthly: m; weekly: w; daily: d
	# http://chart.finance.yahoo.com/table.csv?s=BIDU&a=5&b=12&c=2016&d=6&e=12&f=2016&g=m&ignore=.csv
	def yahoo_price(self, ticker):
		# split date parameters
		start_y, start_m, start_d = self.time_start.split("-")
		if len(self.time_end.split("-")) == 3: end_y, end_m, end_d = self.time_end.split("-")
		else: end_y = "2999"; end_m = "12"; end_d = "01" # until now

		if self.time_type not in ['d', 'w', 'm']: sys.exit("Error, unknown type!")
		start_m, end_m = str(int(start_m) - 1), str(int(end_m) - 1)
	
		# Construct url
		url1 = "http://chart.finance.yahoo.com/table.csv?s=" + ticker
		url2 = "&a=" + start_m + "&b=" + start_d + "&c=" + start_y
		url3 = "&d=" + end_m + "&e=" + end_d + "&f=" + end_y + "&g=" + self.time_type + "&ignore=.csv"

		# parse url
		response = urllib2.urlopen(url1 + url2 + url3)
		csv = response.read().split('\n')
		# get historical price
		ticker_price = {}
		for num, line in enumerate(csv):
			line = line.strip().split(',')
			if len(line) < 7 or num == 0: continue
			date = line[0]
			# check if the date type matched with the standard type
			if not re.search(r'^[12]\d{3}-[01]\d-[0123]\d$', date): continue
			# tailor the date if this is a month type
			#if time_type == "m": date = date[0:7]
			for arg in range(1, len(line)):
				line[arg] = float(line[arg])
			ticker_price[date] = self.Price(line[1:])
		return ticker_price
	def yahoo_return(self, ticker):
		ticker_price = self.yahoo_price(ticker)
		# get the return of a ticker, sort a dict with 2-dim key by time
		# the hard part is that different company may have different IPO time
		ticker_return = {}
		for num, date in enumerate(sorted(ticker_price)):
			if num != 0:
				tmp = []
				tmp.append(ticker_price[date].Open / last.Open - 1)
				tmp.append(ticker_price[date].High / last.High - 1)
				tmp.append(ticker_price[date].Low / last.Low - 1)
				tmp.append(ticker_price[date].Close / last.Close - 1)
				tmp.append(ticker_price[date].Volume / last.Volume - 1)
				tmp.append(ticker_price[date].adjClose / last.adjClose - 1)
				ticker_return[date] = self.Price(tmp)
				#print date, ticker_price[date].adjClose, ticker_return[date].adjClose
			last = ticker_price[date]
		return ticker_return, 1

# save data to local, so that we don't need to crawl data every time we run our model
def saveLocal(time_start, time_end, time_type):
	# get the ticker list from NASDAQ
	cl = stock_info_collector(time_start, time_end, time_type)
	file_time = time_start + "_" + time_end + "_" + time_type

	# delete directory if it already exists
	if os.path.isdir("./dat/" + file_time):
		sys.exit("Directory already existed")
	else:
		os.system("mkdir " + "./dat/" + file_time)

	for ticker in cl.Basic:
		if not os.path.isdir("./dat/" + file_time + "/" + ticker[0]):
			os.system("mkdir " + "./dat/" + file_time + "/" + ticker[0])
		if ticker not in cl.Balance or ticker not in cl.Return: continue
		f = open("./dat/" + file_time + "/" + ticker[0] + "/" + ticker, 'w')
		f.write("{ticker:" + ticker + "} ")
		f.write("{name:" + cl.Basic[ticker].name + "} ")
		f.write("{lastSale:" + cl.Basic[ticker].lastSale + "} ")
		f.write("{marketCap:" + str(cl.Basic[ticker].marketCap) + "} ")
		f.write("{IPO:" + cl.Basic[ticker].IPO + "} ")
		f.write("{sector:" + cl.Basic[ticker].sector + "} ")
		f.write("{industry:" + cl.Basic[ticker].industry + "} ")
		f.write("{exchange:" + cl.Basic[ticker].exchange + "} ")	
		f.write("{EBITDA:" + cl.Balance[ticker].EBITDA + "} ")
		f.write("{PE:" + cl.Balance[ticker].PE + "} ")
		f.write("{PE_growth:" + cl.Balance[ticker].PE_growth + "} ")
		f.write("{price_book:" + str(cl.Balance[ticker].price_book) + "} ")
		f.write("{moving_avg_50:" + cl.Balance[ticker].moving_avg_50 + "} ")
		f.write("{moving_avg_200:" + cl.Balance[ticker].moving_avg_200 + "} ")
		f.write("{price_sales:" + cl.Balance[ticker].price_sales + "} ")
		f.write("{bookValue:" + cl.Balance[ticker].bookValue + "} ")
		f.write("{marketCap:" + cl.Balance[ticker].marketCap + "} ")
		f.write("{dividend_share:" + cl.Balance[ticker].dividend_share + "} ")
		f.write("{dividend_yield:" + cl.Balance[ticker].dividend_yield + "} ")
		f.write("{short_ratio:" + cl.Balance[ticker].short_ratio + "} " + "\n")
		for date in sorted(cl.Return[ticker]):
			f.write("{date_" + date + ":"  + str(cl.Return[ticker][date].adjClose) + "} " + "\n")




def main():
	saveLocal("2000-01-01","2016-09-04", "d")

if __name__ == "__main__":
	sys.exit(main())



