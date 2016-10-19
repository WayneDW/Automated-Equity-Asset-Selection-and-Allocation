#!/usr/bin/python

import sys
import urllib2
import re
import os
import time
import urllib
import urllib2

import random
from math import log

## next update: continue to download failed tickers
## next update: add column attribute to the 1st row
## next update: add industry, sector, ... as feature

# collect stock information, eg, price, volume, sector, key_ratios, etc.
class stock_info_collector:
	def __init__(self, d0, d1, dtype):
		self.time_start, self.time_end, self.time_type = d0, d1, dtype
		# if dir doesn't exist, create one
		if not os.path.isdir("./input/"): os.system("mkdir " + "./input/")
		# if file existed, exit
		fileName = "_".join(["raw", self.time_start, self.time_end, self.time_type])
		if os.path.isfile("./input/" + fileName): sys.exit("file already existed!")

		self.failedList, cnt = {}, 1
		# get basic indicators associated with the ticker
		for exchange in ["NASDAQ", "NYSE", "AMEX"]:
			print "Download tickers from " + exchange
			u0 = "http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange="
			u1 = "&render=download"
			response = urllib2.urlopen(u0 + exchange + u1)
			csv = response.read().split('\n')
			for num, line in enumerate(csv):
				line = line.strip().strip('"').split('","')
				if num == 0 or len(line) != 9: continue # filter unmatched format
				#if cnt > 10: continue # test purpose
				ticker, name, lastSale, MarketCap, IPOyear, sector, \
				industry = line[0: 4] + line[5: 8]
				tag, priceStr, ratioStr = self.repeatDownload(ticker)

				if tag == 0: 
					self.failedList[ticker] = [ticker, name, lastSale, MarketCap, \
					IPOyear, sector, industry]
					print ticker, "failed"; continue
				print ticker, cnt
				self.writeFile(fileName, ticker, name, lastSale, MarketCap, \
					IPOyear, sector, industry, ratioStr, priceStr)
				cnt += 1
		self.downloadFailed()
	
	def writeFile(self, fileName, ticker, name, lastSale, MarketCap, \
			IPOyear, sector, industry, ratioStr, priceStr):
		fout = open("./input/" + fileName, 'a+')
		#print ratioStr, priceStr
		fout.write( "{ticker:" + ticker + "," + "name:" + name + "," \
		"lastSale:" + lastSale + "," + "MarketCap:" + MarketCap + "," \
		"IPOyear:" + IPOyear + "," + "sector:" + sector + "," + "industry:" + \
		industry + ',\n' + ratioStr + "\n" + priceStr + "}\n") 
		fout.close()

	def repeatDownload(self, ticker):
		repeat_times = 3 # repeat download for N times
		priceStr, ratioStr = "", ""
		for _ in range(repeat_times): 
			try:
				time.sleep(random.uniform(2, 5))
				priceStr = self.yahoo_price(ticker)
				ratioStr = self.key_ratios(ticker)
				if priceStr and ratioStr: # skip loop if data fetched
					break
			except:
				if _ == 0: print ticker, "Http error!"
		#tag = 0 if _ == repeat_times - 1 else 0 # decide if skip this loop
		tag = 0 if len(priceStr) == 0 or len(ratioStr) == 0 else 1
		return tag, priceStr, ratioStr

	def downloadFailed(self):
		print "start downloading failed ticker list", self.failedList.keys()
		for num, ticker in enumerate(self.failedList):
			tag, priceStr, ratioStr = self.repeatDownload(ticker)
			print "2nd", num, ticker, tag
			if tag == 0: continue
			
			ticker, name, lastSale, MarketCap, IPOyear, \
				sector, industry = self.failedList[ticker]
			self.writeFile(self, fileName, ticker, name, lastSale, MarketCap, \
				IPOyear, sector, industry, ratioStr, priceStr)

	# Classical indicators for fundamental analysis
	# Data source: http://www.morningstar.com/
	def key_ratios(self, ticker):
		u0 = "http://financials.morningstar.com/ajax/exportKR2CSV.html?&callback=?&t="
		u1 = "&region=usa&culture=en-US&cur=&order=asc"
		response = urllib2.urlopen(u0 + ticker + u1)
		csv = response.read().split('\n')
		dt = ["Revenue %", "Operating Income %", "Net Income %", "EPS %"]
		standard_output, tag = "", ""
		for num, lines in enumerate(csv):
			m, name =  re.match(r"^([\w\d %*&/\-\\']+|),", lines), "" # parse attribute
			if m: name =  m.group(1)
			elif lines in dt: tag = lines + "_"; continue # add tag to N-Year_Average data
			elif lines == "Key Ratios -> Cash Flow": tag = ""; continue
			else: continue
			if num == 2: name = "Time"
			res =  re.findall(r",(\"\-?\d+[,.]?\d*\"|\-?\d+\.?\d*|\w+|)", lines) # parse values  
			string, if_skip_row = [], 0
			for item in res:
				item = item.replace(',', '').replace('"', '')
				if item == "":
					item = "nan" # label missing value as numpy None
				if item in ["TTM", "Latest"] and num != 2: 
					if_skip_row = 1 # tag redundant title rows
				string.append(item)
			if if_skip_row: continue # delete redundant rows
			name = tag.replace(" ", "_") + name.replace(" ", "_")
			#print ticker, name, ",".join(string)
			standard_output += "key_ratios_" + name.replace("'", "") \
				+ ":" + " ".join(string) + ",\n"
		return standard_output [:len(standard_output) - 1] # delete last char in the end 

	# Return the historical stock price, date type -> monthly: m; weekly: w; daily: d
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
		url3 = "&d=" + end_m + "&e=" + end_d + "&f=" + end_y + \
			"&g=" + self.time_type + "&ignore=.csv"

		# parse url
		response = urllib2.urlopen(url1 + url2 + url3)
		csv = response.read().split('\n')
		# get historical price
		ticker_price = {}
		res_str = ""
		for num, line in enumerate(csv):
			line = line.strip().split(',')
			if len(line) < 7 or num == 0: continue
			date = line[0]
			# check if the date type matched with the standard type
			if not re.search(r'^[12]\d{3}-[01]\d-[0123]\d$', date): continue
			# tailor the date if this is a month type
			if self.time_type == "m": date = date[0:7]
			# open, high, low, close, volume, adjClose : 1,2,3,4,5,6
			res_str += date + "_Volume:" + line[5] + ',' + date + "_adjClose:" + line[6] + ",\n"
		return res_str[:len(res_str) - 2] # delete last 2 char in the end of string


if __name__ == "__main__":
	s = stock_info_collector("2000-01-01", "2016-12-31", 'd')
