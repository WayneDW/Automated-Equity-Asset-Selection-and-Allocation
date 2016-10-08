#!/usr/bin/python
import sys
import numpy as np

import util
import class_crawler


def invest_simulation(tickers, weight, time_start, time_end):
	# create object web_spider to fetch data
	spider = class_crawler.web_spider()	
	try: 
		returns = spider.yahoo_price(tickers[0], time_start, time_end, 'd')
	except:
		sys.exit("Http error")
	dates = sorted(returns.keys())
	buy_hold_return = []

	# get the starting and ending time in this file
	date_start = dates[0]
	date_end = dates[len(dates) - 1]
	buy_hold_return.append(returns[date_end].adjClose / returns[date_start].adjClose - 1)
	
	for ticker in tickers[1:]:
		try:
			returns = spider.yahoo_price(ticker, time_start, time_end, 'd')
		except:
			sys.exit("Http error")
		buy_hold_return.append(returns[date_end].adjClose / returns[date_start].adjClose - 1)
	realized_return = np.dot(weight, buy_hold_return)

	# calc the market return, namely S&P 500
	market_prices = spider.yahoo_price("^GSPC", time_start, time_end, 'd')
	market_return = market_prices[date_end].adjClose / market_prices[date_start].adjClose - 1

	
	print " Realized return:", util.perc(realized_return, 2), "from", time_start, "to", time_end
	print "Equally weighted:", util.perc(np.mean(buy_hold_return), 2)
	print "  S&P 500 return:", util.perc(market_return, 2)
	return realized_return



def main():
	invest_simulation(["BABA"], [1], "2016-05-30", "2016-06-30")


if __name__ == "__main__":
	sys.exit(main())
