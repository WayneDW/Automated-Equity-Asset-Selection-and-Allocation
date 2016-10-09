#!/usr/bin/python
import sys

# date available is above 30 (monthly level)
# Size, small, mid or large-cap
# B/M, growing 
# leverage ratio

def init_check(model, invest_type):
	ticker = model.ticker
	returns = model.returns
	ticker_info = model.ticker_info
	if sample_size(ticker, returns): return 0
	#if capSize(ticker, ticker_info.marketCap, invest_type): return 0
	return 1

def mid_check(obj):
	print "=============================================================================="
	if alpha_neg(obj): return 0
	# PE check
	# if 200 moving avg > 50 moving avg check
	# Price_to_Book check
	return 1




def sample_size(ticker, returns):
	if len(returns) < 30:
		return warining(ticker, "has insufficient data")
	return 0

# small (<2B), mid (2-10B), large-cap (>10B): 6093, 887, 428
# http://www.fool.com/investing/general/2014/08/05/mid-cap-stocks-investing-essentials.aspx
def capSize(ticker, par, target):
	if par < 2 * 10 ** 9:
		out = "small-cap"
	elif par < 10 ** 10:
		out = "mid-cap"
	else:
		out = "large-cap"
	if out != target:
		return warining(ticker, "is a " + out +  " company, rather than a " + target)
	return 0

def alpha_neg(obj):
	if obj.alpha < 0:
		return warining(obj.ticker_info.ticker, "has negative alpha")
	return 0

def warining(ticker, s):
	print ticker, s, "!"
	print "=============================================================================="
	return 1




