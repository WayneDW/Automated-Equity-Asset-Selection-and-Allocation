#!/usr/bin/python
import sys
import markowitz
import back_testing
import const_markowitz

def model():
	time_start = "2010-01-01"
	time_end = "2015-01-01"
	time_end_test = "2015-06-01"
	invest_type = "mid-cap"

	solver = const_markowitz.const_markowitz(time_start, time_end, "m", 5, invest_type)
	tickers, weight = solver.const_optimization(0, 1)
	#tickers, weight = markowitz.mean_variance_portfolio(time_start, time_end, "m", 5, invest_type)
	back_testing.invest_simulation(tickers, weight, time_end, time_end_test)

def main():
	model()

if __name__ == "__main__":
	sys.exit(main())
