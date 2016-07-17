#!/usr/bin/python
import sys
import markowitz
import back_testing

def main():
	time_start = "2011-02-01"
	time_end = "2016-02-01"
	time_end_test = "2016-07-14"
	invest_type = "mid-cap"
	tickers, weight = markowitz.mean_variance_portfolio(time_start, time_end, "m", 5, invest_type)
	back_testing.invest_simulation(tickers, weight, time_end, time_end_test)

if __name__ == "__main__":
	sys.exit(main())
