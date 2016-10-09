#!/usr/bin/python

def perc(num, tag):
	num = str(round(float(num) * 100, tag))
	return num + "%"

def array_perc(arr, tag):
	out = []
	for num in arr:
		out.append(perc(num, tag))
	return out
