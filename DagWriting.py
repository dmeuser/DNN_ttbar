import numpy as np

def makeDag(n, subName):
	f = open(subName[:-4]+".dag", "w")
	for i in range(n):
		f.write("JOB A{:02d} ".format(i+1)+subName)
		f.write("\n")
	f.write("\n")
	for i in range(n-1):
		f.write("PARENT A{:02d} CHILD A{:02d}\n".format(i+1, i+2))

if __name__ == "__main__":
	makeDag(50, "BayesOpt.sub")
