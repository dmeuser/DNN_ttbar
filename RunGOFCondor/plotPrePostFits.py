import numpy as np
import argparse
import os
import subprocess as sp
import matplotlib.pyplot as plt
#  ~from utilities.auxiliary import *
#  ~from utilities import style
from ROOT import gROOT, TCanvas, TF1 , TGraph, TPad, THStack, TLegend, gPad, TGraphErrors, TLine, TString, TLatex, TFile
from ROOT import kGray, kAzure, kRed, kGreen, kCyan, kBlue, kBlack, gStyle
import json

def drawPrePostHistos(year, file_path, toPlot, procList, oPath, varName, binnings, varLatex):
	
	for file_struct in toPlot:
		vals = []
		
		ifile = TFile.Open(file_path)
		if (not ifile) or ifile.IsZombie() or ifile.TestBit(TFile.kRecovered): return []
		
		for proc in procList:
			procName = file_struct + proc
			
			iHisto = ifile.Get(procName)
			if not iHisto:
				KILL('get_combine_values -- target TTree object not found in input file: '+file_path+':'+procName)
			vals.append(iHisto)
			#  ~print(type(iHisto))
			#  ~drawHisto
			#  ~iHisto.Draw()
		drawHisto(year, vals, file_struct, procList, oPath, varName, binnings, varLatex)
		#  ~runKStest(year, vals, file_struct, procList, oPath, varName, binnings, varLatex)
		ifile.Close()	

def runKStest(year, histoArr, file_struct, procList, oPath, varName, binnings, varLatex):
	dataP = histoArr[-1]
	histoAllProcs = histoArr[-2]
	
	print(file_struct)
	dataArr, mcArr = [], []
	
	for i in range(dataP.GetNbinsX()):
		dataArr.append(dataP.GetBinContent(i))
		mcArr.append(histoAllProcs.GetBinContent(i))
		if dataP.GetBinContent(i)<10:
			print(dataP.GetBinContent(i), histoAllProcs.GetBinContent(i))
	
	
	dataCumArr = np.cumsum(np.array(dataArr))
	mcCumArr = np.cumsum(np.array(mcArr))
	
	N = mcCumArr[-1]
	valKS = np.max(np.abs(dataCumArr-mcCumArr))/N
	#  ~print(valKS/dataCumArr[-1])
	print(valKS)
	
	p_val_ks = 0
	
	for i in range(1,100):
		p_val_ks += (-1)**(i-1)*np.exp(-2*i**2*N*valKS**2)
		
	
	print(p_val_ks)
	
	
	

def drawHisto(year, histoArr, file_struct, procList, oPath, varName, binnings, varLatex):
	
	n1, n2, ndiv = binnings
	
	stack = THStack("stack", "")
	leg = TLegend(.14,0.85,0.93,0.92)
	leg.SetBorderSize(0)
	leg.SetFillStyle(0)
	leg.SetNColumns(6)
	
	for histo, fillcol, name in zip(histoArr[:-2], [kGreen-7, kAzure-8, kGray+1, kRed-8, kRed-6], procList[:-2]):
		histo.SetFillColor(fillcol)
		#  ~histo.SetAxisRange(-150,150)
		stack.Add(histo)
		leg.AddEntry(histo, name)
	
	minV, maxV = max(histoArr[0].GetMinimum()*0.5, 0.02), stack.GetMaximum()*5
	stack.SetMinimum(0.0001)
	#  ~stack.SetMaximum(stack.GetMaximum()*1.5)
	#  ~stack.GetYaxis().SetTickLength(0.001)
	#  ~stack.GetYaxis().SetTitle("")
	
	dataP = histoArr[-1]
	dataP.SetTitle("")
	dataP.GetYaxis().SetTitle("Events/bin")
	dataP.SetMinimum(minV)
	dataP.SetMaximum(maxV)
	dataP.GetYaxis().SetTitleSize(20)
	dataP.GetYaxis().SetTitleFont(43)
	dataP.GetYaxis().SetTitleOffset(0)
	dataP.GetYaxis().SetTickLength(0.01)
	dataP.SetLineWidth(2)
	leg.AddEntry(dataP, "data")
	

	histoAllProcs = histoArr[-2]
	#  ~histoAllProcs.SetAxisRange(-150,150)
	dataRatio = dataP.Clone("dataRatio")
	stackRatio = histoAllProcs.Clone("stackRatio")
	zeroErrs = np.zeros(stackRatio.GetNbinsX())
	stackRatio.SetError(zeroErrs)
	
	#  ~dataRatio.SetMinimum(0.45)
	#  ~dataRatio.SetMaximum(1.65)
	dataRatio.SetMinimum(0.75)
	dataRatio.SetMaximum(1.33)
	dataRatio.Sumw2()
	dataRatio.Divide(stackRatio)
	#  ~dataRatio.SetAxisRange(-150,150)
	dataRatio.SetTitle("")
	dataRatio.GetXaxis().SetTitle(varLatex+" (GeV)")
	#  ~dataRatio.GetXaxis().SetTitleOffset(1.5)
	dataRatio.GetXaxis().SetTitleSize(0.09)
	dataRatio.GetXaxis().SetTitleOffset(0.95)
	dataRatio.GetXaxis().SetNdivisions(-(ndiv-1))
	
	for i in range(1,ndiv+1,5):
		dataRatio.GetXaxis().ChangeLabel(i, -1,-1,-1,-1,-1,str(n1+(i-1)*(n2-n1)/(ndiv-1)))
	for i in range(1,ndiv+1):
		dataRatio.GetXaxis().ChangeLabel(i, -1,-1,-1,-1,-1," ")
	dataRatio.GetXaxis().SetLabelSize(0.1)
	dataRatio.GetYaxis().SetTitle("data/MC")
	dataRatio.GetYaxis().SetTitleSize(20)
	dataRatio.GetYaxis().SetTitleFont(43)
	dataRatio.GetYaxis().SetTitleOffset(1.55)
	dataRatio.GetYaxis().SetTickLength(0.01)
	dataRatio.GetYaxis().SetLabelSize(0.1)
	dataRatio.GetYaxis().SetNdivisions(505)
	dataRatio.SetLineWidth(2)
	
	errsHisto = TGraphErrors(histoAllProcs)
	errsHisto.SetTitle("")
	#  ~errsRatio.GetXaxis().SetRangeUser(0,30)
	errsHisto.SetFillColor(kGray+3)
	#  ~gStyle.SetHatchesSpacing(0.1)
	errsHisto.SetFillStyle(3154)
	
	histoAllProcs.Divide(stackRatio)
	errsRatio = TGraphErrors(histoAllProcs)
	errsRatio.SetTitle("")
	#  ~errsRatio.GetXaxis().SetRangeUser(0,30)
	errsRatio.SetFillColor(kGray+3)
	errsRatio.SetFillStyle(3154)
	

	
	
	latex = TLatex()
	latex.SetNDC()
	latex.SetTextAngle(0)
	latex.SetTextColor(kBlack)
	latex.SetTextSize(0.035)
	if (file_struct.split("_")[0]=="ee"):
		chanText = "ee"
	elif (file_struct.split("_")[0]=="emu"):
		chanText = "e#mu"
	else: chanText = "#mu#mu"

	#  ~errsRatio2
	
	#Create Canvas
	
	can = TCanvas("c", "canvas", 800, 800)
	#  ~can.SetLeftMargin(90)
	# Upper histogram plot is pad1
	pad1 = TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
	pad1.SetBottomMargin(0)  # joins upper and lower plot
	pad1.SetLeftMargin(0.1)
	pad1.SetLogy()
	pad1.Draw()
	# Lower ratio plot is pad2
	can.cd()  # returns to main canvas before defining pad2
	pad2 = TPad("pad2", "pad2", 0, 0.02, 1, 0.3)
	pad2.SetTopMargin(0)  # joins upper and lower plot
	pad2.SetLeftMargin(0.1)
	pad2.SetBottomMargin(0.2)
	pad2.Draw()
	
	#Upper PLot
	pad1.cd()
	dataP.Draw("P0E1")
	stack.Draw("same hist")
	errsHisto.Draw("same 2")
	#  ~stackSum.Draw()
	dataP.Draw("same P0E0")
	dataP.Draw("same axis")
	leg.Draw()
	l = pad1.GetLeftMargin()
	t = pad1.GetTopMargin()
	r = pad1.GetRightMargin()
	latex.DrawLatex(.845,0.79,chanText+", "+file_struct.split("_")[-1][:-1])
	draw_lumi(year, pad1, False, True)
	
	#Lower Plot
	pad2.cd()
	#  ~errsRatio.Draw()
	dataRatio.Draw("P0E1")
	#  ~errsR
	errsRatio.Draw("same 2")
	dataRatio.Draw("same P0E1")
	dataRatio.Draw("same axis")
	aline = TLine()
	aline.DrawLine(0, 1, dataRatio.GetNbinsX(), 1);
	#  ~can.SaveAs(oPath+varName+"_"+file_struct[:-1]+"_mumu_only.pdf")
	can.SaveAs(oPath+varName+"_"+file_struct[:-1]+".pdf")

def draw_lumi(year, pad, simulation, drawLumiText):
	if year=="2017":
		lumi=41.5
	elif year=="2016_preVFP":
		lumi=19.5
	else:
		lumi=59.8
	cmsText = "CMS"
	extraText = "Private work"
	
	lumiText = str(lumi)+" fb^{-1}"
	sqrtsText= "(13 TeV)"

	cmsTextFont   = 61
	extraTextFont = 52
	lumiTextSize     = 0.6
	lumiTextOffset   = 0.2
	cmsTextSize      = 0.75
	extraOverCmsTextSize  = 0.76

	l = pad.GetLeftMargin()
	t = pad.GetTopMargin()
	r = pad.GetRightMargin()

	pad.cd()

	latex = TLatex()
	latex.SetNDC()
	latex.SetTextAngle(0)
	latex.SetTextColor(kBlack)

	extraTextSize = extraOverCmsTextSize*cmsTextSize

	latex.SetTextFont(42)
	latex.SetTextAlign(31)
	latex.SetTextSize(lumiTextSize*t)
	if (drawLumiText): latex.DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText+" "+sqrtsText)

	latex.SetTextAlign(11)
	latex.SetTextSize(cmsTextSize*t)
	#  ~cmsLabel = "#font[{}]\{{}\}".format(
	cmsLabel = "#font[%f]{%s}" % (cmsTextFont,cmsText)
	#  ~cmsLabel+=TString.Format("#font[%f]{%s}",cmsTextFont,cmsText.Data())
	cmsLabel+= "#color[922]{#scale[%f]{#font[%f]{ %s}}}" % (extraTextSize/float(cmsTextSize),extraTextFont,extraText)
	#  ~cmsLabel+= "#color[922]{#scale[%f]{#font[%f]{ %s}}}" % (extraTextSize/float(cmsTextSize),extraTextFont,"#bullet P"+file_struct.split("_")[-1][1:-1])
	#  ~cmsLabel+= "#color[922]{#scale[%f]{#font[%f]{ %s}}}" % (extraTextSize/float(cmsTextSize),extraTextFont,extraText)

	latex.DrawLatex(l,1-t+lumiTextOffset*t,cmsLabel)

	pad.Update()

def get_xsecPulls(varNames, procs):
	arr = []
	for varName in varNames:
		iPath = "/net/data_cms1b/user/dmeuser/top_analysis/2018/v06/output_framework/datacards/"+varName+"/"
		shapeFileName = "fitDiagnosticsTest.root"
		#  ~print(varName)
		vals = []

		ifile = TFile.Open(iPath+shapeFileName)
		if (not ifile) or ifile.IsZombie() or ifile.TestBit(TFile.kRecovered): print("Error, File not known:", iPath+shapeFileName)

		itree = ifile.Get("tree_fit_sb")
		if not itree:
			KILL("Error, Tree in file not known:", iPath+shapeFileName)
		
		for expr in procs:
			for i_ent in itree:
				val = getattr(i_ent, expr) if isinstance(expr, basestring) else expr(itree.front())
				vals.append(val)
		if np.abs(val)>1.: print(varName)
		arr.append(vals)
		ifile.Close()
	return np.array(arr, dtype=float)
	
def plotPulls(varNames):
	oPath = "/home/home4/institut_1b/dmeuser/top_analysis/DNN_ttbar/RunGOFCondor/"
	procs = ['DY_xsec', 'ST_xsec', 'TTother_xsec', 'other_xsec', 'r']
	procNames = ['Drell-Yan', 'Single t', 'tt other', 'other bkg', 'signal strength']
	
	niceNames = []
	for varName in varNames: niceNames.append("$"+legends[varName]+"$")

	xsec_Pulls = get_xsecPulls(varNames, procs)
	xBinC = np.linspace(0.5, len(procNames)-0.5, len(procNames))
	yBinC = np.linspace(0.5, len(niceNames)-0.5, len(niceNames))

	fig, ax = plt.subplots(1,1, figsize=(4,7))
	mesh1 = ax.pcolormesh(range(len(procs)+1), range(len(varNames)+1), xsec_Pulls, vmin=-1., vmax=1., cmap=plt.get_cmap("seismic"))
	cbar = fig.colorbar(mesh1, ax=ax)
	cbar.set_label("Pulls")
	#  ~cbar.set_cmap(plt.get_cmap("seismic"))
	#  ~cbar.set_clim(vmin=-1., vmax=1.)
	for i,col in enumerate(xsec_Pulls):
		for j,vali in enumerate(col):
			#  ~print(vali)
			if np.abs(vali)>1: formI = 0
			else: formI = 1
			
			if vali>0:
				ax.text(xBinC[j]-0.4, yBinC[i]-0.25, "{:.2f}".format(vali)[formI:], fontsize=12, color="green", fontweight="bold")
			else: 
				ax.text(xBinC[j]-0.4, yBinC[i]-0.25, "-"+"{:.2f}".format(vali)[formI+1:], fontsize=12, color="green", fontweight="bold")
				#  ~print(vali, "-{:.2f}".format(vali)[2:])
	plt.yticks(np.linspace(0.5, len(niceNames)-0.5, len(niceNames)), niceNames, fontsize=9)
	plt.xticks(np.linspace(0.5, len(procNames)-0.5, len(procNames)), procNames, fontsize=9, rotation=20)
	plt.subplots_adjust(left=0.2, right=0.8, top=0.98, bottom=0.1)
	plt.savefig(oPath+"xsec_Pulls.pdf")

def plotPvals():
	pVals=[]
	with open('all_P_vals.txt', 'r') as f:
		for line in f.readlines():
			pVals.append(line[:-1].split(","))
	
	print(pVals)

legends = {
	"METunc_Puppi" : "#sigma_{MET}^{Puppi}",
	"PuppiMET_X" : "p_{x}^{miss, old Puppi}",
	"PuppiMET_Y" : "p_{y}^{miss, old Puppi}",
	"PuppiMET_xy_X" : "p_{x}^{miss, Puppi}",
	"PuppiMET_xy_Y" : "p_{y}^{miss, Puppi}",
	"MET_X" : "p_{x}^{miss, old PF}",
	"MET_Y" : "p_{y}^{miss, old PF}",
	"MET_xy_X" : "p_{x}^{miss, PF}",
	"MET_xy_Y" : "p_{y}^{miss, PF}",
	"CaloMET_X" : "p_{x}^{miss, Calo}",
	"CaloMET_Y" : "p_{y}^{miss, Calo}",
	"CaloMET" : "p_{T}^{miss, Calo}",
	"vecsum_pT_allJet_X" : "p_{x}^{all jets}",
	"vecsum_pT_allJet_Y" : "p_{y}^{all jets}",
	"Jet1_pX" : "p_{x}^{Jet 1}",
	"Jet1_pY" : "p_{y}^{Jet 1}",
	"MHT" : "MHT",
	"mass_l1l2_allJet" : "m_{l1,l2,all jets}",
	"Jet2_pX" : "p_{x}^{Jet 2}",
	"Jet2_pY" : "p_{y}^{Jet 2}",
	"mjj" : "m_{jj}",
	"n_Interactions" : "n_{Interactions}",
	"MT2" : "MT2",
	"dPhiMETleadJet_Puppi" : "#Delta#phi(p_{T}^{miss, Puppi}, Jet 1)",
	"Lep2_pX" : "p_{x}^{l 2}",
	"Lep2_pY" : "p_{y}^{l 2}",
	"HT" : "HT",    
	"Lep1_pX" : "p_{x}^{l 1}",
	"Lep1_pY" : "p_{y}^{l 1}",
	"vecsum_pT_allJet" : "p_{T}^{all jets}",
	"nJets" : "n_{jets}",
	"Jet1_E" : "E_{jet 1}",
	"DeepMET_reso_X" : "p_{x}^{\rm miss, DReso}",
	"DeepMET_reso_Y" : "p_{y}^{\rm miss, DReso}",
	"DeepMET_resp_X" : "p_{x}^{\rm miss, DResp}",
	"DeepMET_resp_Y" : "p_{y}^{\rm miss, DResp}",
   "DNN_MET_pT" : "p^{miss, DNN}"
}

if __name__ == '__main__':
	procList = np.array(["otherBKG", "SingleTop", "DrellYan_comb", "TTbar_other", "TTbar_diLepton", "TotalProcs", "data_obs"])	
	#  ~toPlot = ["ee_prefit/", "emu_prefit/", "mumu_prefit/", "ee_postfit/", "emu_postfit/", "mumu_postfit/"]
	toPlot = ["ee_prefit/", "emu_prefit/", "mumu_prefit/", "ee_postfit/", "emu_postfit/", "mumu_postfit/"]
	year = "2018"
	version = "v08"
	
	#  ~binnings = [(0, 30, 31), (0, 36, 37), (0, 30, 31), (0, 30, 31), (0, 30, 31), (0, 25, 26), (0, 30, 31)]
	#  ~varNames = ["PuppiMET_xy_X_VS_Jet2_pX", "PuppiMET_xy_Y_VS_MET_xy_Y", "PuppiMET_xy_Y_VS_CaloMET", "MET_xy_Y_VS_Jet2_pY", "Jet1_pX_VS_nJets", "Lep1_pY_VS_MT2", "PuppiMET_xy_Y_VS_MT2"]
	#  ~binnings = [(-150, 150, 31)]
	#  ~varNames = ["PuppiMET_xy_Y_VS_MET_xy_Y"]
	#  ~varNames = ["PuppiMET_xy_Y_VS_Lep1_pX"]
	#  ~binnings = [(0, 30, 31)]
	#  ~varNames = ["PuppiMET_xy_X_VS_Jet1_pY"]
	#  ~binnings = [(0, 36, 37)]
	#  ~varNames = ["PuppiMET_xy_X_VS_mjj"]
	#  ~varNames = ["PuppiMET_X", "PuppiMET_Y", "MET_X", "MET_Y", "CaloMET", "vecsum_pT_allJet_X", "vecsum_pT_allJet_Y"]
	#  ~varNames = ["PuppiMET_xy_X", "PuppiMET_xy_Y", "MET_xy_X", "MET_xy_Y", "vecsum_pT_allJet_Y", "vecsum_pT_allJet_X", "mass_l1l2_allJet", "Jet1_pY", "MHT", "Lep1_pX", "Lep1_pY", "Jet1_pX", "CaloMET", "vecsum_pT_allJet", "MT2", "mjj", "nJets", "Jet1_E", "HT"]
	#  ~varNames = ["HT"]
	#  ~binnings = [(-150, 150, 31)]
	#  ~binnings = [(-400, 400, 41)]
	#  ~varNames = ["Jet1_pX"]
	#  ~varNames = ["HT", "nJets", "METunc_Puppi", "MT2"]
	#  ~varNames = ["MET_xy_X", "MET_xy_Y", "PuppiMET_xy_X", "PuppiMET_xy_Y"]
	#  ~varNames = ["MET_xy_Y"]
	#  ~varNames = ["n_Interactions"]
	#  ~binnings = [(-150, 150, 31), (-150, 150, 31), (-150, 150, 31), (-150, 150, 31)]
	#  ~binnings = [(50, 1200, 24), (1.5, 8.5, 8), (0, 40, 21), (0, 160, 41)]
	#  ~binnings = [(-250, 250, 26), (0, 200, 41), (0, 25, 26)]
	#  ~varNames = ["Lep1_pX", "CaloMET", "Lep1_pX_VS_CaloMET"]

	#  ~binnings = [(0, 25, 26), (0, 36, 37), (-400, 400, 41), (40, 500, 24)]
	#  ~varNames = ["Jet1_pY_VS_Jet1_E", "MET_xy_Y_VS_vecsum_pT_allJet_X", "Jet1_pY", "Jet1_E"]
	
	#  ~varNames = ["DeepMET_reso_X"]
	#  ~varNames = ["PuppiMET_xy_X"]
	#  ~binnings = [(-150, 150, 31)]
	#  ~varNames = ["DNN_MET_pT"]
	#  ~binnings = [(0, 500, 20)]
	varNames = ["PuppiMET_xy_X_VS_Jet1_E"]
	binnings = [ (0, 29, 30)]
	
	for varName, binning in zip(varNames, binnings):
		iPath = "/net/data_cms1b/user/dmeuser/top_analysis/"+year+"/"+version+"/output_framework/datacards/"+varName+"/"
		oPath = "/home/home4/institut_1b/dmeuser/top_analysis/DNN_ttbar/RunGOFCondor/CondorGOFsubmits/"+year+"/"+varName+"_files/Results_"+varName+"/"
		shapeFileName = "FitDiaShapes_"+varName+".root"
		produceShapes = True
		
		if "_VS_" in varName:
			legName = legends[varName.split("_VS_")[0]]+" vs "+legends[varName.split("_VS_")[1]]
		else:
			legName = legends[varName]
		
		if produceShapes:
			t2wCMD = "text2workspace.py "+iPath+"ttbar_"+varName+".txt -o "+iPath+"ttbar_"+varName+".root"
			t2w = sp.check_output(t2wCMD.split())
			print(t2wCMD)
			print(t2w)
			
			#  ~fitDiaCMD = "combine -M FitDiagnostics ttbar_"+varName+".root --cminDefaultMinimizerStrategy 0 --cminDefaultMinimizerTolerance 50.1 --setParameters r=1 --setParameterRanges r=-2.,2. -m 125 --robustFit 1 --verbose 3 --X-rtd MINIMIZER_MaxCalls=999999999 --cminDefaultMinimizerPrecision 1E-12"
			fitDiaCMD = "combine -M FitDiagnostics ttbar_"+varName+".root --robustFit 1 --cminDefaultMinimizerStrategy 0  --setParameters r=1 --setParameterRanges r=-2.,2. --X-rtd MINIMIZER_MaxCalls=999999999 --cminDefaultMinimizerPrecision 1E-12 --verbose 3"
			fitDia = sp.check_output(fitDiaCMD.split(), cwd=iPath)
			print(iPath, fitDiaCMD)
			print(fitDia)
			
			ShapeFromW = sp.check_output(["PostFitShapesFromWorkspace", "-w", iPath+"ttbar_"+varName+".root", "-o", iPath+shapeFileName, "--postfit", "-f", iPath+"fitDiagnosticsTest.root:fit_s", "--samples", "2000"])
			print(ShapeFromW)
		
		drawPrePostHistos(year, iPath+shapeFileName, toPlot, procList, oPath, varName, binning, legName)
	plotPulls(varNames)
	#  ~plotPvals()
