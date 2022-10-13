#!/usr/bin/env python
# import argparse, os, json, ROOT
import argparse

# from tools.common import *
from utilities.auxiliary import *
from utilities import style
import numpy as np
from ROOT import TLatex, kBlack
import json

def get_combine_values(file_path, tree_name, expr):

    vals = []

    ifile = ROOT.TFile.Open(file_path)
    if (not ifile) or ifile.IsZombie() or ifile.TestBit(ROOT.TFile.kRecovered): return []

    itree = ifile.Get(tree_name)
    if not itree:
       KILL('get_combine_values -- target TTree object not found in input file: '+file_path+':'+tree_name)

    for i_ent in itree:
        val = getattr(i_ent, expr) if isinstance(expr, basestring) else expr(i_ent)
        vals.append(val)

    ifile.Close()

    return vals

#### main
# if __name__ == '__main__':
#     ### args
#     parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
#
#     parser.add_argument('-d', '--data', dest='data', action='store', default=None, required=True,
#                         help='path to input file for Data (output of "combine -M GoodnessOfFit")')
#
#     parser.add_argument('-t', '--toys', dest='toys', nargs='+', default=[], required=True,
#                         help='list of paths to input files for MC-Toys (output of "combine -M GoodnessOfFit")')
#
#     parser.add_argument('-o', '--output-plot-basename', dest='output_plot_basename', action='store', default=None, required=True,
#                         help='basename of plot output files (without extension)')
#
#     # parser.add_argument('-j', '--output-json', dest='output_json', action='store', default=None,
#     #                     help='path to output .json file')
#
#     # parser.add_argument('-e', '--exts', dest='exts', nargs='+', default=['pdf'],
#     #                     help='list of extensions for output file(s)')
#
#     # parser.add_argument('--title', dest='title', action='store', default='',
#     #                     help='text for plot title [format: "title;x-title;y-title"]')
#
#     parser.add_argument('-L', '--txtTL', dest='txtTL', action='store', default='',
#                         help='text for top-left corner of the plot')
#
#     parser.add_argument('-R', '--txtTR', dest='txtTR', action='store', default='',
#                         help='text for top-right corner of the plot')
#
#     # parser.add_argument('--chi2-fit', dest='chi2_fit', action='store_true', default=True,
#     #                     help='fit MC-Toys distribution with chi^2 function')
#
#     # parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,
#     #                     help='enable verbose mode')
#
#     opts, opts_unknown = parser.parse_known_args()
#     ###

def plotGOF(data, toysArr, output_plot_basename, txtTL="", txtTR="", varName="", year="2018"):
    ### conf
    log_prx = os.path.basename(__file__)+' -- '

    if not os.path.isfile(data):
       KILL(log_prx+'invalid path to input file for Data [-d]: '+data)

    # ext_ls = sorted(list(set(exts)))
    ext_ls = sorted(list(set(['pdf','png'])))

    if len(ext_ls) == 0:
       KILL(log_prx+'empty list of extensions for output file(s) [-e]')

    ROOT.gROOT.SetBatch()
    ### -------------------

    ### raw values
    data_vals = get_combine_values(data, 'limit', 'limit')
    if len(data_vals) != 1: KILL(log_prx+'logic error: more than one value found in DATA input: '+str(data_vals))

    DATA = data_vals[0]
    print(DATA)

    TOYS = []

    # for i_toyf in toys:
    #
    #     if not os.path.isfile(i_toyf):
    #        WARNING(log_prx+'invalid path to input path for MC-Toys: '+i_toyf)
    #        continue
    #
    #     TOYS += get_combine_values(i_toyf, 'limit', 'limit')
    
    for toys in toysArr:
        if not os.path.isfile(toys):
           WARNING(log_prx+'invalid path to input path for MC-Toys: '+toys)
        TOYS += get_combine_values(toys, 'limit', 'limit')
    
    
    if len(TOYS) == 0: KILL(log_prx+'logic error: no values found in TOYS input')
    
    #  ~for toy in TOYS:
        #  ~if toy>100: print(toy)

    TOYS_gtrDATA = [toy_v for toy_v in TOYS if toy_v > DATA]

    P_VALUE = float(len(TOYS_gtrDATA)) / float(len(TOYS))
    ### -------------------

    ### output plot
    output_plot_basename = (output_plot_basename if output_plot_basename else 'GoF')

    out_plot_files = []
    for ext in ext_ls:
        out_plot_f = os.path.abspath(output_plot_basename)+'.'+ext
        # if os.path.isfile(out_plot_f):
        #    WARNING(log_prx+'target output file exists already: '+out_plot_f)
        #    continue
        out_plot_files.append(out_plot_f)

    if len(out_plot_files) == 0:
       raise SystemExit(1)

    ### output json
    # output_json = (output_json if output_json else 'GoF.json')
    # output_json = os.path.abspath(output_json)
    # if os.path.isfile(output_json):
    #    KILL(log_prx+'target output file exists already: '+output_json)

    output_plot_dir = os.path.dirname(os.path.abspath(output_plot_basename))
    if not os.path.isdir(output_plot_dir): EXE('mkdir -p '+output_plot_dir)

    # output_json_dir = os.path.dirname(os.path.abspath(output_json))
    # if not os.path.isdir(output_json_dir): EXE('mkdir -p '+output_json_dir)

    # output_json_dict = { 'Data': DATA, 'Toys': sorted(TOYS), 'p_value': P_VALUE, }
    # json.dump(output_json_dict, open(output_json, 'w'), sort_keys=True, indent=2)

    ### output plot
    canvas_name = os.path.basename(output_plot_basename)

    cgof = ROOT.TCanvas(canvas_name, canvas_name, 900, 800)

    L, R, T, B = 0.150, 0.035, 0.050, 0.150

    cgof.SetLeftMargin  (L)
    cgof.SetRightMargin (R)
    cgof.SetTopMargin   (T)
    cgof.SetBottomMargin(B)

    ROOT.TGaxis.SetMaxDigits(4)
    ROOT.TGaxis.SetExponentOffset(-L+.50*L, 0.03, 'y')

    binN = int(float(len(TOYS)) / 10.)
    #  ~print(binN)
    #  ~print(len(TOYS))

    xmin0 = max(0, min(min(TOYS + [DATA]) * 0.90 , min(TOYS + [DATA]) - 5))
    xmax0 = max(TOYS + [DATA]) * 1.10

    if xmax0>3.*np.mean(TOYS + [DATA]):
        xmax0 = np.mean(TOYS + [DATA])*1.8
        
    binW0 = (xmax0-xmin0) / binN

    binW = (DATA-xmin0)
    if round(binW / binW0) > 0: binW /= round(binW / binW0)

    xmin = xmin0
    xmax = xmin + binN*binW

    # if verbose:
    # print 'binN', binN
    # print 'xmin0', xmin0906592832

    # print 'xmax0', xmax0
    # print 'xmin', xmin
    # print 'xmax', xmax
    # print 'binW', binW

    toys_h1 = ROOT.TH1F(canvas_name+'_toys_h1', canvas_name+'_toys_h1', binN, xmin, xmax)
    toys_h2 = ROOT.TH1F(canvas_name+'_toys_h2', canvas_name+'_toys_h2', binN, xmin, xmax)

    toys_h1.SetStats(0)
    toys_h2.SetStats(0)

    toys_h1.Sumw2(0)
    toys_h2.Sumw2(0)

    toys_h1.SetBinErrorOption(ROOT.TH1.kPoisson)
    toys_h2.SetBinErrorOption(ROOT.TH1.kPoisson)

    for toy_v in TOYS        : toys_h1.Fill(toy_v)
    for toy_v in TOYS_gtrDATA: toys_h2.Fill(toy_v)

    toys_h1.SetMarkerStyle(20)
    toys_h1.SetMarkerSize(1.5)
    toys_h1.SetMarkerColor(1)
    toys_h1.SetLineColor(1)
    toys_h1.SetLineWidth(2)
    toys_h1.SetLineStyle(1)
    toys_h1.GetXaxis().SetRangeUser(-3., xmax)

    toys_h2.SetLineColor(ROOT.kViolet-14)
    toys_h2.SetFillColor(ROOT.kViolet-14)

    cgof.cd()
    toys_h1.Draw('lep')
    toys_h2.Draw('hist,same')

    ymin = 1e-4
    ymax = toys_h1.GetMaximum() / 0.7

    data_li = ROOT.TLine(DATA, ymin, DATA, ymax)
    data_li.SetLineColor(ROOT.kBlue+1)
    data_li.SetLineWidth(2)
    data_li.SetLineStyle(1)
    data_li.Draw('same')

    chi2_func = None
    # if chi2_fit:
    if True:

       chi2_func = ROOT.TF1('chi2_func', '[0]*ROOT::Math::chisquared_pdf(x,[1])', xmin, xmax)
       chi2_func.SetLineColor(ROOT.kRed)
       chi2_func.SetFillColor(ROOT.kRed)
       chi2_func.SetParameter(0, toys_h1.Integral())
       chi2_func.SetParameter(1, toys_h1.GetMean(1))

       chi2_fit = toys_h1.Fit(chi2_func, 'mlerso')


       chi2_func.SetLineWidth(2)
       chi2_func.Draw('same')

    toys_h1.Draw('lep,same')
    toys_h1.Draw('axis,same')

    cgof.Update()
    # toys_h1.SetTitle(title)
    #  ~toys_h1.SetTitle(';GoF (saturated);Number of MC Toys;')
    toys_h1.GetXaxis().SetTitle("#lambda_{sat} ("+txtTR+")")
    toys_h1.GetYaxis().SetTitle("Number of MC Toys")
    toys_h1.SetTitle("")
    toys_h1.GetXaxis().SetLabelSize  (0.045)
    toys_h1.GetYaxis().SetLabelSize  (0.045)
    toys_h1.GetXaxis().SetTitleSize  (0.055)
    toys_h1.GetYaxis().SetTitleSize  (0.055)
    toys_h1.GetXaxis().SetTitleOffset(1.05)
    toys_h1.GetYaxis().SetTitleOffset(1.20)
    toys_h1.GetYaxis().SetRangeUser(ymin, ymax)

    legH = (0.28 if chi2_func else 0.21)

    leg = ROOT.TLegend(L+(1-R-L)*.550, B+(1-T-B)*(.975-legH), L+(1-R-L)*.975, B+(1-T-B)*.975)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.AddEntry(toys_h1, 'toy data', 'lep')
    leg.AddEntry(data_li, '#lambda_{sat}^{data}'+' = {:.2f}'.format(DATA)   , 'l')
    leg.AddEntry(toys_h2, 'p-value' +' = {:.2f}'.format(P_VALUE), 'f')

    if chi2_func:
       chi2_func_str  = '#chi^{2} fit, n_{dof} = '
       chi2_func_str += '{:.1f} #pm {:.1f}'.format(chi2_func.GetParameter(1), chi2_func.GetParError(1))
       leg.AddEntry(chi2_func, chi2_func_str, 'l')

    leg.Draw('same')

    #  ~txtTL = ROOT.TLatex(L+(1-R-L)*0.00, (1-T)+T*.25, txtTL)
    #  ~txtTL.SetTextAlign(11)
    #  ~txtTL.SetTextSize(0.055)
    #  ~txtTL.SetTextFont(42)
    #  ~txtTL.SetNDC()
    #  ~txtTL.Draw('same')

    #  ~txtTR = ROOT.TLatex(L+(1-R-L)*1.00, (1-T)+T*.25, txtTR)
    #  ~txtTR.SetTextAlign(31)
    #  ~txtTR.SetTextSize(0.055)
    #  ~txtTR.SetTextFont(42)
    #  ~txtTR.SetNDC()
    #  ~txtTR.Draw('same')
    draw_lumi(cgof, False, True, year)
    

    #  ~with open('all_P_vals.txt', 'a') as f:
        #  ~f.write(varName+",{},{},{}\n".format(P_VALUE, DATA, chi2_func.GetParameter(1)))
    
    for _tmp in out_plot_files:

        # if os.path.exists(_tmp):
        #    WARNING(func_prx+'target output file already exists, will not be overwritten: '+_tmp)
        #    continue

        cgof.SaveAs(_tmp)

    cgof.Close()
    ### --------------------------------------------

def draw_lumi(pad, simulation, drawLumiText, year):
    cmsText = "CMS"
    extraText = "Private work"
    if year=="2017":
        lumi=41.5
    elif year=="2016_preVFP":
        lumi=19.5
    else:
        lumi=59.8
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

legends = {
    "Lep2_pY" : "p_{y}^{l2}",
    "nJets" : "N_{Jets}",
    "krRho" : "#rho_{kin. reco.}",
    "lkrRho" : "#rho_{loose kin. reco.}",
    "lept1Pt" : "p_{T}^{leading lepton}",
    "jet1Pt" : "p_{T}^{first jet}",
    "jet3Pt" : "p_{T}^{third jet}",
    "lept2Pt" : "p_{T}^{trailing lepton}",
    "lept1M" : "m(leading lepton)",
    "lept2M" : "m(trailing lepton)",
    "metPt" : "p_{T}^{miss}",
    "recoAddJetPt" : "p_{T}^{kin. reco. add. jet}",
    "ll1jM" : "m(l,l,jet_{1})",
    "l2j1M" : "m(l_{2},jet_{1})",
    "ll2jM" : "m(l,l,jet_{2})",
    "l1j2M" : "m(l_{1},jet_{2})",
    "l2j2M" : "m(l_{2},jet_{2})",
    "dileptonPt" : "m_{ll}",
    "dileptonMass" : "p_{T}^{ll}",
}

#### main
if __name__ == '__main__':
    #  ~variables = [
        #  ~"nJets", "krRho", "lkrRho",
        #  ~"lept1Pt","lept2Pt","lept1M","lept2M",
        #  ~"dileptonPt","dileptonMass",
        #  ~"jet1Pt","jet3Pt",
        #  ~"metPt","recoAddJetPt",
        #  ~"ll1jM","l2j1M",
        #  ~"ll2jM","l1j2M","l2j2M"
        #  ~]
    variables = [
        "Lep2_pY",
        ]
    for variable in variables:
        #  ~data = "Jobs_InputValidation_try3/workspace_FR2_ll_"+variable+"/GoodnessOfFit/Data/higgsCombine_workspace_FR2_ll_"+variable+".GoodnessOfFit.mH125.root"
        #  ~toys = "Jobs_InputValidation_try3/workspace_FR2_ll_"+variable+"/GoodnessOfFit/toys/higgsCombine_workspace_FR2_ll_"+variable+".GoodnessOfFit.mH125.101.root"
        data = "/home/home4/institut_1b/nattland/DNN_ttbar/RunGOFCondor/CondorGOFsubmits/"+variable+"_files/Results_"+variable+"/higgsCombineTest.GoodnessOfFit.mH125.root"
        #  ~toys = "/home/home4/institut_1b/nattland/DNN_ttbar/RunGOFCondor/CondorGOFsubmits/"+variable+"_files/Results_"+variable+"/higgsCombineTest.GoodnessOfFit.mH125.1544232278.root"
        toys = ["/home/home4/institut_1b/nattland/DNN_ttbar/RunGOFCondor/CondorGOFsubmits/"+variable+"_files/Results_"+variable+"/higgsCombineTest.GoodnessOfFit.mH125.1544232278.root",
                "/home/home4/institut_1b/nattland/DNN_ttbar/RunGOFCondor/CondorGOFsubmits/"+variable+"_files/Results_"+variable+"/higgsCombineTest.GoodnessOfFit.mH125.-563124713.root",
                "/home/home4/institut_1b/nattland/DNN_ttbar/RunGOFCondor/CondorGOFsubmits/"+variable+"_files/Results_"+variable+"/higgsCombineTest.GoodnessOfFit.mH125.-5586677.root"
                ]
                
        variableName = legends[variable]
        outputFolder = "/home/home4/institut_1b/nattland/DNN_ttbar/RunGOFCondor/CondorGOFsubmits/"+variable+"_files/Results_"+variable+"/"
        plotGOF(data, toys, outputFolder+"PValue", txtTL="", txtTR=variableName)
