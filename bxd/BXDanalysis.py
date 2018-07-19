#!/usr/bin/python
import sys
import os
import math


def BXDanalysis(Filename,MFPTthreshold,HistLowBound,HistUppBound,HistBinSize, ThresholdFile=None):
#
# This program processes a *.juj file to generate first passage times
# according to the original BXD definition (i.e., only considering the time delay 
# between subsequent inversions on boundary n).
#
# This script generates a number of files 
#  (1) a raw histogram from HistLowBound to HistUppBound with bin size HistBinSize
#  (2) a raw histogram identical to the the above file with the difference that 
#      each box normalized to unity
#  (3) the above file with the difference that it has been corrected by multiplication 
#      with the box dependent probability
#  (4) a PMF calculated from the above file
#  
# For each box boundary, it also generates a decay trace generated from the first passage
# times in both the forward and the reverse direction so that the user can inspect
# the significance of dynamical recrossing in box-to-box transitions
#
# It may be run as follows:
# python BXDanalysis.py Filename MFPTthreshold LowerBound UpperBound HistBinSize
#
# where:
# 
# Filename is the *.juj file to be analyzed
# LowerBound is the lower bound of the final desired histogram
# UpperBound is the upper box bound of the final desired histogram
# MFPTthreshold is the minimum value of a first passage time for it to be included in 
#                calculation of the mean first passage time (MFPT)
# HistBinSize is the bin size for the final desired histogram.
#
# LowerBound, UpperBound, and HistBinSize should be chosen so that bin boundaries coincide 
# with inversion boundaries. otherwise, the renormalization procedure is not exact.
# 
# Setting MFPTthreshold to zero means that none of the first passage times, no matter 
#         how small, are neglected in calculating the mean first passage time
#
#
# 
# Box-to-box free energy differences are calculated using the box-to-box equilibrium 
# coefficients obtained with the MFPTs generated via this script.
#
# For further details, see Glowacki et al., J. Phys. Chem. B 2009, 113, 16603-16611
#
#


###error checking
   if not(os.path.exists(Filename)):
    print(Filename, 'does not exist... exiting execution')
    sys.exit()
   if(HistUppBound<=HistLowBound):
    print("Upper Bound for Histogram is smaller than the Lower Bound... exiting")
    sys.exit
   if(HistBinSize<=0.0):
    print("Histogram bin size is less than or equal to zero... exiting")
    sys.exit
   if(HistBinSize>=(HistUppBound-HistLowBound)):
    print("Histogram bin size is larger than the difference between HistUppBound and HistLowBound... exiting")
    sys.exit
   if(MFPTthreshold < 0.0):
    print("mean first passage time threshold less than zero... aborting....")
    sys.exit()

   print("\nFirst passage times less than ", MFPTthreshold," will be neglected in calculating means...")

###get the minimum & maximum values of the reaction coordinate, and the boundary list
   print('\nAnalyzing %s to determine box boundaries and Min/Max values of rxn coordinate...'%(Filename))
   Min,Max,BoundaryList,binCents,binCounts=GetMinMaxBoundaryListAndMakeHistogram(Filename,HistLowBound,HistUppBound,HistBinSize)
   print('\nWithin ', Filename, ': ')
   print('\tMin = ', Min,'\tMax = ', Max)

   if (HistLowBound<BoundaryList[0]):
     BoundaryList.insert(0,HistLowBound)
   if (HistUppBound>BoundaryList[len(BoundaryList)-1]):
     BoundaryList.append(HistUppBound)

   nBounds=len(BoundaryList)  

   for i in range(0,nBounds-1):
     print('\tBox ',i+1,' spans ',BoundaryList[i],' to ',BoundaryList[i+1])

#  now calculate the MFPTs in and out of the boxes

   UpperThresholds = [MFPTthreshold] *(nBounds-1)
   LowerThresholds = [MFPTthreshold] *(nBounds-1)
   #Read upper and lower thresholds from file 
   if ThresholdFile is not None:
     LowerThresholds, UpperThresholds =ReadThresholds(ThresholdFile,LowerThresholds, UpperThresholds)
   boxIndex=1
   kUpperList=[]
   kLowerList=[]

   print('\nMean First Passage Times (MPFTs):')
   for i in range(0,nBounds-1):
     LowerBound=BoundaryList[i]
     UpperBound=BoundaryList[i+1]
     if(LowerBound >= UpperBound):
       print("Lower Bound greater than or equal to Upper Bound... aborting....")
       sys.exit()
     kdown,kup,LowerReflections,UpperReflections=GetFirstPassageTimes(Filename,boxIndex,LowerBound,UpperBound,LowerThresholds[i], UpperThresholds[i])
     if(i==0 and not UpperReflections):
       print('No reflections against boundary %s in box %s... check your input. aborting...'%(UpperBound,i+1))
       sys.exit()
     elif(i==(nBounds-2) and not LowerReflections):
       print('No reflections against boundary %s in box %s... check your input. aborting...'%(LowerBound,i+1))
       sys.exit()
     elif(i>0 and i<(nBounds-2) and not UpperReflections):
       print('No reflections against boundary %s in box %s... check your input. aborting...'%(UpperBound,i+1))
       sys.exit()
     elif(i>0 and i<(nBounds-2) and not LowerReflections):
       print('No reflections against boundary %s in box %s... check your input. aborting...'%(LowerBound,i+1))
       sys.exit()
     else:
       kUpperList.append(kup)
       kLowerList.append(kdown)
       boxIndex=boxIndex+1

#  calculate the box averaged Free Energy distribution
   boxFreeEnergy=[0.0]
   for i in range(0,nBounds-2):
     Keq=kUpperList[i]/kLowerList[i+1]
     if (Keq > 0.0):
       dG=-1.0*math.log(Keq)
     else:
       print("\nKeq is zero between box ",i," and ",i+1,". Check your histogram limits. aborting....")
       sys.exit()
#     print Keq, dG, kUpperList[i], kLowerList[i+1]
     boxFreeEnergy.append(boxFreeEnergy[i]+dG)  
   
#  calculate the unnormalized box averaged Probability distribution
   Z=0.0
   boxProbability=[]
   for i in range(0,len(boxFreeEnergy)):
     boxProbability.append(math.exp(-1.0*boxFreeEnergy[i]))
     Z=Z+boxProbability[i]

#  normalize the box averaged Probability distribution
   for i in range(0,len(boxFreeEnergy)):
     boxProbability[i]=boxProbability[i]/Z

#  print outs
   print("\nBox averaged energies/RT & probabilities:")  
   for i in range(0,len(boxFreeEnergy)):
     print('\tBox %s: %f \t %f'%(i+1,boxFreeEnergy[i],boxProbability[i]))

#  count the events in each box
   idx=0
   TotalCountsInBox=[]
   for i in range(0,len(boxFreeEnergy)):
     j=binCents[idx]
     TotalCountsInBox.append(0.0)
     while (j < BoundaryList[i+1]):
#       print j,   BoundaryList[i+1]
       TotalCountsInBox[i] += binCounts[idx]
       if((idx+1)<len(binCounts)):
         idx=idx+1
         j=binCents[idx]
       else:
         break

#  normalized the raw histogram & obtain the box probability, and do print outs
   normalizedHistogram=open('normalizedHistogram.txt','w')
   rawBoxNormalizedHistogram=open('rawBoxNormalizedHistogram.txt','w')
   idx=0
   for i in range(0,len(boxFreeEnergy)):
     j=binCents[idx]
     while (j < BoundaryList[i+1]):
#       print j, BoundaryList[i+1]
       string='%s\t%s\n'%(binCents[idx],binCounts[idx]/TotalCountsInBox[i])
       rawBoxNormalizedHistogram.write(string)
       binCounts[idx]=boxProbability[i]*binCounts[idx]/TotalCountsInBox[i]
       string='%s\t%s\n'%(binCents[idx],binCounts[idx])
       normalizedHistogram.write(string)
       if((idx+1)<len(binCounts)):
         idx=idx+1
         j=binCents[idx]
       else:
         break
   normalizedHistogram.close()
   rawBoxNormalizedHistogram.close()
   print("\nThe raw histogram with each box normalized to 1 is in ", rawBoxNormalizedHistogram.name)
   print("\nThe fully corrected & normalized histogram is printed in ", normalizedHistogram.name)

#  print out the final free energy surface
   finalFreeEnergy=open('finalFreeEnergy.txt','w')
   for i in range(0,len(binCounts)):
     if (binCounts[i] != 0):
      string='%s\t%s\n'%(binCents[i],-1.0*math.log(binCounts[i]/HistBinSize))
      finalFreeEnergy.write(string)
     else:
      print("\nThe final free energy surface cannot be constructed because of zeros in the Histogram...\n")
      sys.exit()
   finalFreeEnergy.close()
   print('\nThe final PMF is printed out to %s\n'%(finalFreeEnergy.name))
   





########### function to construct a 2d array ###########

def make_array(r,c):
  a=[]
  for i in range(r):
    a.append([])
    for ii in range(c):
      a[i].append(0)
  return a


########### function to get Min, Max values of Rxn coordinate #########
###########     & boundary list from the BXD output file    ###########

def GetMinMaxBoundaryListAndMakeHistogram(opfilename,LowBound,UppBound,BinSize):

   if(os.path.exists(opfilename)):
    opfile=open(opfilename,'r')

#   initialize & set up stuff for making a histogram
    span=UppBound-LowBound
    binCenters=[]
    count=[]
    temp_counts = []
    print(span/BinSize)
    print(round(span/BinSize))
    print(abs)
    if(abs(span/BinSize - round(span/BinSize)) <= 0.0001):
      nbins=int(span/BinSize)
    else:
      print("\nYou've chosen a bin size that doesn't give an integer number of bins across the range specified...")
      print("For best results, this needs to be fixed... aborting...\n")
      sys.exit()
    print('\nMaking a histogram from %s to %s with bin size %s (%s bins)...'%(LowBound,UppBound,BinSize,nbins))

    for i in range(0,nbins):
     binCenters.append((i+1)*BinSize+LowBound-BinSize/2)
#     print binCenters[i]
     count.append(0)
     temp_counts.append(0)
    ctr=0
    MinVal=0.0
    MaxVal=0.0
    BounList=[]
    LastBoundHit = None
    LastHitTime = {}
#   read the file
    while 1:
      linestring=opfile.readline()
      if not linestring:break
      linelist=linestring.split()
      if (len(linelist)==0):
        print("\nfound blank line at line number ",ctr,"in file ", opfile.name, " ... Remove and retry...aborting")
        sys.exit()
      else:
        ctr=ctr+1
        RxnCrdVal=float(linelist[1])
        if (len(linelist)>=3):
          boundary=float(linelist[2])
          if boundary not in BounList:
            BounList.append(boundary)
            LastHitTime[boundary] = 0
          if LastBoundHit is None:
            LastBoundHit = boundary
            LastHitTime[boundary] = float(linelist[0])
          else:
            if LastBoundHit != boundary:
              time = float(linelist[0])
              min_passage_time = 12.0
              if time - LastHitTime[boundary] > min_passage_time:
                for i in range(len(temp_counts)):
                  count[i] += temp_counts[i]
              else: 
                print("Discarding hit: " + linestring)
              for i in range(len(temp_counts)):
                temp_counts[i] = 0
              LastHitTime[boundary] = time
              LastBoundHit = boundary
        if(ctr==1 or RxnCrdVal<MinVal):
          MinVal=RxnCrdVal
        if(ctr==1 or RxnCrdVal>MaxVal):
          MaxVal=RxnCrdVal
        if not (len(linelist)==3):
          binIdx=int((RxnCrdVal-LowBound)/BinSize)
#          print RxnCrdVal, binIdx, LowBound, BinSize, len(count)
          if (binIdx < len(count)):
            temp_counts[binIdx] += 1        

    for i in range(len(temp_counts)):
      count[i] += temp_counts[i]
    opfile.close()
    BounList.sort()

#   write out the raw histogram
    rawHistogram=open('rawHistogram.txt','w')
    for i in range(0,nbins):
     string='%s\t%s\n'%(binCenters[i],count[i])
     rawHistogram.write(string)
    rawHistogram.close()
    print("...Raw histogram printed out to ", rawHistogram.name)

    return MinVal,MaxVal,BounList,binCenters,count

   else:
    print(opfilename, 'does not exist')
    sys.exit()


########### function to get Reactive trajectory results #########

def GetFirstPassageTimes(opfilename,boxIdx,Lbound,Ubound,LowerThreshold, UpperThreshold):

   if(os.path.exists(opfilename)):
    opfile=open(opfilename,'r')

    line=0
    StepsInsideBox=1
    j=0
    stopStepsInsideBox=1
    timep=0.0
    distancep=0.0
    LastUpperHitTime=0.0
    LastLowerHitTime=0.0
    FoundFirstUpperHit=False
    FoundFirstLowerHit=False
    InversionsAtLower=False
    InversionsAtUpper=False
    InsideTheBox=False
    NumUpperHits=0
    NumLowerHits=0
    TotalTimeintheBox=0.0
    
#   initialize two lists
    UpperFPTs=[]
    LowerFPTs=[]

    while 1:
#    while (StepsInsideBox<=1000):
     linestring=opfile.readline()
     line=line+1
     if not linestring:break
     else:
      linelist=linestring.split()
      time=float(linelist[0])
      distance=float(linelist[1])

#     what to do if we're inside the box
      if((Lbound<=distance) and (distance <= Ubound)):
       StepsInsideBox=StepsInsideBox+1 
#      what to do if we were outside the box & now we're inside
       if(not InsideTheBox):
        InsideTheBox=True
        TimeWeEnteredTheBox=time

#      what to do if there's an inversion boundary
       if((len(linelist))>=3):
        InversionBoundary=float(linelist[2])
#        print "--------StepsInsideBox", StepsInsideBox, distance,Lbound,Ubound 
#        print "found a 3 character line!!", linelist, "InversionBoundary", InversionBoundary, Lbound, Ubound
#       What to do if the Inversion Boundary is identical to the Upper Bound
        if(InversionBoundary==Ubound):
#         print "InversionBoundary==Ubound!!.... (if) FoundFirstUpperHit = ", FoundFirstUpperHit, InversionBoundary, Lbound, Ubound
         if(FoundFirstUpperHit):
          passageTime=time-LastUpperHitTime
#          print "Upper", time, LastUpperHitTime, passageTime
#         what to do if we obtain a negative passage time b/c files appended together
          if(passageTime<=0):
           FoundFirstUpperHit=False
#           print "Setting First FoundFirstUpperHit to false"
          else:
#           print "Appending data to UpperFPTs array"
           UpperFPTs.append(passageTime)
#           print UpperFPTs
           LastUpperHitTime=time
           NumUpperHits=NumUpperHits+1
         else:
          LastUpperHitTime=time
#          print "LastUpperHitTime", LastUpperHitTime
          FoundFirstUpperHit=True
#          print "(else) FoundFirstUpperHit = ", FoundFirstUpperHit 

#       What to do if the Inversion Boundary is identical to the Lower Bound
        if(InversionBoundary==Lbound):
         if(FoundFirstLowerHit):
          passageTime=time-LastLowerHitTime
#         print "Lower", time, LastLowerHitTime, time-LastLowerHitTime
#         what to do if we obtain a negative passage time b/c files appended together
          if(passageTime<=0):
           FoundFirstLowerHit=False
          else:
           if passageTime > 100000:
            print("Really long FPT (",passageTime,") on line: ", linestring)
           LowerFPTs.append(passageTime)
           LastLowerHitTime=time    
           NumLowerHits=NumLowerHits+1
#           print "NumLowerHits", NumLowerHits
         else:
          LastLowerHitTime=time
#         print LastLowerHitTime
          FoundFirstLowerHit=True

#     what to do if we're outside the box      
      else:
       FoundFirstUpperHit=False
#       print "(outside the box else) FoundFirstUpperHit = ", distance,Lbound, Ubound     
       FoundFirstLowerHit=False
#      what to do if we were inside the box, and now we're inside
       if(InsideTheBox):
        TimeWeLeftTheBox=time
        TimeInTheBox=TimeWeLeftTheBox-TimeWeEnteredTheBox
        TotalTimeintheBox=TotalTimeintheBox+TimeInTheBox
#        print TimeWeEnteredTheBox,TimeWeLeftTheBox,TimeInTheBox,NumLowerHits,NumUpperHits
       InsideTheBox=False
         
#   print stuff
    opfile.close() 
    NA='N/A'

#   MPFTS will only be averaged if they are greater than recrossTime

    ctr=0

#   calculate Mean First Passage Time
#    print "LowerFPTs", LowerFPTs
    Initial=len(LowerFPTs)
    if (Initial != 0):
     lowerFileName='%sto%s.txt'%(boxIdx,boxIdx-1)
     sumtotal=0.0
     for i in range(0,Initial):
      if((LowerFPTs[i])> LowerThreshold):
       sumtotal=sumtotal+LowerFPTs[i]
       ctr=ctr+1
     MFPT=sumtotal/ctr
     klower=1/MFPT
     print("\t MFPT box %s to box %s = %s (See file %s)"%(boxIdx,boxIdx-1,MFPT,lowerFileName))
     print("\tbox %s to box %s = %s (See file %s)"%(boxIdx,boxIdx-1,klower,lowerFileName))
     InversionsAtLower=True

#    calculate Decay Profile
     LowerDecayArray=CalculateDecay(LowerFPTs)
     keylist=list(LowerDecayArray.keys())
     keylist.sort()
     lowerFile=open(lowerFileName,'w')
     lowerFile.write('0.0\t%s\n'%(Initial))
     for key in keylist:
      lowerFile.write('%s\t%s\n'%(str(key),(LowerDecayArray[key])))
     lowerFile.close()
    else:
     print("\tbox %s to box %s = N/A "%(boxIdx,boxIdx-1))
     klower=0.0

    ctr=0

#   calculate Mean First Passage Time
#    print "UpperFPTs", UpperFPTs
    Initial=len(UpperFPTs)
    if (Initial != 0):
     upperFileName='%sto%s.txt'%(boxIdx,boxIdx+1)
     sumtotal=0.0
     for i in range(0,Initial):
      if((UpperFPTs[i])> UpperThreshold):
       sumtotal=sumtotal+UpperFPTs[i]
       ctr=ctr+1
     MFPT=sumtotal/ctr
     kupper=1/MFPT
     print("\t MFPT box %s to box %s = %s (See file %s)"%(boxIdx,boxIdx+1,MFPT,upperFileName))
     print("\tbox %s to box %s = %s (See file %s)"%(boxIdx,boxIdx+1,kupper,upperFileName))
     InversionsAtUpper=True

#   calculate Decay Profile
     UpperDecayArray=CalculateDecay(UpperFPTs)
     keylist=list(UpperDecayArray.keys())
     keylist.sort()
     upperFile=open(upperFileName,'w')
     upperFile.write('0.0\t%s\n'%(Initial))
     for key in keylist:
      upperFile.write('%s\t%s\n'%(str(key),(UpperDecayArray[key])))
     upperFile.close() 
    else:
     print("\tbox %s to box %s = N/A "%(boxIdx,boxIdx+1))
     kupper=0.0

    opfile.close()
    return klower,kupper,InversionsAtLower,InversionsAtUpper

   else:
    print(opfilename, 'does not exist')


 
def CalculateDecay(PassageTimeArray):

   Elements=len(PassageTimeArray)
   PassageTimeArray.sort()
#  initialize a Map
   DecayArray={}
   NumberElementsLeft=Elements

   for i in range(0,Elements):
    NumberElementsLeft=NumberElementsLeft-1
    DecayArray[PassageTimeArray[i]]=NumberElementsLeft	
   
   return DecayArray


if __name__ == "__main__":
#  print sys.argv
  if(len(sys.argv) == 7):
    BXDanalysis( sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6] )
  else:
    BXDanalysis( sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]) )

