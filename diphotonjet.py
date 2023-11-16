import numpy as np
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
import time

import matplotlib.pyplot as plt
import mplhep as hep

import uproot
import hist

file = 'diphotonjets.root'
events = NanoEventsFactory.from_root(file, entry_stop=10000).events()

photon = events.Photon
genpart = events.GenPart[ (abs(events.GenPart.pdgId)==22)   ]
momindex_genpart = genpart.genPartIdxMother
genpart_pidall = events.GenPart.pdgId

print( genpart_pidall ) 
j0 = ak.local_index(genpart_pidall, axis=-1)  
print( j0 ) 


j1_ = ak.local_index(genpart_pidall, axis=-1) 
j1 = j1_[events.GenPart.pdgId==25]
print('---------------------')
print( j1 ) 


print(events.Photon.fields)

print(events.fields)

genpart_ =  genpart[(genpart.status ==1)]
phoeta_nested, geneta_nested = ak.unzip(ak.cartesian([photon.eta, genpart_.eta], nested=True))
phophi_nested, genphi_nested = ak.unzip(ak.cartesian([photon.phi, genpart_.phi], nested=True))
phopt_nested, genpt_nested = ak.unzip(ak.cartesian([photon.pt, genpart_.pt], nested=True))

delta_Pt = abs((genpt_nested - phopt_nested)/genpt_nested)
delta_R = ((phophi_nested - genphi_nested)**2 + (phoeta_nested - geneta_nested)**2 )**0.5 
delta_Pt0 = delta_Pt  < 0.1
delta_R0 = delta_R < 0.1


print(delta_Pt0)
print(delta_R0)
print(delta_Pt0 & delta_R0)
united = ak.any(delta_R0 & delta_Pt0  , axis = -1)
print(united)
select_photon = photon[united]



phoeta_nested, selecteta_nested = ak.unzip(ak.cartesian([photon.eta, select_photon.eta], nested=True))
fakecut = ak.all( abs(phoeta_nested-selecteta_nested) > 0 , axis=-1 )
fakephoton = photon[fakecut ]









fake_photon_endcap = fakephoton[abs(fakephoton.eta) > 1.57 ]
fake_photon_endcap = fake_photon_endcap[abs(fake_photon_endcap.eta) < 2.5 ]
fake_photon_barral = fakephoton[abs(fakephoton.eta) > 0 ]
fake_photon_barral = fake_photon_barral[abs(fake_photon_barral.eta) < 1.44 ]


select_photon_endcap = select_photon[abs(select_photon.eta) > 1.57 ]
select_photon_endcap = select_photon_endcap[abs(select_photon_endcap.eta) < 2.5 ]
select_photon_barral = select_photon[abs(select_photon.eta) > 0 ]
select_photon_barral = select_photon_barral[abs(select_photon_barral.eta) < 1.44 ]


##############################################################


############################################################# H/E
print('\n')




No_matchbarral_HE_ = ak.to_numpy(ak.flatten(fake_photon_barral.hoe), allow_missing=True)
No_matchend_HE_ =  ak.to_numpy(ak.flatten(fake_photon_endcap.hoe) , allow_missing=True)
matchend_HE_ =  ak.to_numpy(ak.flatten(select_photon_endcap.hoe) , allow_missing=True)
matchbarral_HE_ =  ak.to_numpy(ak.flatten(select_photon_barral.hoe) , allow_missing=True)





plt.hist(No_matchbarral_HE_,bins=np.arange(0, 0.1, step=0.001),density=True, histtype='step',color='green',stacked=True, fill=False)
plt.hist(No_matchend_HE_,bins=np.arange(0, 0.1, step=0.001),density=True, histtype='step',color='yellow',stacked=True, fill=False)
plt.hist(matchbarral_HE_,bins=np.arange(0, 0.1, step=0.001),density=True, histtype='step',color='red',stacked=True, fill=False)
plt.hist(matchend_HE_,bins=np.arange(0, 0.1, step=0.001),density=True, histtype='step',color='blue',stacked=True, fill=False)

plt.legend(['fake_barral', 'fake_endcap', 'true_barral', 'true_endcap'])
plt.ylabel('events') 
plt.title('H/E') 
plt.savefig('0coffea_hoe.pdf')
plt.show()
plt.close()


#############################################################  charge isolated



No_matchbarral_CHISO_ = ak.to_numpy(ak.flatten(fake_photon_barral.pfRelIso03_chg), allow_missing=True)
No_matchend_CHISO_ =  ak.to_numpy(ak.flatten(fake_photon_endcap.pfRelIso03_chg) , allow_missing=True)
matchend_CHISO_ =  ak.to_numpy(ak.flatten(select_photon_endcap.pfRelIso03_chg) , allow_missing=True)
matchbarral_CHISO_ =  ak.to_numpy(ak.flatten(select_photon_barral.pfRelIso03_chg) , allow_missing=True)

plt.hist(No_matchbarral_CHISO_,bins=np.arange(0.01, 20, step=0.1),density=True, histtype='step',color='green',stacked=True, fill=False)
plt.hist(No_matchend_CHISO_,bins=np.arange(0.01, 20, step=0.1),density=True, histtype='step',color='yellow',stacked=True, fill=False)
plt.hist(matchbarral_CHISO_,bins=np.arange(0.01, 20, step=0.1),density=True, histtype='step',color='red',stacked=True, fill=False)
plt.hist(matchend_CHISO_,bins=np.arange(0.01, 20, step=0.1),density=True, histtype='step',color='blue',stacked=True, fill=False)

plt.legend(['fake_barral', 'fake_endcap', 'true_barral', 'true_endcap'])
plt.ylabel('events') 
plt.title('Charge isolated') 
plt.ylim(0,1)
plt.savefig('0coffea_chiso.pdf')
plt.show()
plt.close()

#############################################################  sigmaetaeta


No_matchbarral_sieie_ = ak.to_numpy(ak.flatten(fake_photon_barral.sieie), allow_missing=True)
No_matchend_sieie_ =  ak.to_numpy(ak.flatten(fake_photon_endcap.sieie) , allow_missing=True)
matchend_sieie_ =  ak.to_numpy(ak.flatten(select_photon_endcap.sieie) , allow_missing=True)
matchbarral_sieie_ =  ak.to_numpy(ak.flatten(select_photon_barral.sieie) , allow_missing=True)

sigma = 0.001
plt.hist(No_matchbarral_sieie_,bins=np.arange(0.001, 0.1 , sigma),density=True, histtype='step',color='green',stacked=True, fill=False)
plt.hist(No_matchend_sieie_,bins=np.arange(0.001, 0.1 , sigma),density=True, histtype='step',color='yellow',stacked=True, fill=False)
plt.hist(matchbarral_sieie_,bins=np.arange(0.001, 0.1, sigma),density=True, histtype='step',color='red',stacked=True, fill=False)
plt.hist(matchend_sieie_,bins=np.arange(0.001, 0.1 , sigma),density=True, histtype='step',color='blue',stacked=True, fill=False)

plt.legend(['fake_barral', 'fake_endcap', 'true_barral', 'true_endcap'])
plt.ylabel('events') 
plt.title('Photon SigmaSigmaEta') 
plt.savefig('0coffea_sieie.pdf')
plt.show()
plt.close()

#############################################################  R9


No_matchbarral_r9_ = ak.to_numpy(ak.flatten(fake_photon_barral.r9), allow_missing=True)
No_matchend_r9_ =  ak.to_numpy(ak.flatten(fake_photon_endcap.r9) , allow_missing=True)
matchend_r9_ =  ak.to_numpy(ak.flatten(select_photon_endcap.r9) , allow_missing=True)
matchbarral_r9_ =  ak.to_numpy(ak.flatten(select_photon_barral.r9) , allow_missing=True)

plt.hist(No_matchbarral_r9_,bins=np.arange(0.01, 1, step=0.01),density=True ,histtype='step',color='green',stacked=True, fill=False)
plt.hist(No_matchend_r9_,bins=np.arange(0.01, 1, step=0.01),density=True, histtype='step',color='yellow',stacked=True, fill=False)
plt.hist(matchbarral_r9_,bins=np.arange(0.01, 1, step=0.01),density=True, histtype='step',color='red',stacked=True, fill=False)
plt.hist(matchend_r9_,bins=np.arange(0.01, 1, step=0.01),density=True, histtype='step',color='blue',stacked=True, fill=False)

plt.legend(['fake_barral', 'fake_endcap', 'true_barral', 'true_endcap'])
plt.ylabel('events') 
plt.title('Photon R9') 
plt.savefig('0coffea_R9.pdf')
plt.show()
plt.close()




############################################################# photon isolated


No_matchbarral_pfPhoIso03_ = ak.to_numpy(ak.flatten(fake_photon_barral.pfRelIso03_all), allow_missing=True)
No_matchend_pfPhoIso03_ =  ak.to_numpy(ak.flatten(fake_photon_endcap.pfRelIso03_all) , allow_missing=True)
matchend_pfPhoIso03_ =  ak.to_numpy(ak.flatten(select_photon_endcap.pfRelIso03_all) , allow_missing=True)
matchbarral_pfPhoIso03_ =  ak.to_numpy(ak.flatten(select_photon_barral.pfRelIso03_all) , allow_missing=True)

fig,ax = plt.subplots()
ax.hist(No_matchbarral_pfPhoIso03_,bins=np.arange(0.01, 20, step=0.1),density=True,   histtype='step',color='green', fill=False)
ax.hist(No_matchend_pfPhoIso03_,bins=np.arange(0.01, 20, step=0.1),density=True,  histtype='step',color='yellow', fill=False)
ax.hist(matchbarral_pfPhoIso03_,bins=np.arange(0.01, 20, step=0.1),density=True,  histtype='step',color='red', fill=False)
ax.hist(matchend_pfPhoIso03_,bins=np.arange(0.01, 20, step=0.1),density=True,  histtype='step',color='blue', fill=False)

plt.legend(['fake_barral', 'fake_endcap', 'true_barral', 'true_endcap'])
plt.ylabel('events') 
plt.title('Photon Isolated') 
plt.savefig('0coffea_phiso.pdf') 
plt.show()
plt.close()

############################################################# photon pt

m_true_photon =select_photon[ak.num(select_photon, axis=1) == 2]
#m_true_photon_higg =select_photon_higg[ak.num(select_photon_higg, axis=1) == 1]
#a[ak.any(a[:, 0:1] > 40, axis=1) & ak.any(a[:, 1:2] > 30, axis=1)]

m_false_photon =fakephoton[ak.num(fakephoton, axis=1) >= 2]

m_af = m_true_photon.pt[ (m_true_photon.pt>30) & (m_true_photon.pt[: , 0] > 40) ]
m_bf = m_true_photon.pt[ ak.any(m_true_photon.pt[:, 0:1] > 40, axis=1) & ak.any(m_true_photon.pt[: , 1:2] > 30) ]

m_af_pt = ak.sum(m_af ,  axis=-1)
m_bf_pt = ak.sum(m_bf ,  axis=-1)

 
m = m_bf[:,0] + m_bf[:,1]

print(m)

fig,ax = plt.subplots()
ax.hist(m_af_pt ,bins=np.arange(0, 200, step=1) , density=False,   histtype='step',color='red', fill=False)
ax.hist(m_bf_pt  ,bins=np.arange(0, 200, step=1), density=False,  histtype='step',color='green', fill=False)


plt.legend(['true', ])
plt.ylabel('events') 
plt.title('mass') 
plt.savefig('0coffeamass.pdf') 
plt.show()
plt.close()


#########################################################

recojet = events.Jet
genpart_jet = events.GenPart[ (abs(events.GenPart.pdgId)==1)| (abs(events.GenPart.pdgId)==2)¡@| (abs(events.GenPart.pdgId)==3)¡@|(abs(events.GenPart.pdgId)==5) | (abs(events.GenPart.pdgId)==6) ]
#genpart_jet = genpart_jet[(genpart_jet.status ==1)]
recojeteta_nested, genjeteta_nested = ak.unzip(ak.cartesian([recojet.eta, genpart_jet_final.eta], nested=True))
recojetphi_nested, genjetphi_nested = ak.unzip(ak.cartesian([recojet.phi, genpart_jet_final.phi], nested=True))
recojetpt_nested, genjetpt_nested = ak.unzip(ak.cartesian([recojet.pt, genpart_jet_final.pt], nested=True))

delta_Pt_jet = abs((genjetpt_nested - recojetpt_nested)/genjetpt_nested)
delta_R_jet = ((recojetphi_nested - genjetphi_nested)**2 + (recojeteta_nested - genjeteta_nested)**2 )**0.5 
delta_Pt0_jet = delta_Pt_jet  < 0.1
delta_R0_jet = delta_R_jet < 0.1


print(delta_Pt0_jet)
print(delta_R0_jet)
print(delta_Pt0_jet & delta_R0_jet)
united_jet = ak.any(delta_R0_jet & delta_Pt0_jet  , axis = -1)
print(united_jet)
select_recojet = recojet[united_jet]



recojeteta_nested, selectjeteta_nested = ak.unzip(ak.cartesian([recojet.eta, select_recojet.eta], nested=True))
fakecut_jet = ak.all( abs(recojeteta_nested-selectjeteta_nested) > 0 , axis=-1 )
fake_jet = recojet[fakecut_jet ]









##############################################################

#print(momindex_genpart)
#print(genpart)




path = '0ex_pdgid.txt'
f = open(path, 'w')
f.write('\n')
f.write(str(ak.to_list(momindex_genpart)))
f.close()

path = '0ex_genPartIdxMother.txt'
f1 = open(path, 'w')
f1.write('\n')
f1.write(str(ak.to_list(genpart.pdgId)))
f1.close()

path = '0ex_allpdgid.txt'
f2 = open(path, 'w')
f2.write('\n')
f2.write(str(ak.to_list(genpart_pidall)))
f2.close()

path = '0ex_allpdgid_index.txt'
f3 = open(path, 'w')
f3.write('\n')
f3.write(str(ak.to_list(j0)))
f3.close()







print(sle.pdgId)






