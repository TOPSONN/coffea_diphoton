import numpy as np
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
import time
import matplotlib.pyplot as plt
import mplhep as hep
import uproot
import hist


#############################################################################
##### open file #######

file = 'nano_VBS_run2.root'
vbs_events = NanoEventsFactory.from_root(file, entry_stop=1000, treepath='Events;1').events()

file = 'nano_QCD_run2.root'
qcd_events = NanoEventsFactory.from_root(file, entry_stop=1000, treepath='Events;1').events()

file = 'nano_inter_run2.root'
inter_events = NanoEventsFactory.from_root(file, entry_stop=1000,treepath='Events;1').events()


print('---------------------')
print(inter_events.fields)
print(inter_events.Photon.fields)
print(inter_events.GenPart.fields)

##inter_photon inter_gen inter_jet 


##########define INTER photon jet and gen########################################################################################


inter_photon = inter_events.Photon

inter_genpart = inter_events.GenPart[ (abs(inter_events.GenPart.pdgId)==22)  & ((inter_events.GenPart.status) ==1) ]
inter_genpart = inter_genpart[ak.num(inter_events.Photon, axis=1) >= 2]

inter_genpart_status1 = inter_events.GenPart[ (abs(inter_events.GenPart.pdgId)==22)  &  ((inter_events.GenPart.status) ==1)]
inter_photon=inter_photon[ak.num(inter_photon, axis=1) >= 2]


#####--------------define VBS photon jet and gen-------------------#####

vbs_photon = vbs_events.Photon

vbs_genpart = vbs_events.GenPart[ (abs(vbs_events.GenPart.pdgId)==22)  & ((vbs_events.GenPart.status) ==1) ]
vbs_genpart = vbs_genpart[ak.num(vbs_events.Photon, axis=1) >= 2]

vbs_genpart_status1 = vbs_events.GenPart[ (abs(vbs_events.GenPart.pdgId)==22)  &  ((vbs_events.GenPart.status) ==1)]
vbs_photon=vbs_photon[ak.num(vbs_photon, axis=1) >= 2]

#####--------------define QCD photon jet and gen-------------------#####

qcd_photon = qcd_events.Photon

qcd_genpart = qcd_events.GenPart[ (abs(qcd_events.GenPart.pdgId)==22)  & ((qcd_events.GenPart.status) ==1) ]
qcd_genpart = qcd_genpart[ak.num(qcd_events.Photon, axis=1) >= 2]

qcd_genpart_status1 = qcd_events.GenPart[ (abs(qcd_events.GenPart.pdgId)==22)  &  ((qcd_events.GenPart.status) ==1)]
qcd_photon=qcd_photon[ak.num(qcd_photon, axis=1) >= 2]



####################INTER photon selection########################################################################################

inter_phoeta_nested, inter_geneta_nested = ak.unzip(ak.cartesian([inter_photon.eta, inter_genpart.eta], nested=True))
inter_phophi_nested, inter_genphi_nested = ak.unzip(ak.cartesian([inter_photon.phi, inter_genpart.phi], nested=True))
inter_phopt_nested, inter_genpt_nested = ak.unzip(ak.cartesian([inter_photon.pt, inter_genpart.pt], nested=True))

inter_delta_Pt = abs((inter_genpt_nested - inter_phopt_nested)/inter_genpt_nested)
inter_delta_R = ((inter_phophi_nested - inter_genphi_nested)**2 + (inter_phoeta_nested - inter_geneta_nested)**2 )**0.5
print('aaaaaaa')
print(ak.to_list(inter_photon.pt))
#print(ak.to_list(inter_genpart.pt))
#print(ak.to_list(inter_delta_Pt))


delta_R_hist = ak.to_numpy(ak.flatten(inter_delta_R) , allow_missing=True)
#plt.hist(ak.flatten(delta_R_hist) ,bins=np.arange(0, 1, step=0.1),density=True, histtype='step',color='green',stacked=True, fill=False)


#plt.legend(['inter_delta_R'])
#plt.ylabel('events') 
#plt.title('delta_R') 
#plt.savefig('0coffea_deltaR_cms.pdf')


inter_delta_Pt0 = inter_delta_Pt  < 0.7
inter_delta_R0 = inter_delta_R < 0.7


inter_united = ak.any(inter_delta_R0 & inter_delta_Pt0  , axis = -1)
inter_select_photon = inter_photon[inter_united]



inter_phoeta_nested, inter_selecteta_nested = ak.unzip(ak.cartesian([inter_photon.eta, inter_select_photon.eta], nested=True))
inter_fakecut = ak.all( abs(inter_phoeta_nested-inter_selecteta_nested) > 0 , axis=-1 )
inter_fakephoton = inter_photon[inter_fakecut ]


#####--------------VBS photon selection--------------#####


vbs_phoeta_nested, vbs_geneta_nested = ak.unzip(ak.cartesian([vbs_photon.eta, vbs_genpart.eta], nested=True))
vbs_phophi_nested, vbs_genphi_nested = ak.unzip(ak.cartesian([vbs_photon.phi, vbs_genpart.phi], nested=True))
vbs_phopt_nested, vbs_genpt_nested = ak.unzip(ak.cartesian([vbs_photon.pt, vbs_genpart.pt], nested=True))

vbs_delta_Pt = abs((vbs_genpt_nested - vbs_phopt_nested)/vbs_genpt_nested)
vbs_delta_R = ((vbs_phophi_nested - vbs_genphi_nested)**2 + (vbs_phoeta_nested - vbs_geneta_nested)**2 )**0.5
print('aaaaaaa')
print(ak.to_list(vbs_photon.pt))
#print(ak.to_list(vbs_genpart.pt))
#print(ak.to_list(vbs_delta_Pt))


delta_R_hist = ak.to_numpy(ak.flatten(vbs_delta_R) , allow_missing=True)
#plt.hist(ak.flatten(delta_R_hist) ,bins=np.arange(0, 1, step=0.1),density=True, histtype='step',color='green',stacked=True, fill=False)


#plt.legend(['vbs_delta_R'])
#plt.ylabel('events') 
#plt.title('delta_R') 
#plt.savefig('0coffea_deltaR_cms.pdf')


vbs_delta_Pt0 = vbs_delta_Pt  < 0.7
vbs_delta_R0 = vbs_delta_R < 0.7


vbs_united = ak.any(vbs_delta_R0 & vbs_delta_Pt0  , axis = -1)
vbs_select_photon = vbs_photon[vbs_united]



vbs_phoeta_nested, vbs_selecteta_nested = ak.unzip(ak.cartesian([vbs_photon.eta, vbs_select_photon.eta], nested=True))
vbs_fakecut = ak.all( abs(vbs_phoeta_nested-vbs_selecteta_nested) > 0 , axis=-1 )
vbs_fakephoton = vbs_photon[vbs_fakecut ]


#####---------------QCD photon selection------------------#####

qcd_phoeta_nested, qcd_geneta_nested = ak.unzip(ak.cartesian([qcd_photon.eta, qcd_genpart.eta], nested=True))
qcd_phophi_nested, qcd_genphi_nested = ak.unzip(ak.cartesian([qcd_photon.phi, qcd_genpart.phi], nested=True))
qcd_phopt_nested, qcd_genpt_nested = ak.unzip(ak.cartesian([qcd_photon.pt, qcd_genpart.pt], nested=True))

qcd_delta_Pt = abs((qcd_genpt_nested - qcd_phopt_nested)/qcd_genpt_nested)
qcd_delta_R = ((qcd_phophi_nested - qcd_genphi_nested)**2 + (qcd_phoeta_nested - qcd_geneta_nested)**2 )**0.5
print('aaaaaaa')
print(ak.to_list(qcd_photon.pt))
#print(ak.to_list(qcd_genpart.pt))
#print(ak.to_list(qcd_delta_Pt))


delta_R_hist = ak.to_numpy(ak.flatten(qcd_delta_R) , allow_missing=True)
#plt.hist(ak.flatten(delta_R_hist) ,bins=np.arange(0, 1, step=0.1),density=True, histtype='step',color='green',stacked=True, fill=False)


#plt.legend(['qcd_delta_R'])
#plt.ylabel('events') 
#plt.title('delta_R') 
#plt.savefig('0coffea_deltaR_cms.pdf')


qcd_delta_Pt0 = qcd_delta_Pt  < 0.7
qcd_delta_R0 = qcd_delta_R < 0.7


qcd_united = ak.any(qcd_delta_R0 & qcd_delta_Pt0  , axis = -1)
qcd_select_photon = qcd_photon[qcd_united]



qcd_phoeta_nested, qcd_selecteta_nested = ak.unzip(ak.cartesian([qcd_photon.eta, qcd_select_photon.eta], nested=True))
qcd_fakecut = ak.all( abs(qcd_phoeta_nested-qcd_selecteta_nested) > 0 , axis=-1 )
qcd_fakephoton = qcd_photon[qcd_fakecut ]




####################inter_photon Barrel Endcap##########################################################

inter_fake_photon_endcap = inter_fakephoton[abs(inter_fakephoton.eta) > 1.57 ]
inter_fake_photon_endcap = inter_fake_photon_endcap[abs(inter_fake_photon_endcap.eta) < 2.5 ]
inter_fake_photon_barral = inter_fakephoton[abs(inter_fakephoton.eta) > 0 ]
inter_fake_photon_barral = inter_fake_photon_barral[abs(inter_fake_photon_barral.eta) < 1.44 ]


inter_select_photon_endcap = inter_select_photon[abs(inter_select_photon.eta) > 1.57 ]
inter_select_photon_endcap = inter_select_photon_endcap[abs(inter_select_photon_endcap.eta) < 2.5 ]
inter_select_photon_barral = inter_select_photon[abs(inter_select_photon.eta) > 0 ]
inter_select_photon_barral = inter_select_photon_barral[abs(inter_select_photon_barral.eta) < 1.44 ]


#####-------------vbs_photon Barrel Endcap--------------------#####

vbs_fake_photon_endcap = vbs_fakephoton[abs(vbs_fakephoton.eta) > 1.57 ]
vbs_fake_photon_endcap = vbs_fake_photon_endcap[abs(vbs_fake_photon_endcap.eta) < 2.5 ]
vbs_fake_photon_barral = vbs_fakephoton[abs(vbs_fakephoton.eta) > 0 ]
vbs_fake_photon_barral = vbs_fake_photon_barral[abs(vbs_fake_photon_barral.eta) < 1.44 ]


vbs_select_photon_endcap = vbs_select_photon[abs(vbs_select_photon.eta) > 1.57 ]
vbs_select_photon_endcap = vbs_select_photon_endcap[abs(vbs_select_photon_endcap.eta) < 2.5 ]
vbs_select_photon_barral = vbs_select_photon[abs(vbs_select_photon.eta) > 0 ]
vbs_select_photon_barral = vbs_select_photon_barral[abs(vbs_select_photon_barral.eta) < 1.44 ]


#####-------------qcd_photon Barrel Endcap--------------------#####


qcd_fake_photon_endcap = qcd_fakephoton[abs(qcd_fakephoton.eta) > 1.57 ]
qcd_fake_photon_endcap = qcd_fake_photon_endcap[abs(qcd_fake_photon_endcap.eta) < 2.5 ]
qcd_fake_photon_barral = qcd_fakephoton[abs(qcd_fakephoton.eta) > 0 ]
qcd_fake_photon_barral = qcd_fake_photon_barral[abs(qcd_fake_photon_barral.eta) < 1.44 ]


qcd_select_photon_endcap = qcd_select_photon[abs(qcd_select_photon.eta) > 1.57 ]
qcd_select_photon_endcap = qcd_select_photon_endcap[abs(qcd_select_photon_endcap.eta) < 2.5 ]
qcd_select_photon_barral = qcd_select_photon[abs(qcd_select_photon.eta) > 0 ]
qcd_select_photon_barral = qcd_select_photon_barral[abs(qcd_select_photon_barral.eta) < 1.44 ]


############################################################# charge isolated


inter_No_matchbarral_CHISO_ = ak.to_numpy(ak.flatten(inter_fake_photon_barral.pfRelIso03_chg), allow_missing=True)
inter_No_matchend_CHISO_ =  ak.to_numpy(ak.flatten(inter_fake_photon_endcap.pfRelIso03_chg) , allow_missing=True)
inter_matchend_CHISO_ =  ak.to_numpy(ak.flatten(inter_select_photon_endcap.pfRelIso03_chg) , allow_missing=True)
inter_matchbarral_CHISO_ =  ak.to_numpy(ak.flatten(inter_select_photon_barral.pfRelIso03_chg) , allow_missing=True)

vbs_No_matchbarral_CHISO_ = ak.to_numpy(ak.flatten(vbs_fake_photon_barral.pfRelIso03_chg), allow_missing=True)
vbs_No_matchend_CHISO_ =  ak.to_numpy(ak.flatten(vbs_fake_photon_endcap.pfRelIso03_chg) , allow_missing=True)
vbs_matchend_CHISO_ =  ak.to_numpy(ak.flatten(vbs_select_photon_endcap.pfRelIso03_chg) , allow_missing=True)
vbs_matchbarral_CHISO_ =  ak.to_numpy(ak.flatten(vbs_select_photon_barral.pfRelIso03_chg) , allow_missing=True)

qcd_No_matchbarral_CHISO_ = ak.to_numpy(ak.flatten(qcd_fake_photon_barral.pfRelIso03_chg), allow_missing=True)
qcd_No_matchend_CHISO_ =  ak.to_numpy(ak.flatten(qcd_fake_photon_endcap.pfRelIso03_chg) , allow_missing=True)
qcd_matchend_CHISO_ =  ak.to_numpy(ak.flatten(qcd_select_photon_endcap.pfRelIso03_chg) , allow_missing=True)
qcd_matchbarral_CHISO_ =  ak.to_numpy(ak.flatten(qcd_select_photon_barral.pfRelIso03_chg) , allow_missing=True)





fig,ax1 = plt.subplots()

ax1.hist(inter_matchbarral_CHISO_,bins=np.arange(0.01, 2, step=0.1),density=True, histtype='step',color='red',stacked=True, fill=False)
ax1.hist(vbs_matchbarral_CHISO_,bins=np.arange(0.01, 2, step=0.1),density=True, histtype='step',color='green',stacked=True, fill=False)
ax1.hist(qcd_matchbarral_CHISO_,bins=np.arange(0.01, 2, step=0.1),density=True, histtype='step',color='navy',stacked=True, fill=False)
ax1.hist(inter_matchend_CHISO_,bins=np.arange(0.01, 2, step=0.1),density=True, histtype='step',color='purple',stacked=True, fill=False)
ax1.hist(vbs_matchend_CHISO_,bins=np.arange(0.01, 2, step=0.1),density=True, histtype='step',color='black',stacked=True, fill=False)
ax1.hist(qcd_matchend_CHISO_,bins=np.arange(0.01, 2, step=0.1),density=True, histtype='step',color='orange',stacked=True, fill=False)

plt.legend(['inter_true_barral', 'vbs_true_barrel', 'qcd_true_barrel','inter_true_endcap', 'vbs_true_endcap', 'qcd_true_endcap' ])
plt.ylabel('events') 
plt.title('charge_isolated') 
plt.savefig('0coffea_chiso_cms.pdf')
plt.show()
plt.close()





fig,ax = plt.subplots()

ax.hist(inter_No_matchbarral_CHISO_,bins=np.arange(0.01, 2, step=0.1),density=True, histtype='step',color='red',stacked=True, fill=False)
ax.hist(vbs_No_matchbarral_CHISO_,bins=np.arange(0.01, 2, step=0.1),density=True, histtype='step',color='green',stacked=True, fill=False)
ax.hist(qcd_No_matchbarral_CHISO_,bins=np.arange(0.01, 2, step=0.1),density=True, histtype='step',color='navy',stacked=True, fill=False)
ax.hist(inter_No_matchend_CHISO_,bins=np.arange(0.01, 2, step=0.1),density=True, histtype='step',color='purple',stacked=True, fill=False)
ax.hist(vbs_No_matchend_CHISO_,bins=np.arange(0.01, 2, step=0.1),density=True, histtype='step',color='black',stacked=True, fill=False)
ax.hist(qcd_No_matchend_CHISO_,bins=np.arange(0.01, 2, step=0.1),density=True, histtype='step',color='orange',stacked=True, fill=False)

plt.legend(['inter_true_barral', 'vbs_true_barrel', 'qcd_true_barrel','inter_true_endcap', 'vbs_true_endcap', 'qcd_true_endcap' ])
plt.ylabel('events') 
plt.title('charge_isolated') 
plt.savefig('0coffea_chiso_nomatch_cms.pdf')
plt.show()
plt.close()











#############################################################  charge isolated











#############################################################  sigmaetaeta






#############################################################  R9




############################################################# photon isolated





############################################################# photon pt









