# This script processes MIMIC-III dataset and builds a binary matrix or a count matrix depending on your input.
# The output matrix is a Numpy matrix of type float32, and suitable for training medGAN.
# Written by Edward Choi (mp2893@gatech.edu)
# Usage: Put this script to the folder where MIMIC-III CSV files are located. Then execute the below command.
# python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv <output file> <"binary"|"count">
# Note that the last argument "binary/count" determines whether you want to create a binary matrix or a count matrix.

# Output files
# <output file>.pids: cPickled Python list of unique Patient IDs. Used for intermediate processing
# <output file>.matrix: Numpy float32 matrix. Each row corresponds to a patient. Each column corresponds to a ICD9 diagnosis code.
# <output file>.types: cPickled Python dictionary that maps string diagnosis codes to integer diagnosis codes.

import numpy as np
from datetime import datetime
import pdb
import pandas as pd
import os
from collections import defaultdict
import gc
from tqdm import tqdm
import json
import jsonlines
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder


def convert_to_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3] + '.' + dxStr[3:]
        else: return dxStr

def convert_to_3digit_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3]
        else: return dxStr

# specify input and output dir
input_dir = './data/MIMIC/files/mimiciii/1.4/'
output_dir = './data/processed'

# build input and output file names
admissionFile = f"{input_dir}/ADMISSIONS.csv"
diagnosisFile = f"{input_dir}/DIAGNOSES_ICD.csv"
prescriptionFile = f"{input_dir}/PRESCRIPTIONS.csv"
procedureFile = f"{input_dir}/PROCEDURES_ICD.csv"
patientFile = f"{input_dir}/PATIENTS.csv"
outputMergeFile = f"{output_dir}/MIMIC-III-Merge.jsonl"
outputTrainFile = f"{output_dir}/MIMIC-III-Merge-train.jsonl"
outputValFile = f"{output_dir}/MIMIC-III-Merge-val.jsonl"
outputTestFile = f"{output_dir}/MIMIC-III-Merge-test.jsonl"
ICDTokenFile = f"{output_dir}/diagnosis_token_list.txt"
DrugTokenFile = f"{output_dir}/drug_token_list.txt"
ProdTokenFile = f"{output_dir}/procedure_token_list.txt"
DrugTokenMapFile = f"{output_dir}/drug_map_list.txt"
outputCatCardinalities = f"{output_dir}/cat_cardinalities.txt"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# get patient features
df_patient = pd.read_csv(patientFile)
df_adm = pd.read_csv(admissionFile)
df_patient = df_patient.sort_values('SUBJECT_ID')
df_adm = df_adm.sort_values('SUBJECT_ID')
first_adm = df_adm[['ADMITTIME','SUBJECT_ID']].groupby('SUBJECT_ID').min()
df_patient['age'] = pd.DatetimeIndex(first_adm['ADMITTIME']).year - pd.DatetimeIndex(df_patient['DOB']).year
patient_feature = df_patient[['age','GENDER','SUBJECT_ID']].rename(columns={'SUBJECT_ID':'pid','GENDER':'gender'})
patient_feature['age'] = StandardScaler().fit_transform(patient_feature['age'].values.reshape(-1,1))
patient_feature['gender'] = OrdinalEncoder().fit_transform(patient_feature['gender'].values.reshape(-1,1))
cat_cardinalities = [patient_feature['gender'].nunique()]
with open(outputCatCardinalities, 'w', encoding='utf-8') as f:
    for cat_card in cat_cardinalities:
        f.write(str(cat_card)+'\n')

pidFeatureMap = {}
for idx in patient_feature.index:
    pid = int(patient_feature.loc[idx].pid)
    pidFeatureMap[pid] = {}
    pidFeatureMap[pid]['x_num'] = patient_feature.loc[idx][['age']].values.tolist()
    pidFeatureMap[pid]['x_cat'] = patient_feature.loc[idx][['gender']].values.tolist()

# save a list of new special tokens: ICD code and others for the language model to learn
ICDTokenList = []
DrugTokenList = []
# LabTokenList = []
ProdTokenList = []

print('Building pid-admission mapping, admission-date mapping')
pidAdmMap = {}
admDateMap = {}
infd = open(admissionFile, 'r')
infd.readline()
for line in infd:
    tokens = line.strip().split(',')
    pid = int(tokens[1])
    # admId is 'HADM_ID' column in the original csv file
    admId = int(tokens[2])
    admId = str(admId) # important!!
    admTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
    admDateMap[admId] = admTime
    if pid in pidAdmMap: pidAdmMap[pid].append(admId)
    else: pidAdmMap[pid] = [admId]
infd.close()

# every HADM_ID corresponds to a sequence of ICD-9 codes of diagnosis
print('Building admission-dxList mapping')
admDxMap = {}
infd = open(diagnosisFile, 'r')
infd.readline()
for line in infd:
    tokens = line.strip().split(',')
    # admId is 'HADM_ID' column in the original csv file
    admId = int(tokens[2])
    admId = str(admId)

    #dxStr = 'D_' + convert_to_icd9(tokens[4][1:-1]) ############## Uncomment this line and comment the line below, if you want to use the entire ICD9 digits.
    dxStr = 'Diag_' + convert_to_3digit_icd9(tokens[4][1:-1])
    if admId in admDxMap: admDxMap[admId].append(dxStr)
    else: admDxMap[admId] = [dxStr]
    ICDTokenList.append(dxStr)
infd.close()
ICDTokenList = np.unique(ICDTokenList).tolist()
with open(ICDTokenFile, 'w', encoding='utf-8') as f:
    for token in ICDTokenList:
        f.write(token + '\n')

# every HADM_ID corresponds to procedure codes
print('Building admission-procedure mapping')
infd = open(procedureFile, 'r')
infd.readline()
admPdMap = defaultdict(list)
for line in tqdm(infd):
    tokens = line.strip().split(',')
    pid = int(tokens[1])
    admId = int(tokens[2])
    admId = str(admId)

    icdCode = tokens[4]
    icdCodeStr = 'Prod_' + convert_to_3digit_icd9(icdCode[1:-1])
    admPdMap[admId].append(icdCodeStr)
    ProdTokenList.append(icdCodeStr)
infd.close()
ProdTokenList = np.unique(ProdTokenList).tolist()
with open(ProdTokenFile, 'w', encoding='utf-8') as f:
    for token in ProdTokenList:
        f.write(token + '\n')

# every HADM_ID corresponds to prescriptions (filter out rare drugs)
print('Building admission-prescription mapping')
df_prescription = pd.read_csv(prescriptionFile, usecols=['HADM_ID', 'STARTDATE', 'DRUG', 'FORM_VAL_DISP'], dtype={'HADM_ID':int, 'STARTDATE':str, 'DRUG':str, 'FORM_VAL_DISP':str}, parse_dates=['STARTDATE'])
drug_ts = df_prescription['DRUG'].value_counts()
drugList = drug_ts[drug_ts>=500].keys().tolist() # filter out rare drugs

# build drug list map
drugTokenMap = {}
with open(DrugTokenMapFile, 'w', encoding='utf-8') as f:
    for idx,drugName in enumerate(drugList):
        drugID = f'Drug_{str(idx)}'
        drugTokenMap[drugName.strip()] = drugID
        f.write(f'{drugID} {drugName}\n')

df_prescription = df_prescription[df_prescription['DRUG'].isin(drugList)]
df_prescription['DRUG'] = df_prescription['DRUG'].map(lambda x: drugTokenMap[x.strip()]) # map drug name to drug ID

admDrugMap = defaultdict(list)
admIdList = df_prescription['HADM_ID'].unique().tolist()

for admId in tqdm(admIdList):
    admId = str(int(admId))
    df_pres = df_prescription[df_prescription['HADM_ID'] == int(admId)]
    df_pres = df_pres.dropna(subset=['STARTDATE'])
    thisDrugList = df_pres['DRUG'].unique().tolist()
    admDrugMap[admId] = thisDrugList

del df_prescription
gc.collect()
with open(DrugTokenFile, 'w', encoding='utf-8') as f:
    for k,v in drugTokenMap.items():
        f.write(str(v)+'\n')

print('''
######################################################################################
# Conclude and save the processed files
######################################################################################
''')

fout = jsonlines.open(outputMergeFile, mode='w')
for pid, admIdList in tqdm(pidAdmMap.items()):
    # pid: patient ID; admIdList: a list of admissions for this patient
    patientDict = defaultdict(list)
    patientDict['pid'] = pid
    if pid in pidFeatureMap:
        patientDict['x_num'] = pidFeatureMap[pid]['x_num']
        patientDict['x_cat'] = pidFeatureMap[pid]['x_cat']

    # adm2diagnosis
    sortedDxList = sorted([(admDateMap[admId], admDxMap[admId]) for admId in admIdList])
    # adm2procedure
    sortedPdList = sorted([(admDateMap[admId], admPdMap[admId]) for admId in admIdList])
    # adm2drug
    sortedDrugList = sorted([(admDateMap[admId], admDrugMap[admId]) for admId in admIdList])
    # adm2labtest
    # sortedLabList = sorted([(admDateMap[admId], admLabMap[admId]) for admId in admIdList])

    patientDict['diagnosis'] = [s[1] for s in sortedDxList]
    patientDict['procedure'] = [s[1] for s in sortedPdList]
    patientDict['drug'] = [s[1] for s in sortedDrugList]
    # patientDict['labtest'] = [s[1] for s in sortedLabList]
    fout.write(patientDict)

print('split train - val - test set')
samples = []
f_out_train = open(outputTrainFile, 'w', encoding='utf-8')
f_out_test = open(outputTestFile, 'w', encoding='utf-8')
f_out_val = open(outputValFile, 'w', encoding='utf-8')
with open(outputMergeFile, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        ind = random.random()
        if ind < 0.1: # 0.1 for test
            f_out_test.write(line)
        elif ind > 0.15: # 0.85 for train
            f_out_train.write(line)
        else: # 0.05 for val
            f_out_val.write(line)
f_out_train.close()
f_out_test.close()
f_out_val.close()
print("Done")
