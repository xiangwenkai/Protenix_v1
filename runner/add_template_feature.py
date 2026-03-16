import os
import pandas as pd
import hashlib


test_seqs = pd.read_csv('/inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix/rna_data/rna_sequences_unique.txt',header=None)
test_seqs.columns = ['sequence']
test_seqs['num_id_pre'] = [f"ID{i}" for i in range(test_seqs.shape[0])]
test_seqs['ID_pre'] = test_seqs['sequence'].apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())
rna_tem = pd.read_csv(f'/inspire/ssd/project/sais-bio/public/xiangwenkai/project/submission_simple.csv')
rna_tem = rna_tem.rename(columns={'ID': 'num_id'})
rna_tem['num_id_pre'] = rna_tem['num_id'].apply(lambda x: x.split('_')[0])
rna_tem['num_id_suf'] = rna_tem['num_id'].apply(lambda x: x.split('_')[1])
rna_tem = pd.merge(rna_tem, test_seqs[['num_id_pre', 'ID_pre']], on='num_id_pre', how='left')
rna_tem['ID'] = rna_tem.apply(lambda x: x['ID_pre'] + '_' + x['num_id_suf'], axis=1)
rna_tem = rna_tem.drop(['num_id', 'num_id_pre', 'num_id_suf', 'ID_pre'], axis=1)
rna_tem.to_csv(f"/inspire/ssd/project/sais-bio/public/xiangwenkai/Protenix_v1/release_data/rna_templates.csv", index=False)


