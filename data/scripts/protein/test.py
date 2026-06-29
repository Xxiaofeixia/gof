import pandas as pd
# 只跳过 ##，保留 #Uploaded_variation 表头
with open('/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/master_vep_output.txt') as f:
    lines = [l for l in f if not l.startswith('##')]
import io
df = pd.read_csv(io.StringIO(''.join(lines)), sep='\t', low_memory=False)
for c in ['MutPred_score', 'SpliceAI_pred_DS_AG', 'SpliceAI_pred_DS_AL', 'SpliceAI_pred_DS_DG', 'SpliceAI_pred_DS_DL']:
    if c in df.columns:
        vals = df[c].dropna()
        non_dot = int((df[c].fillna('.') != '.').sum())
        print(f'{c}: 非空={len(vals)}, 非.= {non_dot}, 前5= {df[c].head(5).tolist()}')
    else:
        print(f'{c}: 列不存在')