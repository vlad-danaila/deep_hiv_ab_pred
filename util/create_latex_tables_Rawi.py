from collections import defaultdict
import numpy as np
acc_mean, acc_std = 'acc_mean', 'acc_std'
auc_mean, auc_std = 'auc_mean', 'auc_std'
mcc_mean, mcc_std = 'mcc_mean', 'mcc_std'

'''
Results Rawi
'''
str_antibodies_rawi = '	b12	4E10	2F5	2G12	VRC01	PG9	PGT121	PGT128	PGT145	3BNC117	PG16	10-1074	PGDM1400	VRC26.08	VRC26.25	VRC13	VRC03	VRC-PG04	35O22	NIH45-46	VRC-CH31	8ANC195	HJ16	PGT151	VRC38.01	CH01	PGT135	DH270.1	DH270.5	DH270.6	VRC29.03	VRC34.01	VRC07'
str_acc_rawi = '0.79 (0.01)	0.94 (0)	0.95 (0)	0.91 (0.01)	0.92 (0)	0.86 (0.01)	0.88 (0.01)	0.86 (0.01)	0.86 (0.02)	0.9 (0.01)	0.84 (0.01)	0.94 (0.01)	0.89 (0)	0.85 (0.01)	0.87 (0.01)	0.88 (0.01)	0.81 (0.02)	0.87 (0.01)	0.66 (0.02)	0.89 (0.01)	0.87 (0.01)	0.89 (0.02)	0.66 (0.02)	0.83 (0.01)	0.87 (0.03)	0.77 (0.03)	0.74 (0.02)	0.9 (0.02)	0.91 (0.01)	0.93 (0.01)	0.84 (0.01)	0.79 (0.03)	0.93 (0.01)'
str_auc_rawi = '0.82 (0.01)	0.82 (0.02)	0.97 (0)	0.93 (0)	0.89 (0.01)	0.85 (0.01)	0.92 (0)	0.89 (0.01)	0.86 (0.02)	0.88 (0.02)	0.79 (0.02)	0.95 (0.01)	0.83 (0.02)	0.89 (0.01)	0.89 (0.01)	0.83 (0.01)	0.83 (0.02)	0.78 (0.05)	0.63 (0.02)	0.8 (0.02)	0.78 (0.03)	0.9 (0.03)	0.67 (0.02)	0.78 (0.02)	0.87 (0.02)	0.77 (0.02)	0.77 (0.02)	0.92 (0.02)	0.93 (0.02)	0.93 (0.01)	0.82 (0.02)	0.78 (0.03)	0.78 (0.05)'
str_mcc_rawi = '0.56 (0.02)	0.63 (0.02)	0.89 (0.01)	0.75 (0.01)	0.7 (0.02)	0.61 (0.02)	0.75 (0.01)	0.72 (0.01)	0.67 (0.04)	0.69 (0.03)	0.57 (0.04)	0.86 (0.01)	0.66 (0.02)	0.7 (0.02)	0.71 (0.04)	0.63 (0.03)	0.61 (0.03)	0.57 (0.05)	0.38 (0.04)	0.59 (0.05)	0.6 (0.06)	0.77 (0.04)	0.42 (0.03)	0.58 (0.03)	0.7 (0.05)	0.56 (0.04)	0.54 (0.01)	0.82 (0.03)	0.83 (0.02)	0.85 (0.02)	0.64 (0.02)	0.61 (0.05)	0.66 (0.04)'

antibodies_rawi = str_antibodies_rawi.split()
acc_rawi = str_acc_rawi.split()
auc_rawi = str_auc_rawi.split()
mcc_rawi = str_mcc_rawi.split()

results_rawi = defaultdict(dict)

for i in range(len(antibodies_rawi)):
    ab = antibodies_rawi[i]
    results_rawi[ab][acc_mean] = float(acc_rawi[i * 2])
    results_rawi[ab][acc_std] = float(acc_rawi[i * 2 + 1][1:-1])
    results_rawi[ab][auc_mean] = float(auc_rawi[i * 2])
    results_rawi[ab][auc_std] = float(auc_rawi[i * 2 + 1][1:-1])
    results_rawi[ab][mcc_mean] = float(mcc_rawi[i * 2])
    results_rawi[ab][mcc_std] = float(mcc_rawi[i * 2 + 1][1:-1])

'''
Results FC-ATT-GRU
'''
str_header = 'Run ID,Name,Source Type,Source Name,User,Status,cv_folds_trim,n_trials,prune_trehold,cv mean acc 10-1074,cv mean acc 2F5,cv mean acc 2G12,cv mean acc 35O22,cv mean acc 3BNC117,cv mean acc 4E10,cv mean acc 8ANC195,cv mean acc CH01,cv mean acc DH270.1,cv mean acc DH270.5,cv mean acc DH270.6,cv mean acc HJ16,cv mean acc NIH45-46,cv mean acc PG16,cv mean acc PG9,cv mean acc PGDM1400,cv mean acc PGT121,cv mean acc PGT128,cv mean acc PGT135,cv mean acc PGT145,cv mean acc PGT151,cv mean acc VRC-CH31,cv mean acc VRC-PG04,cv mean acc VRC01,cv mean acc VRC03,cv mean acc VRC07,cv mean acc VRC26.08,cv mean acc VRC26.25,cv mean acc VRC34.01,cv mean acc VRC38.01,cv mean acc b12,cv mean auc 10-1074,cv mean auc 2F5,cv mean auc 2G12,cv mean auc 35O22,cv mean auc 3BNC117,cv mean auc 4E10,cv mean auc 8ANC195,cv mean auc CH01,cv mean auc DH270.1,cv mean auc DH270.5,cv mean auc DH270.6,cv mean auc HJ16,cv mean auc NIH45-46,cv mean auc PG16,cv mean auc PG9,cv mean auc PGDM1400,cv mean auc PGT121,cv mean auc PGT128,cv mean auc PGT135,cv mean auc PGT145,cv mean auc PGT151,cv mean auc VRC-CH31,cv mean auc VRC-PG04,cv mean auc VRC01,cv mean auc VRC03,cv mean auc VRC07,cv mean auc VRC26.08,cv mean auc VRC26.25,cv mean auc VRC34.01,cv mean auc VRC38.01,cv mean auc b12,cv mean mcc 10-1074,cv mean mcc 2F5,cv mean mcc 2G12,cv mean mcc 35O22,cv mean mcc 3BNC117,cv mean mcc 4E10,cv mean mcc 8ANC195,cv mean mcc CH01,cv mean mcc DH270.1,cv mean mcc DH270.5,cv mean mcc DH270.6,cv mean mcc HJ16,cv mean mcc NIH45-46,cv mean mcc PG16,cv mean mcc PG9,cv mean mcc PGDM1400,cv mean mcc PGT121,cv mean mcc PGT128,cv mean mcc PGT135,cv mean mcc PGT145,cv mean mcc PGT151,cv mean mcc VRC-CH31,cv mean mcc VRC-PG04,cv mean mcc VRC01,cv mean mcc VRC03,cv mean mcc VRC07,cv mean mcc VRC26.08,cv mean mcc VRC26.25,cv mean mcc VRC34.01,cv mean mcc VRC38.01,cv mean mcc b12,cv std acc 10-1074,cv std acc 2F5,cv std acc 2G12,cv std acc 35O22,cv std acc 3BNC117,cv std acc 4E10,cv std acc 8ANC195,cv std acc CH01,cv std acc DH270.1,cv std acc DH270.5,cv std acc DH270.6,cv std acc HJ16,cv std acc NIH45-46,cv std acc PG16,cv std acc PG9,cv std acc PGDM1400,cv std acc PGT121,cv std acc PGT128,cv std acc PGT135,cv std acc PGT145,cv std acc PGT151,cv std acc VRC-CH31,cv std acc VRC-PG04,cv std acc VRC01,cv std acc VRC03,cv std acc VRC07,cv std acc VRC26.08,cv std acc VRC26.25,cv std acc VRC34.01,cv std acc VRC38.01,cv std acc b12,cv std auc 10-1074,cv std auc 2F5,cv std auc 2G12,cv std auc 35O22,cv std auc 3BNC117,cv std auc 4E10,cv std auc 8ANC195,cv std auc CH01,cv std auc DH270.1,cv std auc DH270.5,cv std auc DH270.6,cv std auc HJ16,cv std auc NIH45-46,cv std auc PG16,cv std auc PG9,cv std auc PGDM1400,cv std auc PGT121,cv std auc PGT128,cv std auc PGT135,cv std auc PGT145,cv std auc PGT151,cv std auc VRC-CH31,cv std auc VRC-PG04,cv std auc VRC01,cv std auc VRC03,cv std auc VRC07,cv std auc VRC26.08,cv std auc VRC26.25,cv std auc VRC34.01,cv std auc VRC38.01,cv std auc b12,cv std mcc 10-1074,cv std mcc 2F5,cv std mcc 2G12,cv std mcc 35O22,cv std mcc 3BNC117,cv std mcc 4E10,cv std mcc 8ANC195,cv std mcc CH01,cv std mcc DH270.1,cv std mcc DH270.5,cv std mcc DH270.6,cv std mcc HJ16,cv std mcc NIH45-46,cv std mcc PG16,cv std mcc PG9,cv std mcc PGDM1400,cv std mcc PGT121,cv std mcc PGT128,cv std mcc PGT135,cv std mcc PGT145,cv std mcc PGT151,cv std mcc VRC-CH31,cv std mcc VRC-PG04,cv std mcc VRC01,cv std mcc VRC03,cv std mcc VRC07,cv std mcc VRC26.08,cv std mcc VRC26.25,cv std mcc VRC34.01,cv std mcc VRC38.01,cv std mcc b12,global_acc,global_auc,global_mcc,corrections,freeze,input,model,optuna trials,pretrain_epochs,prune,splits,trial'
str_data = '708a375930dd4ff1bb2d1b9686a7ddde,,LOCAL,/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py,root,FINISHED,10,1000,0.05,0.9726771196283391,0.9193173836698858,0.8851809136137493,0.7316402116402118,0.9278454106280194,0.9357835820895523,0.7954558404558405,0.8568297101449273,0.947904761904762,0.9701904761904763,0.9461666666666667,0.7557846153846154,0.963034188034188,0.8389368770764117,0.9047895967270602,0.9254024390243903,0.9280549019607843,0.8978470588235297,0.8269927536231885,0.8705918367346941,0.8803538461538463,0.9434188034188034,0.9442580645161291,0.9509375,0.8828787878787878,0.9553413940256046,0.9708292682926828,0.9486111111111112,0.7561904761904762,0.8636833333333334,0.8021439749608764,0.9900966802237057,0.9462033953828941,0.8843815167987185,0.699084126934488,0.9117984541308419,0.7954258024239255,0.7737859162578344,0.8764605706453134,0.9610437140104682,0.9858273976101708,0.9760267998574722,0.6991521720182794,0.9390702164413264,0.8106749985585893,0.9202381803409929,0.9326172442418357,0.9536252701481005,0.9300168177453171,0.8440257124411537,0.8856617159557162,0.8827349033663277,0.9310144108416929,0.9511201087508837,0.9428636254295654,0.9171541484706836,0.9457713621372303,0.9832710883718921,0.9733131734120443,0.7331745783317213,0.8224907228116884,0.805725947622488,0.9417771381012259,0.8383021237072754,0.661890529201868,0.498359587485109,0.7755523207759647,0.6288051886454801,0.5927003127221329,0.7256791904821117,0.8981102265047586,0.9414492022367371,0.893632000947654,0.4518031779691772,0.868428959108013,0.6157655632602482,0.7658430592075511,0.7988044578341303,0.850006234483945,0.7937345278912963,0.6552235596052262,0.7174544427463936,0.7387692768254027,0.8303500634697191,0.8380811367349604,0.825598602297651,0.7709805006026094,0.8197055034501882,0.9419035477687037,0.8932632182810827,0.5437797925935132,0.6691541829751544,0.5522782895421828,0.023728705503777116,0.02905214167812729,0.045521324797844764,0.07372467672593641,0.03445176254149689,0.040232178315591685,0.07382604074121549,0.06413015587837492,0.048783709782426916,0.03303176510071684,0.040170971062444744,0.0791698406505852,0.03440362637920177,0.05922988959482176,0.036625229671447504,0.03813869340397963,0.03225987422191124,0.03734480360126314,0.07090279777116723,0.04770520243144773,0.05281153496019663,0.046572735929755034,0.03668754175106839,0.02624069775653841,0.052250571721073676,0.10338973847128684,0.02638147931640813,0.03648592650863554,0.09122994297011845,0.06778900886000784,0.04497571754517159,0.012313491437247462,0.02458574645551765,0.062012406092962546,0.09408110755753078,0.05862048441541732,0.1254214406439702,0.1086286807484105,0.07006323799672207,0.04918253495625953,0.023059510300963124,0.028936546294977834,0.09409281734183084,0.09588311110490856,0.08363763816001013,0.04776176297225621,0.05193353431749871,0.02580549072022852,0.03547600469680421,0.08696100036260947,0.05190270213597419,0.07036116556801823,0.07644065854819285,0.048065629564310146,0.04119320848778164,0.04781121251324986,0.10810700954979255,0.019337991318992574,0.02515452707735497,0.125490997408089,0.1150916067829845,0.04937518957835583,0.052441734332831856,0.0577626640188863,0.10676700023599672,0.12710712240350983,0.10758281976625984,0.12773826077566827,0.12948991103840368,0.11553442680121072,0.09194934298003006,0.06416684914538,0.07802046515040348,0.1224788443639934,0.12537077814577907,0.1268203573845277,0.09052049079419518,0.10309265676898821,0.06707090858270867,0.07284149388263488,0.1327711713114875,0.09608171216470299,0.10595636450832668,0.12446233980707107,0.10627726709095546,0.08857260087672097,0.10140601866638896,0.1703915082785775,0.051487071675189226,0.07205910003064378,0.15526098703340008,0.14393764213012677,0.08090499515122443,0.8935184807710627,0.8904467990875278,0.7528124489502408,gaps and amino acids and padding,antb and embed,props-only,fc_att_gru_1_layer,1000,100,treshold 0.05,uniform,252'

header = str_header.split(',')
data = str_data.split(',')

results_fc_att_gru = defaultdict(dict)

for i, txt in enumerate(header):
    ab = txt.split()[-1]
    if 'cv mean acc' in txt:
        results_fc_att_gru[ab][acc_mean] = float(data[i])
    elif 'cv mean auc' in txt:
        results_fc_att_gru[ab][auc_mean] = float(data[i])
    elif 'cv mean mcc' in txt:
        results_fc_att_gru[ab][mcc_mean] = float(data[i])
    elif 'cv std acc' in txt:
        results_fc_att_gru[ab][acc_std] = float(data[i])
    elif 'cv std auc' in txt:
        results_fc_att_gru[ab][auc_std] = float(data[i])
    elif 'cv std mcc' in txt:
        results_fc_att_gru[ab][mcc_std] = float(data[i])

def display_table_row(ab, metrics):
    table_row = f'{ab} & {str(metrics[0])[:4]}({round(metrics[1], 2)}) & ' + \
                f'{str(metrics[2])[:4]}({round(metrics[3], 2)}) & ' + \
                f'{str(metrics[4])[:4]}({round(metrics[5], 2)}) & ' + \
                f'{str(metrics[6])[:4]}({round(metrics[7], 2)}) & ' + \
                f'{str(metrics[8])[:4]}({round(metrics[9], 2)}) & ' + \
                f'{str(metrics[10])[:4]}({round(metrics[11], 2)})\\\\'
    return table_row

totals = np.zeros(12)
for ab, metrics_us in results_fc_att_gru.items():
    metrics_Rawi = results_rawi[ab]
    metrics = np.array([
        metrics_Rawi[mcc_mean], metrics_Rawi[mcc_std],
        metrics_Rawi[auc_mean], metrics_Rawi[auc_std],
        metrics_Rawi[acc_mean], metrics_Rawi[acc_std],
        metrics_us[mcc_mean], metrics_us[mcc_std],
        metrics_us[auc_mean], metrics_us[auc_std],
        metrics_us[acc_mean], metrics_us[acc_std]
    ])
    totals = totals + metrics
    print(display_table_row(ab, metrics))
totals = totals / len(results_fc_att_gru)
print(display_table_row('Average', totals))