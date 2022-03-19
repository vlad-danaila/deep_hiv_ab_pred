from collections import defaultdict
import numpy as np
from deep_hiv_ab_pred.compare_to_SLAPNAP.compute_metrics_for_SLAPNAP import compute_metrics_for_SLAPNAP

acc_mean, acc_std = 'cv_mean_acc', 'cv_std_acc'
auc_mean, auc_std = 'cv_mean_auc', 'cv_std_auc'
mcc_mean, mcc_std = 'cv_mean_mcc', 'cv_std_mcc'

'''
Results SLAPNAP
'''
results_slapnap = compute_metrics_for_SLAPNAP()

'''
Results FC-ATT-GRU
'''
str_header_cv1 = 'Run ID,Name,Source Type,Source Name,User,Status,n_trials,prune_trehold,global_acc,global_auc,global_mcc,test acc 10-1074,test acc 2F5,test acc 2G12,test acc 35O22,test acc 3BNC117,test acc 4E10,test acc 8ANC195,test acc CH01,test acc DH270.1,test acc DH270.5,test acc DH270.6,test acc HJ16,test acc NIH45-46,test acc PG16,test acc PG9,test acc PGDM1400,test acc PGT121,test acc PGT128,test acc PGT135,test acc PGT145,test acc PGT151,test acc VRC-CH31,test acc VRC-PG04,test acc VRC01,test acc VRC03,test acc VRC07,test acc VRC26.08,test acc VRC26.25,test acc VRC34.01,test acc VRC38.01,test acc b12,test auc 10-1074,test auc 2F5,test auc 2G12,test auc 35O22,test auc 3BNC117,test auc 4E10,test auc 8ANC195,test auc CH01,test auc DH270.1,test auc DH270.5,test auc DH270.6,test auc HJ16,test auc NIH45-46,test auc PG16,test auc PG9,test auc PGDM1400,test auc PGT121,test auc PGT128,test auc PGT135,test auc PGT145,test auc PGT151,test auc VRC-CH31,test auc VRC-PG04,test auc VRC01,test auc VRC03,test auc VRC07,test auc VRC26.08,test auc VRC26.25,test auc VRC34.01,test auc VRC38.01,test auc b12,test mcc 10-1074,test mcc 2F5,test mcc 2G12,test mcc 35O22,test mcc 3BNC117,test mcc 4E10,test mcc 8ANC195,test mcc CH01,test mcc DH270.1,test mcc DH270.5,test mcc DH270.6,test mcc HJ16,test mcc NIH45-46,test mcc PG16,test mcc PG9,test mcc PGDM1400,test mcc PGT121,test mcc PGT128,test mcc PGT135,test mcc PGT145,test mcc PGT151,test mcc VRC-CH31,test mcc VRC-PG04,test mcc VRC01,test mcc VRC03,test mcc VRC07,test mcc VRC26.08,test mcc VRC26.25,test mcc VRC34.01,test mcc VRC38.01,test mcc b12,fine_tuning_trials,freeze,input,model,pretrain_epochs,prune,splits,trial'
str_data_cv1 = 'fe4b5e68e85b494ca4b522c737ee107c,,LOCAL,/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py,root,FINISHED,1000,0.05,0.869915313470858,0.8789097755094192,0.6934436773723306,0.9316239316239316,0.927536231884058,0.8602941176470589,0.6551724137931034,0.9285714285714286,0.9136690647482014,0.6226415094339622,0.7959183673469388,0.9285714285714286,1,0.9047619047619048,0.65,0.9636363636363636,0.9213483146067416,0.8852459016393442,0.9224137931034483,0.9166666666666666,0.8446601941747572,0.82,0.9081632653061225,0.9090909090909091,0.9122807017543859,0.8888888888888888,0.9216867469879518,0.8472222222222222,0.9375,0.926829268292683,0.8928571428571429,0.7619047619047619,0.86,0.8082191780821918,0.9720364741641337,0.9443624868282403,0.836036036036036,0.6523297491039426,0.9354157872520051,0.7121504339440694,0.790909090909091,0.8411371237458194,0.9444444444444444,1,0.9565217391304348,0.707250341997264,0.9044444444444445,0.938375350140056,0.9462639109697933,0.9203869047619049,0.9542722759341521,0.8809523809523809,0.830220713073005,0.8883672404799164,0.9428571428571428,0.8553191489361702,0.9534574468085106,0.8509803921568627,0.8843749999999999,0.8106666666666666,0.95375,0.9350227420402859,0.8159090909090909,0.8974358974358974,0.7905525846702317,0.8577507598784194,0.8547009557261552,0.5564711459966303,0.3434840149394188,0.750542916715115,0.5412975099921123,0.4438126822992973,0.5903010033444817,0.8660254037844386,1,0.8260869565217391,0.3328770246548891,0.8751899489873673,0.7735349896567848,0.7689206903228784,0.8022991556224564,0.8076158750939425,0.6772632893595626,0.6112501821507025,0.7675389574256993,0.8020767218098109,0.6706345949012916,0.7694442171660333,0.7162930661527004,0.6959341379213441,0.5146502354656654,0.84625,0.7570332986102252,0.5272727272727272,0.636572388047428,0.5136291487229319,1000,antb and embed,props-only,fc_att_gru_1_layer,100,treshold 0.05,uniform,252'

str_header_cv2 = 'Run ID,Name,Source Type,Source Name,User,Status,n_trials,prune_trehold,global_acc,global_auc,global_mcc,test acc 10-1074,test acc 2F5,test acc 2G12,test acc 35O22,test acc 3BNC117,test acc 4E10,test acc 8ANC195,test acc CH01,test acc DH270.1,test acc DH270.5,test acc DH270.6,test acc HJ16,test acc NIH45-46,test acc PG16,test acc PG9,test acc PGDM1400,test acc PGT121,test acc PGT128,test acc PGT135,test acc PGT145,test acc PGT151,test acc VRC-CH31,test acc VRC-PG04,test acc VRC01,test acc VRC03,test acc VRC07,test acc VRC26.08,test acc VRC26.25,test acc VRC34.01,test acc VRC38.01,test acc b12,test auc 10-1074,test auc 2F5,test auc 2G12,test auc 35O22,test auc 3BNC117,test auc 4E10,test auc 8ANC195,test auc CH01,test auc DH270.1,test auc DH270.5,test auc DH270.6,test auc HJ16,test auc NIH45-46,test auc PG16,test auc PG9,test auc PGDM1400,test auc PGT121,test auc PGT128,test auc PGT135,test auc PGT145,test auc PGT151,test auc VRC-CH31,test auc VRC-PG04,test auc VRC01,test auc VRC03,test auc VRC07,test auc VRC26.08,test auc VRC26.25,test auc VRC34.01,test auc VRC38.01,test auc b12,test mcc 10-1074,test mcc 2F5,test mcc 2G12,test mcc 35O22,test mcc 3BNC117,test mcc 4E10,test mcc 8ANC195,test mcc CH01,test mcc DH270.1,test mcc DH270.5,test mcc DH270.6,test mcc HJ16,test mcc NIH45-46,test mcc PG16,test mcc PG9,test mcc PGDM1400,test mcc PGT121,test mcc PGT128,test mcc PGT135,test mcc PGT145,test mcc PGT151,test mcc VRC-CH31,test mcc VRC-PG04,test mcc VRC01,test mcc VRC03,test mcc VRC07,test mcc VRC26.08,test mcc VRC26.25,test mcc VRC34.01,test mcc VRC38.01,test mcc b12,fine_tuning_trials,freeze,input,model,pretrain_epochs,prune,splits,trial'
str_data_cv2 = '425808c3b38c4f6b926980f9f711ec7a,,LOCAL,/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py,root,FINISHED,1000,0.05,0.8808069213773908,0.8926592734127814,0.7086118762486084,0.9827586206896551,0.9057971014492754,0.875,0.7068965517241379,0.9444444444444444,0.9064748201438849,0.7358490566037735,0.9375,0.9285714285714286,0.9523809523809523,0.8809523809523809,0.75,0.9818181818181818,0.8181818181818182,0.8666666666666667,0.9224137931034483,0.9166666666666666,0.912621359223301,0.8163265306122449,0.8877551020408163,0.7777777777777778,0.9473684210526315,0.9523809523809523,0.9457831325301205,0.8450704225352113,0.9240506329113924,0.9878048780487805,0.9642857142857143,0.7380952380952381,0.8125,0.7808219178082192,0.9937402190923318,0.954945290710148,0.9084821428571428,0.7833333333333334,0.9367283950617284,0.7644628099173554,0.7666666666666666,0.9756521739130435,0.9652777777777778,0.9886363636363635,0.9336384439359269,0.6803977272727272,0.9975845410628019,0.7799331103678929,0.87109375,0.9375661375661375,0.9696663296258847,0.9273109243697479,0.8140350877192982,0.9370314842578711,0.7870370370370371,0.9683794466403162,0.9722222222222223,0.9235278443911538,0.904610492845787,0.9203980099502488,0.9910714285714286,0.9744370054777844,0.782312925170068,0.7714285714285714,0.7908297829174276,0.9646120856989797,0.8095713569330298,0.6705214012982357,0.49180061804999753,0.7575704874652001,0.5008285009824737,0.4328717432773201,0.8757605390397141,0.8561706738324666,0.9082951062292475,0.7616920277666713,0.2075143391598224,0.9383148632568364,0.5164332221429929,0.6527912098338668,0.8145402808450309,0.8202205958567957,0.8028819157482489,0.6182035236467687,0.7687447891285594,0.5,0.8263820186634814,0.8665407417674355,0.7912813192070985,0.694759488155743,0.7052238805970149,0.9759000729485332,0.924119025664048,0.4767312946227962,0.5142653638627285,0.5224256780257229,1000,antb and embed,props-only,fc_att_gru_1_layer,100,treshold 0.05,uniform,252'

str_header_cv3 = 'Run ID,Name,Source Type,Source Name,User,Status,n_trials,prune_trehold,global_acc,global_auc,global_mcc,test acc 10-1074,test acc 2F5,test acc 2G12,test acc 35O22,test acc 3BNC117,test acc 4E10,test acc 8ANC195,test acc CH01,test acc DH270.1,test acc DH270.5,test acc DH270.6,test acc HJ16,test acc NIH45-46,test acc PG16,test acc PG9,test acc PGDM1400,test acc PGT121,test acc PGT128,test acc PGT135,test acc PGT145,test acc PGT151,test acc VRC-CH31,test acc VRC-PG04,test acc VRC01,test acc VRC03,test acc VRC07,test acc VRC26.08,test acc VRC26.25,test acc VRC34.01,test acc VRC38.01,test acc b12,test auc 10-1074,test auc 2F5,test auc 2G12,test auc 35O22,test auc 3BNC117,test auc 4E10,test auc 8ANC195,test auc CH01,test auc DH270.1,test auc DH270.5,test auc DH270.6,test auc HJ16,test auc NIH45-46,test auc PG16,test auc PG9,test auc PGDM1400,test auc PGT121,test auc PGT128,test auc PGT135,test auc PGT145,test auc PGT151,test auc VRC-CH31,test auc VRC-PG04,test auc VRC01,test auc VRC03,test auc VRC07,test auc VRC26.08,test auc VRC26.25,test auc VRC34.01,test auc VRC38.01,test auc b12,test mcc 10-1074,test mcc 2F5,test mcc 2G12,test mcc 35O22,test mcc 3BNC117,test mcc 4E10,test mcc 8ANC195,test mcc CH01,test mcc DH270.1,test mcc DH270.5,test mcc DH270.6,test mcc HJ16,test mcc NIH45-46,test mcc PG16,test mcc PG9,test mcc PGDM1400,test mcc PGT121,test mcc PGT128,test mcc PGT135,test mcc PGT145,test mcc PGT151,test mcc VRC-CH31,test mcc VRC-PG04,test mcc VRC01,test mcc VRC03,test mcc VRC07,test mcc VRC26.08,test mcc VRC26.25,test mcc VRC34.01,test mcc VRC38.01,test mcc b12,fine_tuning_trials,freeze,input,model,pretrain_epochs,prune,splits,trial'
str_data_cv3 = '75b63c8162024fd684cff459f4a800f5,,LOCAL,/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py,root,FINISHED,1000,0.05,0.8820466142989403,0.8942317062657142,0.730965984784343,0.9396551724137931,0.8978102189781022,0.8602941176470589,0.6666666666666666,0.9206349206349206,0.9708029197080292,0.7735849056603774,0.9166666666666666,0.9024390243902439,0.975609756097561,0.9761904761904762,0.6166666666666667,0.9811320754716981,0.8977272727272727,0.9083333333333333,0.9655172413793104,0.9083969465648855,0.883495145631068,0.8367346938775511,0.8673469387755102,0.8333333333333334,0.9473684210526315,0.9193548387096774,0.9698795180722891,0.8732394366197183,0.9367088607594937,0.9382716049382716,0.9523809523809523,0.7073170731707317,0.7916666666666666,0.8082191780821918,0.9703989703989704,0.9111729452054793,0.830026455026455,0.7007389162561577,0.8712871287128713,0.7761627906976745,0.7953869047619048,0.9252173913043478,0.9436274509803921,0.988095238095238,0.9725400457665904,0.7699999999999999,1,0.9202037351443124,0.9207487056949424,0.9821428571428572,0.9324799196787149,0.913713405238829,0.8637992831541219,0.8584280303030303,0.820216049382716,0.9597902097902098,0.9591836734693877,0.984113475177305,0.9198412698412698,0.9623015873015873,0.9816849816849816,0.9736677115987461,0.688095238095238,0.8263736263736264,0.7997448979591837,0.8688462744078409,0.7968834921156437,0.5671714145005305,0.4213367187201608,0.7357600776003063,0.696392422427928,0.5267857142857143,0.8353970306134466,0.7997022133049316,0.9522700151838143,0.9532552922349359,0.3706246583305506,0.9159179958249472,0.7769553956640867,0.7581425055034353,0.9136904761904762,0.802710843373494,0.7611642683570728,0.6429709717149927,0.6908070910282453,0.6324555320336759,0.8486052868659789,0.7505379197758463,0.8844125983629422,0.7620142231167202,0.736765956104158,0.8784664459746295,0.9027048427131934,0.48080770186847693,0.44849169573672276,0.5478984543797384,1000,antb and embed,props-only,fc_att_gru_1_layer,100,treshold 0.05,uniform,252'

str_header_cv4 = 'Run ID,Name,Source Type,Source Name,User,Status,n_trials,prune_trehold,global_acc,global_auc,global_mcc,test acc 10-1074,test acc 2F5,test acc 2G12,test acc 35O22,test acc 3BNC117,test acc 4E10,test acc 8ANC195,test acc CH01,test acc DH270.1,test acc DH270.5,test acc DH270.6,test acc HJ16,test acc NIH45-46,test acc PG16,test acc PG9,test acc PGDM1400,test acc PGT121,test acc PGT128,test acc PGT135,test acc PGT145,test acc PGT151,test acc VRC-CH31,test acc VRC-PG04,test acc VRC01,test acc VRC03,test acc VRC07,test acc VRC26.08,test acc VRC26.25,test acc VRC34.01,test acc VRC38.01,test acc b12,test auc 10-1074,test auc 2F5,test auc 2G12,test auc 35O22,test auc 3BNC117,test auc 4E10,test auc 8ANC195,test auc CH01,test auc DH270.1,test auc DH270.5,test auc DH270.6,test auc HJ16,test auc NIH45-46,test auc PG16,test auc PG9,test auc PGDM1400,test auc PGT121,test auc PGT128,test auc PGT135,test auc PGT145,test auc PGT151,test auc VRC-CH31,test auc VRC-PG04,test auc VRC01,test auc VRC03,test auc VRC07,test auc VRC26.08,test auc VRC26.25,test auc VRC34.01,test auc VRC38.01,test auc b12,test mcc 10-1074,test mcc 2F5,test mcc 2G12,test mcc 35O22,test mcc 3BNC117,test mcc 4E10,test mcc 8ANC195,test mcc CH01,test mcc DH270.1,test mcc DH270.5,test mcc DH270.6,test mcc HJ16,test mcc NIH45-46,test mcc PG16,test mcc PG9,test mcc PGDM1400,test mcc PGT121,test mcc PGT128,test mcc PGT135,test mcc PGT145,test mcc PGT151,test mcc VRC-CH31,test mcc VRC-PG04,test mcc VRC01,test mcc VRC03,test mcc VRC07,test mcc VRC26.08,test mcc VRC26.25,test mcc VRC34.01,test mcc VRC38.01,test mcc b12,fine_tuning_trials,freeze,input,model,pretrain_epochs,prune,splits,trial'
str_data_cv4 = 'e0511d9893b342949696662260d8dfe6,,LOCAL,/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py,root,FINISHED,1000,0.05,0.8677602521808784,0.8817189445802581,0.7061891711773645,0.9741379310344828,0.8978102189781022,0.8308823529411765,0.6842105263157895,0.9523809523809523,0.927007299270073,0.7692307692307693,0.8125,0.975609756097561,0.975609756097561,0.975,0.4,0.9245283018867925,0.875,0.9166666666666666,0.9130434782608695,0.8615384615384616,0.8349514563106796,0.7708333333333334,0.8877551020408163,0.8703703703703703,0.9649122807017544,0.9032258064516129,0.9636363636363636,0.8591549295774648,0.9746835443037974,0.9753086419753086,0.891566265060241,0.7317073170731707,0.8333333333333334,0.773972602739726,0.9851988899167438,0.9521064301552107,0.8518661023470565,0.7123456790123456,0.9303503588011819,0.7049941927990708,0.7987711213517665,0.8663194444444444,1,0.9976190476190475,0.9949494949494949,0.5277777777777778,0.9744186046511628,0.8450793650793651,0.9365712214570496,0.9512195121951219,0.9273202614379085,0.8641686182669789,0.7870370370370371,0.9171882522869522,0.9062049062049062,0.9807692307692308,0.9375,0.9311465721040189,0.8918918918918919,0.904109589041096,0.9925925925925926,0.9406207827260459,0.7511961722488039,0.8197802197802199,0.7521739130434782,0.9471869512231342,0.7943023643058195,0.5296154237881244,0.40299181431190684,0.8361301035252264,0.5326009374423744,0.5207373271889401,0.6255432421712243,0.9513635438635394,0.9523809523809523,0.950950166988625,0.21821789023599236,0.7534883720930232,0.6837493894061094,0.782529902138456,0.7919788463615374,0.7187509246336419,0.6560500505786635,0.531323373049757,0.7488509233491598,0.7416198487095662,0.8996469021204838,0.7472826086956522,0.8475207682870765,0.7344199279357152,0.8055363982396382,0.95,0.7456105750331009,0.47719251741261215,0.578021978021978,0.4362702830062673,1000,antb and embed,props-only,fc_att_gru_1_layer,100,treshold 0.05,uniform,252'

str_header_cv5 = 'Run ID,Name,Source Type,Source Name,User,Status,n_trials,prune_trehold,global_acc,global_auc,global_mcc,test acc 10-1074,test acc 2F5,test acc 2G12,test acc 35O22,test acc 3BNC117,test acc 4E10,test acc 8ANC195,test acc CH01,test acc DH270.1,test acc DH270.5,test acc DH270.6,test acc HJ16,test acc NIH45-46,test acc PG16,test acc PG9,test acc PGDM1400,test acc PGT121,test acc PGT128,test acc PGT135,test acc PGT145,test acc PGT151,test acc VRC-CH31,test acc VRC-PG04,test acc VRC01,test acc VRC03,test acc VRC07,test acc VRC26.08,test acc VRC26.25,test acc VRC34.01,test acc VRC38.01,test acc b12,test auc 10-1074,test auc 2F5,test auc 2G12,test auc 35O22,test auc 3BNC117,test auc 4E10,test auc 8ANC195,test auc CH01,test auc DH270.1,test auc DH270.5,test auc DH270.6,test auc HJ16,test auc NIH45-46,test auc PG16,test auc PG9,test auc PGDM1400,test auc PGT121,test auc PGT128,test auc PGT135,test auc PGT145,test auc PGT151,test auc VRC-CH31,test auc VRC-PG04,test auc VRC01,test auc VRC03,test auc VRC07,test auc VRC26.08,test auc VRC26.25,test auc VRC34.01,test auc VRC38.01,test auc b12,test mcc 10-1074,test mcc 2F5,test mcc 2G12,test mcc 35O22,test mcc 3BNC117,test mcc 4E10,test mcc 8ANC195,test mcc CH01,test mcc DH270.1,test mcc DH270.5,test mcc DH270.6,test mcc HJ16,test mcc NIH45-46,test mcc PG16,test mcc PG9,test mcc PGDM1400,test mcc PGT121,test mcc PGT128,test mcc PGT135,test mcc PGT145,test mcc PGT151,test mcc VRC-CH31,test mcc VRC-PG04,test mcc VRC01,test mcc VRC03,test mcc VRC07,test mcc VRC26.08,test mcc VRC26.25,test mcc VRC34.01,test mcc VRC38.01,test mcc b12,fine_tuning_trials,freeze,input,model,pretrain_epochs,prune,splits,trial'
str_data_cv5 = 'af0ca4ed46a941d8875ddda274b0e01e,,LOCAL,/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py,root,FINISHED,1000,0.05,0.8707055895933805,0.8757778440376518,0.703535320560595,0.9482758620689655,0.8613138686131386,0.9264705882352942,0.7017543859649122,0.9206349206349206,0.9197080291970803,0.7843137254901961,0.8297872340425532,0.925,0.975,0.975,0.5172413793103449,0.9622641509433962,0.8045977011494253,0.8333333333333334,0.9210526315789473,0.8923076923076924,0.9019607843137255,0.7916666666666666,0.9081632653061225,0.8888888888888888,0.9285714285714286,0.9516129032258065,0.9695121951219512,0.8169014084507042,0.9620253164556962,0.9629629629629629,0.9146341463414634,0.725,0.75,0.821917808219178,0.9792927683975788,0.9126402070750648,0.8736772486772487,0.7062499999999999,0.9489235964542,0.819672131147541,0.7351097178683386,0.884963768115942,0.9846547314578006,0.9974937343358395,0.9797979797979799,0.510939510939511,0.9666666666666667,0.8172554347826088,0.8443322981366459,0.9031746031746032,0.9421692727724001,0.9262096774193549,0.8555555555555556,0.9015492253873063,0.9012345679012346,0.900709219858156,0.8948306595365418,0.9379751900760304,0.8589030206677265,0.9880952380952381,0.9901719901719902,0.9501683501683502,0.6791979949874686,0.7098901098901099,0.8476086956521738,0.8907831576024138,0.7435289042277826,0.7660323462854265,0.45519488238817113,0.7666169154562781,0.4945821760052167,0.6020797289396148,0.6836932211734658,0.8597269536210952,0.950950166988625,0.950950166988625,0.09736038343008391,0.8527777777777777,0.535460635374818,0.5847312864165932,0.7916158915529475,0.782274992469413,0.7951465679458907,0.6261585997769191,0.7744757622156312,0.7456011350793257,0.7167132003459626,0.8287754140107481,0.8267747397006819,0.6429452692577902,0.8190441583537076,0.9254954181635742,0.8052074567209673,0.549169647365276,0.367032967032967,0.5786950147106535,1000,antb and embed,props-only,fc_att_gru_1_layer,100,treshold 0.05,uniform,252'

def bold(text):
    return '\\textbf{' + text + '}'

def display_table_row(ab, metrics):
    rawi_mcc = f'{str(metrics[0])[:4]}({round(metrics[1], 2)})'
    rawi_auc = f'{str(metrics[2])[:4]}({round(metrics[3], 2)})'
    rawi_acc = f'{str(metrics[4])[:4]}({round(metrics[5], 2)})'
    net_mcc = f'{str(metrics[6])[:4]}({round(metrics[7], 2)})'
    net_auc = f'{str(metrics[8])[:4]}({round(metrics[9], 2)})'
    net_acc = f'{str(metrics[10])[:4]}({round(metrics[11], 2)})'

    if metrics[0] > metrics[6]:
        rawi_mcc = bold(rawi_mcc)
    else:
        net_mcc = bold(net_mcc)

    if metrics[2] > metrics[8]:
        rawi_auc = bold(rawi_auc)
    else:
        net_auc = bold(net_auc)

    if metrics[4] > metrics[10]:
        rawi_acc = bold(rawi_acc)
    else:
        net_acc = bold(net_acc)

    table_row = f'{ab} & {rawi_mcc} & {rawi_auc} & {rawi_acc} & {net_mcc} & {net_auc} & {net_acc}\\\\'
    return table_row

def compute_metrics_fc_att_gru(str_header, str_data):
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

    return results_fc_att_gru

def display_results(results_slapnap, results_fc_att_gru_cross_valid):
    totals = np.zeros(12)

    del results_slapnap['global_acc']
    del results_slapnap['global_mcc']
    del results_slapnap['global_auc']

    # TEMPORARY FIX
    # del results_slapnap['10-996']

    for ab, metrics_slapnap in results_slapnap.items():
        metrics_slapnap_np = np.array([
            metrics_slapnap[mcc_mean], metrics_slapnap[mcc_std],
            metrics_slapnap[auc_mean], metrics_slapnap[auc_std],
            metrics_slapnap[acc_mean], metrics_slapnap[acc_std]
        ])
        metrics_us = np.zeros(6)
        for results_fc_att_gru in results_fc_att_gru_cross_valid:
            m = results_fc_att_gru[ab]
            metrics_us = metrics_us + np.array([
                m[mcc_mean], m[mcc_std],
                m[auc_mean], m[auc_std],
                m[acc_mean], m[acc_std]
            ])
        metrics_us = metrics_us / len(results_fc_att_gru_cross_valid)
        metrics = np.concatenate((metrics_slapnap_np, metrics_us))
        totals = totals + metrics
        print(display_table_row(ab, metrics))
    totals = totals / len(results_slapnap)
    print(display_table_row('Average', totals))

# TODO Don't forget to check results corectness against script compare_metrics_for_SLAPNAP.py
display_results(results_slapnap, [
    compute_metrics_fc_att_gru(str_header_cv1, str_data_cv1),
    compute_metrics_fc_att_gru(str_header_cv2, str_data_cv2),
    compute_metrics_fc_att_gru(str_header_cv3, str_data_cv3),
    compute_metrics_fc_att_gru(str_header_cv4, str_data_cv4),
    compute_metrics_fc_att_gru(str_header_cv5, str_data_cv5)
])