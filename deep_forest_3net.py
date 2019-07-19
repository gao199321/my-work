import deepwalk.graph as graph
from gensim.models import Word2Vec
import networkx  as nx
import random
import math
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from gcforest.gcforest import GCForest
import random
def mat_uniform(mat):
    col_min=np.ones(mat.shape[1])
    col_max= np.ones(mat.shape[1])
    for i in range(mat.shape[1]):
        col_min[i]=np.min(mat[:,i])
        col_max[i]=np.max(mat[:,i])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            low=(col_max[j])-(col_min[j])
            mat[i,j]=(mat[i,j]-col_min[j])/low
    return mat
def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"]=2
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config
def getkolddata(data):
    result=[]
    lens=int(len(data)/5)
    data_new=data.copy()
    for i in range(4):
        data=np.array(data_new)
        indices = np.random.choice(data,lens,replace=False)
        result.append(indices)
        data_new=[]
        for val in data:
            if val not in indices:
                data_new.append(val)
    result.append(data_new)
    return result

def compute(deep,i,j):
    l1=deep[i,:]
    l2=deep[j,:]
    sum1=0.00
    sum2=0.00
    sum3=0.00
    for i in range(len(l1)):
        sum1+=(l1[i]*l2[i])
        sum2+=(l1[i]*l1[i])
        sum3+=(l2[i]*l2[i])
    return sum1/((math.sqrt(sum2))*(math.sqrt(sum3)))
def MBSI(mRNA_similaritity,data_train):
    MBSI_result=np.ones((data_train.shape[1],data_train.shape[0]))*0
    for i in range(data_train.shape[1]):
        for j in range(data_train.shape[0]):
            val=0.00
            sum_up=0.00
            sum_low=0.00
            for l in range(data_train.shape[0]):
                if(l!=j):
                    sum_up+=(mRNA_similaritity[j,l]*data_train[l,i])
                    sum_low+=(mRNA_similaritity[j,l])
            val=sum_up/sum_low
            MBSI_result[i,j]=val
    return MBSI_result
def DBSI(disease_similaritity,data_train):
    DBSI_result=np.ones((data_train.shape[1],data_train.shape[0]))*0
    for i in range(data_train.shape[1]):
        for j in range(data_train.shape[0]):
            val=0.00
            sum_up=0.00
            sum_low=0.00
            for l in range(data_train.shape[1]):
                if(l!=i):
                    sum_up+=(disease_similaritity[i,l]*data_train[j,l])
                    sum_low+=(disease_similaritity[i,l])
            val=sum_up/sum_low
            DBSI_result[i,j]=val
    return DBSI_result
def DBSI_uniform(result_mat):
    for i in range(result_mat.shape[0]):
        for j in range(result_mat.shape[1]):
            low=(np.max(result_mat[i,:]))-(np.min(result_mat[i,:]))
            result_mat[i,j]=(result_mat[i,j]-np.min(result_mat[i,:]))/low
    return result_mat
def MBSI_uniform(result_mat):
    for i in range(result_mat.shape[0]):
        for j in range(result_mat.shape[1]):
            low=(np.max(result_mat[:,j]))-(np.min(result_mat[:,j]))
            result_mat[i,j]=(result_mat[i,j]-np.min(result_mat[:,j]))/low
    return result_mat
mRNA_name=open('new_mRNAname.txt')
disease_name=open('new_diseasename.txt')
gene_name=open('new_genename.txt')
disease=[]
mRNA=[]
gene=[]
disease_dict={}
mRNA_dict={}
gene_dict={}
for line in disease_name.readlines():
    disease.append(line.strip())
for line in mRNA_name.readlines():
    mRNA.append(line.strip())
for line in gene_name.readlines():
    gene.append(line.strip())

i=0
for strs in disease:
    disease_dict[strs]=i
    i=i+1
i=0
for strs in mRNA:
    mRNA_dict[strs]=i
    i=i+1
i=0
for strs in gene:
    gene_dict[strs]=i
    i=i+1
disease_name.close()
mRNA_name.close()
gene_name.close()
str_name='StomachNeoplasms'
label=disease_dict[str_name]
DG=np.loadtxt('matdata/DG_mat.txt',dtype=int)
MD=np.loadtxt('matdata/MD_mat.txt',dtype=int)
MG=np.loadtxt('matdata/MG_mat.txt',dtype=int)
GG=np.loadtxt('matdata/GG_mat.txt',dtype=int)
#******************交叉验证生成训练矩阵*********************
top_num=[10, 20, 30, 40,50,60,70,80,90,100]
average_pre=np.ones(10)*0
average_rec=np.ones(10)*0
all_fpr=np.ones(69)*0
all_tpr=np.ones(69)*0
average_auc=[]

disease_select=MD[:,label]
disease_select.reshape(-1)
index = np.nonzero(disease_select)
index = np.array(index)[0]
#indices = np.random.choice(index,int(len(index)/5), replace=False)
train_index=getkolddata(index)
index_train=[]
index_train.append(random.randint(0,len(train_index)-1))
for m in index_train:
    indices=train_index[m]
    has_assos=[]
    for val in index:
        if val not in indices:
            has_assos.append(val)
    MD_copy = MD.copy()
    for val in indices:
        MD_copy[val,label]=0

    a1=MD[:,label]
    a1_one=np.ones(len(a1))
    a1_zero=a1_one-a1
    result_false=np.nonzero(a1_zero)[0]

    G = nx.Graph()
    node_counts = len(disease) + len(gene) + len(mRNA)
    for i in range(1, node_counts + 1):
        G.add_node(i)
    for i in range(DG.shape[0]):
        for j in range(DG.shape[1]):
            if (DG[i, j] == 1):
                G.add_edge(i+1, len(disease) + j + 1)
    for i in range(MD_copy.shape[0]):
        for j in range(MD_copy.shape[1]):
            if (MD_copy[i,j]==1):
                G.add_edge(len(disease) + len(gene) + i + 1, j + 1)
    for i in range(MG.shape[0]):
        for j in range(MG.shape[1]):
            if (MG[i,j]==1):
                G.add_edge(len(disease)+len(gene)+i+1, len(disease) + j + 1)

    for i in range(GG.shape[0]):
        for j in range(GG.shape[1]):
            if(GG[i,j]==1):
                G.add_edge(len(disease)+1,len(disease)+1)

    kk=G.number_of_edges()

    key_len=128
    nx.write_adjlist(G,"deepforest.adjlist")
    Gra = graph.load_adjacencylist("deepforest.adjlist")
    walks = graph.build_deepwalk_corpus(Gra,num_paths=100,path_length=50,alpha=0,rand=random.Random(0))
    model = Word2Vec(walks,size=key_len,window=5,min_count=0,sg=1,hs=1,workers=1)
    model.wv.save_word2vec_format("deepforest.txt")
#*****************根据deepwalk结果开始计算相似矩阵***********
    deepwork_data = open("deepforest.txt")
    fr = deepwork_data.readlines()
    deep_mRNA=np.ones((len(mRNA),key_len))*0
    deep_disease=np.ones((len(disease),key_len))*0
    for i in range(1,len(fr)):
        da=fr[i].split(" ")
        if(int(da[0])<(len(disease)+1)):
            for j in range(1,len(da)):
                deep_disease[int(da[0])-1,j-1]= (float(da[j]))
        if(int(da[0])>(len(disease)+len(gene))):
            for j in range(1,len(da)):
                deep_mRNA[int(da[0])-len(disease)-len(gene)-1,j-1] = (float(da[j]))
#************************生成测试集和验证集****************************
    tests_len=len(mRNA)-len(has_assos)
    trains_len=len(mRNA)*len(disease)-tests_len
    trains_label=[]
    trains_data=[]
    tests_label=[]
    tests_data=[]
    train_true=[]
    train_false=[]
    really_train_false=[]

    trains_label_sum=[]
    trains_data_sum=[]
    tests_label_sum=[]
    tests_data_sum=[]
    train_true_sum=[]
    train_false_sum=[]
    really_train_false_sum=[]

    for i in range(MD_copy.shape[0]):
        for j in range(MD_copy.shape[1]):
            if(j!=label):
                trains_label.append(MD_copy[i,j])
                da=[]
                da1=(deep_mRNA[i]+deep_disease[j])/2
                for p in range(key_len):
                    da.append(deep_mRNA[i,p])
                for q in range(key_len):
                    da.append(deep_disease[j,q])
                trains_data.append(da)
                trains_data_sum.append(list(da1))
                if(MD_copy[i,j]==1):
                    train_true.append(da)
                    train_true_sum.append(list(da1))
                else:
                    train_false.append(da)
                    train_false_sum.append(list(da1))
    for i in range(len(has_assos)):
        da=[]
        da1 = (deep_mRNA[has_assos[i]] + deep_disease[label]) / 2
        for p in range(key_len):
            da.append(deep_mRNA[has_assos[i],p])
        for q in range(key_len):
            da.append(deep_disease[label,q])
        trains_data.append(da)
        trains_label.append(1)
        train_true.append(da)

        trains_data_sum.append(list(da1))
        trains_label_sum.append(1)
        train_true_sum.append(list(da1))
    for i in range(len(indices)):
        da=[]
        da1 = (deep_mRNA[indices[i]]+deep_disease[label]) / 2
        for p in range(key_len):
            da.append(deep_mRNA[indices[i],p])
        for q in range(key_len):
            da.append(deep_disease[label,q])
        tests_data.append(da)
        tests_label.append(1)

        tests_data_sum.append(list(da1))
        tests_label_sum.append(1)
    for i in range(len(result_false)):
        da=[]
        da1 = (deep_mRNA[indices[i]] + deep_disease[label])/2
        for p in range(key_len):
            da.append(deep_mRNA[result_false[i],p])
        for q in range(key_len):
            da.append(deep_disease[label,q])
        tests_data.append(da)
        tests_label.append(0)

        tests_data_sum.append(list(da1))
        tests_label_sum.append(0)
#***************************随机选择获取真正的训练数据集*******************************
    get_trian = np.random.choice(a=len(train_false),size=len(train_true),replace=False)
    for num in get_trian:
        really_train_false.append(train_false[num])
    tf_len=len(train_true)+len(really_train_false)
    really_train_label=np.zeros(tf_len,int)
    for i in range(len(train_true)):
        really_train_label[i]=1
    for i in range(len(really_train_false)):
        train_true.append(really_train_false[i])
    train_true=np.array(train_true)
    really_train_label=np.array(really_train_label)

    get_trian_sum= np.random.choice(a=len(train_false_sum), size=len(train_true_sum), replace=False)
    for num in get_trian_sum:
        really_train_false_sum.append(train_false_sum[num])
    tf_len_sum=len(train_true_sum)+len(really_train_false_sum)
    really_train_label_sum=np.zeros(tf_len_sum,int)
    for i in range(len(train_true_sum)):
        really_train_label_sum[i]=1
    for i in range(len(really_train_false_sum)):
        train_true_sum.append(really_train_false_sum[i])
    train_true_sum=np.array(train_true_sum)
    really_train_label_sum=np.array(really_train_label_sum)
#************************进行模型训练和预测**********************
    config=get_toy_config()
    model_sum=GCForest(config)
    model_sum.set_keep_model_in_mem(False)
    model_sum.fit_transform(np.array(train_true_sum),np.array(really_train_label_sum))
    weights_sum=model_sum.predict_proba(np.array(tests_data_sum))
    score1_sum=[]
    score2_sum=[]
    for i in range(len(weights_sum)):
        score1_sum.append(weights_sum[i,0])
        score2_sum.append(weights_sum[i,1])
    fpr_sum, tpr_sum, threshold_sum = metrics.roc_curve(tests_label_sum,score2_sum)  ###计算真正率和假正率
    roc_auc_sum=metrics.auc(fpr_sum, tpr_sum)

    model=GCForest(config)
    model.set_keep_model_in_mem(False)
    model.fit_transform(np.array(train_true),np.array(really_train_label))
    weights=model.predict_proba(np.array(tests_data))
    score1=[]
    score2=[]
#*****************计算正确率和召回率**************************
    for i in range(len(weights)):
        score1.append(weights[i,0])
        score2.append(weights[i,1])
    top_result_pre = []
    score2 = np.array(score2)
    index_mRNA=np.argsort((-1)*score2)
    top_result_rec = []
    for i in range(10):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        val = top_num[i]
        for j in range(val):
            ab=index_mRNA[j]
            bb=len(indices)
            if (index_mRNA[j]<len(indices)):
                TP += 1
            else:
                FP += 1
        for j in range(val,len(weights)):
            if ((index_mRNA[j])<len(indices)):
                FN += 1
            else:
                TN += 1
        top_result_pre.append(TP/val)
        top_result_rec.append(TP/(TP+FN))
    average_pre += np.array(top_result_pre)
    average_rec += np.array(top_result_rec)

    fpr,tpr, threshold =metrics.roc_curve(tests_label,score2)  ###计算真正率和假正率
    roc_auc = metrics.auc(fpr,tpr)
    average_auc.append(roc_auc)
    lw = 2
    plt.figure()
    #plt.plot(fpr,tpr, color='darkorange',lw=lw, label='ROC curve (area=%0.2f)' % roc_auc)
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='A与B拼接 (area=%0.2f)' % roc_auc)
    plt.plot(fpr_sum, tpr_sum, color='green', lw=lw, label='A与B平均 (area=%0.2f)' % roc_auc_sum)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(str_name)
    plt.legend(loc="lower right")

average_pre = list(average_pre)
average_rec = list(average_rec)

for ii in range(len(average_auc)):
    print(average_auc[ii])
fig = plt.figure()
plt.bar(top_num,average_pre,2,color="green")
plt.xlabel("top-k")
plt.ylabel("PRE")
plt.title(str_name)
fig1 = plt.figure()
plt.bar(top_num, average_rec,2,color="green")
plt.xlabel("top-k")
plt.ylabel("REC")
plt.title(str_name)
plt.show()




