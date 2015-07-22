#-*- coding: utf-8 -*-
import os
import re
import MeCab
import math
import sys
sys.path.append('LibSVMをダウンロードしたディレクトリ/libsvm-3.12/python/')
from collections import Counter
import random
import numpy 
from numpy import *
from scipy import linalg
import matplotlib.pyplot as plt
import sklearn.decomposition
def near(vector, center_vectors):
    """
    vectorに対し、尤も近いラベルを返す
    :param vector:
    :param center_vectors:
    :return:
    """
    euclid_dist = lambda vector1, vector2: (sum([(vec[0]-vec[1])**2 for vec in list(zip(vector1, vector2))]))**0.5#式です
    d = [euclid_dist(vector, center_vector) for center_vector in center_vectors]
    return d.index(min(d))#場所だけ返す


def clustering(vectors, label_count, learning_count_max=1000):
    """
    K-meansを行い、各ラベルの重心を返す
    :param vectors:
    :param label_count:
    :param learning_count_max:
    :return:
    """
    import random
    #各vectorに割り当てられたクラスタラベルを保持するvector
    label_vector = [random.randint(0, label_count-1) for i in vectors]
    #一つ前のStepで割り当てられたラベル。終了条件の判定に使用
    old_label_vector = list()
    #各クラスタの重心vector
    center_vectors = [[0 for i in range(len(vectors[0]))] for label in range(label_count)]#label_count=3,venters[0]=3,(3,2)=0
    # print "center_vectors",len(center_vectors)
    # for i in range(len(center_vectors)):
    #     print center_vectors[i]
    print len(center_vectors)
    for step in range(learning_count_max):
        #各クラスタの重心vectorの作成
        for vec, label in zip(vectors, label_vector):
            # print vec,label
            center_vectors[label] = [c+v for c, v in zip(center_vectors[label], vec)]#center_vectorと同じラベルのxとyを足して合計を出している。
        for i, center_vector in enumerate(center_vectors):
            center_vectors[i] = [v/label_vector.count(i) for v in center_vector]#x,yの平均を求めている
        #各ベクトルのラベルの再割当て
        for i, vec in enumerate(vectors):
            label_vector[i] = near(vec, center_vectors)
        #前Stepと比較し、ラベルの割り当てに変化が無かったら終了
        if old_label_vector == label_vector:
            break
        #ラベルのベクトルを保持
        old_label_vector = [l for l in label_vector]
    return center_vectors,label_vector

# # 書き込む文字列
# str = """1 1:1.0 2:1.0 3:1.1
# 2 1:2.0 2:1.2 3:2.1
# 3 1:2.3 2:1.5 3:1.1"""
 
# f = open('train_data.txt', 'w') # 書き込みモードで開く
# f.write(str) # 引数の文字列をファイルに書き込む
# f.close() # ファイルを閉じる

# fp = open('test.txt','w')
# fp.write('This is a test.')
# fp.write('this is a pen.')
# fp.close()
# fp = open('test.txt','a')
# fp.write('Additional Texts\n')
# fp.write('Additional Texts')
# fp.write('Additional Texts\n')
# fp.close()

def get_file_contents(path):
	file = os.listdir(path)
	doc = []
	for i in sorted(file):
		f = open(path+"/"+i,'r')#""でそのフォルダの中を探せでiでファイルを指定
		doc.append(f.read())
	# f = open("data/p601.json",'r')
	# doc.append(f.read())
	# file = os.listdir("pos_data")
	# for i in sorted(file):
	# 	f = open("pos_data/"+i,'r')#""でそのフォルダの中を探せでiでファイルを指定
	# 	doc.append(f.read())
	return doc


def calc_tf(sentence):
	m = MeCab.Tagger ("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")#メカブを使う
	# f = open(file_pass, 'r')#00000370.txtフィアルを開く
	# doc=f.read()#ファイルの内容をdocにいれる
	nobe = 0
	hinshinobe = 0
	# doc.decode("utf-8")
	# print doc
	# print m.parse(doc)
	cd = m.parse(sentence)#メカブを使用したものをcdに入れる
	list1 = []#名詞のみを入れるリスト
	cp2 = cd.split()#真ん中の空白をなくす
	# # for i in range(len(cp2)):#
	# # 	j = i%2
	# # 	if j == 0:
	# # 		print cp2[i], cp2[i+1]
	# print "異なり語数"
	word = cp2[0::2] #偶数だけとるプログラム
	hinshi = cp2[1::2] #奇数だけ取るプログラム
	# for i in range(len(word)):#メカブの情報を取り出すがエラーをはく
	# 	# print word[i],hinshi[i]

	for i in range(len(hinshi)):#1から品詞の数分繰り返す
		spell = re.search("名詞",hinshi[i])#名詞と書いているものを取り出す
		if spell:#一致した時
			list1.append(word[i])#wordの配列に入れる
	tf = {}#リストをつくる
	for tfword in list1:#名詞のリストからすべてを取り出す
		if tfword not in tf:#リストの中に今までのものが入ってないときに
			tf[tfword] = 0#ｔｆに0を入れる
		tf[tfword] += 1#数をふやす

	# for z in list1:#正規化をする
	# 	print z,tf[z]#単語と頻度の表示
		# tf[z] = 1.0*tf[z]/len(list1)#正規化シたものの表示
	nomalize_tf = {}
		#### 単語をキーとして正規化した値を保存 ####
	for k, l in tf.items():
		nomalize_tf[k] = 1.0 * l / len(list1)

	return nomalize_tf

def calc_df(tf):
	df = {}#dfのディクショナリを宣言
	for dfword in tf:	#tfの配列を渡してもらう
		for df2word in dfword.keys():	#tfで取り出した文字列を繰り返す
			if df2word not in df:	#ユニークをする
				df[df2word] = 0	#単語を新しく入れる
			df[df2word] += 1 #dfの値を入れる
	return df

def calc_idf(df,sentencenum):
	idf={}#idfのディクショナリを宣言
	for idfword in df.keys():#dfの単語を繰り返す
		idf[idfword] = math.log(1.0*sentencenum/df[idfword])+1#idfを求める
	return idf

def calc_tfidf(tf,idf):
	tfidfdic = {}#ディクショナリ
	tfidf = []#リスト（結果）をいれる
	for tfdoc in tf:#一つの文章をとりだす
		for tfword in tfdoc.keys():# 文字に入れる。
			tfidfdic[tfword] = tfdoc[tfword] * idf[tfword]#tfidfに値をいれる
		tfidf.append(tfidfdic)#一致した時にtfidfにいれる
		tfidfdic = {}#再初期化
	# print tf[0]["KDDI"]
	# print idf["KDDI"]
	return tfidf
#ベクトルの大きさを計算するプログラム
def calc_scalar(vector):
    scalar_vector = 0
    for i in range(len(vector)):
        scalar_vector += vector[i] * vector[i]
    scalar_vector = math.sqrt(scalar_vector)
    return scalar_vector
#cos類似度を出力するプログラム
def cosine_similarity(vector1,vector2):
    denominator = 0#分母
    numerator = 0#分子
    for i in range(len(vector2)):
        numerator += vector1[i] * vector2[i]
    scalar_vector1 = calc_scalar(vector1)
    scalar_vector2 = calc_scalar(vector2)
    denominator = scalar_vector1 * scalar_vector2
    if denominator == 0:
    	print "0",scalar_vector1,scalar_vector2,vector1
    	denominator = 1
    	numerator = 0
    return numerator / denominator

#コサイン尺度の1に近い順位rankを返す番号は0からなので注意
def cosine_similarity_rank(vector,ob_vector,rank):
    cos_sim=[]#cos_simをいれるリスト
    high_list=[]#cos_simの高いものを入れるリスト[3,2,1]は3、2、1の順に高いものが入っていたという事
    for i in range(len(vector)):
        cos_sim.append(cosine_similarity(vector[i],ob_vector))#コサイン類似度を入れたリストを作成 
        if cos_sim == 0:
        	print "cos_sim=0",i   
    for i in range(rank):
        zantei = 0
        while zantei in high_list:
            zantei += 1
        for j in range(len(cos_sim)):
            # print "a"
            if j not in high_list:
                # print "j",j,"zantei",zantei
                if cos_sim[j] > cos_sim[zantei]:
                    zantei = j
        high_list.append(zantei)
    return high_list

def extract_body(data_file):
	ary5=[]
	section5=[]
	body51=[]

	#1000件のデータを入れたモノを作る

	#aryは20この配列であり、そのひとつひとつに50の文が入っている。
	for i in range(len(data_file)):
		# ary = review5[i].split("},{")
		# print i
		ary5.append(data_file[i].split("},{"))
	# ary = review5[0].split("},{")
		# print len(ary[i])
	# print len(ary)
	# print ary[0][49]

	#50の文をtitleとbodyとrateに分類する。
	for i in range(len(ary5)):
		for j in range(len(ary5[i])):
			section5.append(ary5[i][j].split(","))

	# print section[999][1]
	#bodyから必要な情報だけをとりだす。"",body,\nなどの文字をけしている.
	for i in range(len(section5)):
		section5[i][1] = section5[i][1].replace("\"body\":\"","")
		section5[i][1] = section5[i][1].rstrip("\"") 
		section5[i][1] = section5[i][1].replace("\\n","")
		section5[i][1] = section5[i][1].lstrip("|")
		body51.append(section5[i][1])

	# star1 = list(set(star1))
	# section5 = list(set(section5))

	# print section[0][1]
	# for i in range(0,1000):
		# print section5[i][1]
	# print "body5の数",len(body5)
	body51 = list(set(body51))
	print "もらったファイルのbody数",len(body51)
	# for i in range(len(body5)):
	# 	print body5[i]

	# for i in range(0,1000):
	# 	body5.append(body51[i])
	# print "body5の数",len(body5)

	return body51

def tfidf_process(body_file):
	df5=[]
	idf5=[]
	tfidf5=[]
	for sentence in body_file:#星5のtfを求める
		tf5.append(calc_tf(sentence))
	df5 = calc_df(tf5)
	idf5 = calc_idf(df5,len(body_file))
	tfidf5 = calc_tfidf(tf5,idf5)
	return tfidf5

if __name__=="__main__":
	rest_idf5 = []
	rest_df5 =[]
	rest_tf5 = []
	rest_idf1 = []
	rest_df1 = []
	rest_tf1 = []
	review = []
	idf5 = []
	df5 =[]
	tf5 = []
	review5 = []
	idf1 = []
	df1 = []
	tf1 = []
	review1 = []
	review51 = []
	review11 = []
	rest_body5 = []
	rest_body1 = []
	body = []
	body5=[]
	body1=[]
	body51 = []
	body11 = []
	rate = []
	star5 = []
	star1 = []
	attrs = {}
	words = {}
	sim = []
	rest_tfidf5=[]
	rest_tfidf1=[]
	letter_tfidf=[]#0701文書のtfidfが入ったリスト
	class_data=[]#クラスタリングに渡すデータ
	num = 0
	tag = 1
	j = 0
	i = 0
	k = 0
	nazo = []
	dnn = []
	tate = 0
	yoko = 0
	gyou = 0
	gyou1 = 0
	gyou5 = 0
	rest_gyou1 = 0
	rest_gyou5 = 0
	fp = open('rdata_class_kmeans.csv','w')
	# fpp = open('test_data_ad.csv','w')
	review5 = get_file_contents("/home/seko/text_mining/rakuten_data_program/pos_data2")
	# review1 = get_file_contents("neg_data2")
	# review51 = get_file_contents("pos_data")
	# review11 = get_file_contents("neg_data")
	# for r5 in review5:
	# 	review.append(r5)
	# for r1 in review1:
	# 	review.append(r1)
	# for r5 in review51:
	# 	review.append(r5)
	# for r1 in review11:
	# 	review.append(r1)
	for r5 in review5:
		review51.append(r5)
	# for r1 in review1:
	# 	review11.append(r1)
	print "5",len(review5)#フォルダの数
	# print "1",len(review1)#フォルダの数
	# print "5_1",len(review51)#フォルダの数
	# print "1_1",len(review11)#フォルダの数
	# print "all",len(review)#フォルダの数
	body51 = extract_body(review51)#bodyだけ取り出す
	# body11 = extract_body(review11)#bodyだけ取り出す
	for i in range(0,500): #＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃ここで数を変更
	# for i in range(20,30):
		body5.append(body51[i])
	print "body5の数",len(body5)
	for i in range(len(body5)):
		print i,body5[i]
	# print body5[31],body5[69]
	# for i in range(0,1000):
	# # for i in range(20,30):
	# 	body1.append(body11[i])
	# print "body1の数",len(body1)

	# for i in range(1000,len(body51)):
	# # for i in range(1010,1020):
	# 	rest_body5.append(body51[i])
	# for i in range(1000,len(body11)):
	# # for i in range(1010,1020):
	# 	rest_body1.append(body11[i])
	# print "rest_body5",len(rest_body5)
	# print "rest_body1",len(rest_body1)
	# # for i in range(0,1000):
	# # 	print body1[i]
	# for i in range(0,10):
	# 	print body5[i]

	# tfidf5=tfidf_process(body5)
	for sentence in body5:#星5のtfを求める
		tf5.append(calc_tf(sentence))
	df5 = calc_df(tf5)
	idf5 = calc_idf(df5,len(body5))
	tfidf5 = calc_tfidf(tf5,idf5)

	# hatena = 0
	for title5 in tfidf5:	#tfidf1の配列を渡してもらう
		for title5_word in title5.keys():	#tfidfで取り出した文字列を繰り返す
			# hatena += 1
			# print title1_word, title1[title1_word]#tfidfの単語と状態表示
			#/////////////////////////////////////////////////////////////////////////1行のプログラムは一時的に消す
			# if title5_word not in words:#言葉にタグをつけるプログラム
			words[title5_word] = num
			num = num + 1
	# print "hatena",hatena,"num",num
	# # tfidf1=tfidf_process(body1)
	# for sentence in body1:#星1のtfを求める
	# 	tf1.append(calc_tf(sentence))#
	# df1 = calc_df(tf1)
	# idf1 = calc_idf(df1,len(body1))
	# tfidf1 = calc_tfidf(tf1,idf1)

	# # rest_tfidf5=tfidf_process(rest_body5)
	# for sentence in rest_body5:#星5のtfを求める
	# 	rest_tf5.append(calc_tf(sentence))
	# rest_df5 = calc_df(rest_tf5)
	# rest_idf5 = calc_idf(rest_df5,len(rest_body5))
	# rest_tfidf5 = calc_tfidf(rest_tf5,rest_idf5)

	# # rest_tfidf1=tfidf_process(rest_body1)
	# for sentence in rest_body1:#星1のtfを求める
	# 	rest_tf1.append(calc_tf(sentence))#
	# rest_df1 = calc_df(rest_tf1)
	# rest_idf1 = calc_idf(rest_df1,len(rest_body1))
	# rest_tfidf1 = calc_tfidf(rest_tf1,rest_idf1)

	# 	#ここから下svr_lightに渡す学習データを作る！！！
	# #言葉をタグづけする
	# for title1 in tfidf1:	#tfidf1の配列を渡してもらう
	# 	for title1_word in title1.keys():	#tfidfで取り出した文字列を繰り返す
	# 		# print title1_word, title1[title1_word]#tfidfの単語と状態表示
	# 		if title1_word not in words:#言葉にタグをつけるプログラム
	# 			words[title1_word] = num
	# 			num = num + 1
	# #タグづけされたでーたを元にして書き込む
	# for title1 in tfidf1:
	# 	fp.write('1 ')
	# 	for title1_word in title1.keys():
	# 		if title1_word not in words.keys():#もし、単語が入ってないとき
	# 			break
	# 		tag = words[title1_word]#タグを取り出す
	# 		attrs[tag] = title1[title1_word]
	# 		# if tag in attrs:#
	# 		# 	attrs[tag] = attrs[tag] + 1
	# 		# else:
	# 		# 	attrs[tag] = 1
	# 		# print title1_word,tag#出ているものの表示
	# 	for ak in sorted(attrs.keys()):
	# 		if attrs[ak] != 0:
	# 			# print str(ak) + ":" + str(attrs[ak])
	# 			fp.write(str(ak+1))
	# 			fp.write(':')
	# 			fp.write(str(attrs[ak]))
	# 			fp.write(' ')
	# 	for i in range(len(attrs)):#初期化
	# 		attrs[i] = 0
	# 	fp.write('\n')
	# 		# if title1_word not in str1:	#ユニークをする
	# 		# 	df[df2word] = 0	#単語を新しく入れる
	# 		# df[df2word] += 1 #dfの値を入れる

	tate = len(tfidf5)
	yoko = num
	print "tate",tate,"yoko",yoko
	dnn = [[0 for col in range(yoko)] for row in range(tate)]
	# print "last_number",dnn[3017][2268]

	
	for title5 in tfidf5:	#tfidf1の配列を渡してもらう
		for title5_word in title5.keys():	#tfidfで取り出した文字列を繰り返す
			# print title1_word, title1[title1_word]#tfidfの単語と状態表示
			if title5_word not in words:#言葉にタグをつけるプログラム
				words[title5_word] = num
				num = num + 1

	for title5 in tfidf5:
		for title5_word in title5.keys():
			if title5_word not in words.keys():
				break
			tag = words[title5_word]
			attrs[tag] = title5[title5_word]
		for ak in sorted(attrs.keys()):
			if attrs[ak] != 0:
				dnn[gyou][ak] = attrs[ak]
		for i in range(len(attrs)):
			attrs[i] = 0
		gyou = gyou + 1
	gyou5 = gyou
	# for i in range(gyou5):
	# 	print dnn[i]
#/////////////////////////////////////////////////////////////////////////////////////////////次元圧縮
	dim = 2#次元を変える
	pca=sklearn.decomposition.PCA(dim)
	result=pca.fit_transform(dnn)
	vectors = result
	# print "result",result,result.shape
	for i in range(len(result)):
		print i,result[i][0],result[i][1]
	#上がPCA下がLDA上もしくは下を消すことで動作します。
	# u,sigma,v=linalg.svd(dnn) # 特異値分解
	# rank = shape(dnn)[0] # 階数
	# print "rank",rank
	# u = matrix(u)
	# s = matrix(linalg.diagsvd(sigma, rank, rank))
	# v = matrix(v[:rank, :])
	# print u*s*v
	 
	# z = 5
	# u3 = u[:, :z]
	# s3 = matrix(linalg.diagsvd(sigma[:z], z, z))
	# v3 = v[:z, :]
	# result = u3*s3*v3
	# print result[0]

	# print "result",result.shape

	# print sum(sigma[:z])/sum(sigma)
#////////////////

	classnum = 3#1〜6まで変更可能だが大きくなるほどエラーが発生する
    # #分類対象のデータのリスト。各要素はfloatのリスト
    # vectors = [[random.random(), random.random()] for i in range(200)]#0<random.random()<1
    # print vectors
    #分類対象のデータをクラスタ数3でクラスタリング
	centers,label_vector = clustering(result, classnum)
    # print centers
    # print len(vectors)
    # print "label_vector",label_vector
	for i in range(len(vectors)):
		plt.scatter(vectors[i][0],vectors[i][1])
		print label_vector[i]
    # print len(label_vector)
	# print type(label_vector)#type
	for i in range(len(vectors)):
        # print label_vector[i],i,type(label_vector[i])
		if label_vector[i]==0:
			plt.scatter(vectors[i][0],vectors[i][1],c='b',marker="o")
		elif label_vector[i]==1:
			plt.scatter(vectors[i][0],vectors[i][1],c='g',marker="s")
 		elif label_vector[i]==2:
 			plt.scatter(vectors[i][0],vectors[i][1],c='r',marker="^")
 # 		elif label_vector[i]==3:
 # 			plt.scatter(vectors[i][0],vectors[i][1],c='c')
 # 		elif label_vector[i]==4:
 # 			plt.scatter(vectors[i][0],vectors[i][1],c='m')
 # 		elif label_vector[i]==5:
 # 			plt.scatter(vectors[i][0],vectors[i][1],c='w')
 # 		elif label_vector[i]==6:
 # 			plt.scatter(vectors[i][0],vectors[i][1],c='y')
	plt.scatter(centers[0][0],centers[0][1],c = 'k',marker="+")
	plt.scatter(centers[1][0],centers[1][1],c = 'k',marker="+")
	plt.scatter(centers[2][0],centers[2][1],c = 'k',marker="+")

	for i in range(classnum):
		# plt.plot(centers[i][0],centers[i][1],'+k')
		print "重心",i+1,centers[i]
		print "文書数",label_vector.count(i)
		print "cos類似度",cosine_similarity_rank(result,centers[i],5)
		# ffff.write(str("重心",i+1,centers[i],"\n文書数",label_vector.count(i),"\ncos類似度",cosine_similarity_rank(dnn,centers[i],5)))
		for j in cosine_similarity_rank(result,centers[i],5):
			print body5[j]
			print result[j]
	plt.show()
	#以下dnnの形変形をしようと試みたが必要なくdnn[][]に入っていた。
	# for j in range(gyou5):#DNN用のものを作る！！
	# 	# if j<gyou1:
	# 	# 	fp.write("1,")
	# 	# else:
	# 	# 	fp.write("0,")
	# for j in range(gyou5):#値をかきこむ 
	# 	# letter_tfidf.append(dnn[j][i])#いらないプログラム
	# 	for i in range(yoko):
	# 		# fp.write(str(int(10*dnn[j][i])))
	# 		fp.write(str(result[j][i]))
	# 		if i<yoko-1:
	# 			fp.write(",")
	# 	fp.write("\n")
	# # 	class_data.append(letter_tfidf)#いらないプログラム
	# # 	letter_tfidf=[]#いらないプログラム
	# fp.close()
	# print class_data


	dic_tfidf = {}
	z = 0
	# for title5 in tfidf5:	#tfidf1の配列を渡してもらう
	# 	for title5_word in title5.keys():	#tfidfで取り出した文字列を繰り返す
	# 		dic_tfidf[z] = title5_word
	# 		z += 1

	# for j, k in dic_tfidf.items():
	# 	print j, k

	# for title5 in tfidf5:
	# 	for k,v in title5.items():
	# 		print k,v
	# print len(dnn[0]),dnn[1]		



	f = open("write_test_pca1.txt", "w")

	for i in range(classnum):
		f.write("重心")
		f.write(str(i+1))
		f.write(str(centers[i]))
		f.write("\n")
		f.write("文書数")
		f.write(str(label_vector.count(i)))
		f.write("\n")
		f.write("cos類似度 \n")
		f.write(str(cosine_similarity_rank(result,centers[i],5)))
		f.write("\n")
		# ffff.write(str("重心",i+1,centers[i],"\n文書数",label_vector.count(i),"\ncos類似度",cosine_similarity_rank(dnn,centers[i],5)))
		for j in cosine_similarity_rank(result,centers[i],5):
			f.write(str(body5[j]))
			f.write(str(result[j]))
			f.write("\n")
		f.write("\n\n")