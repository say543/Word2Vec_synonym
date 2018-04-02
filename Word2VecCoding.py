
# # this is for type chinese in code
## -*- coding: utf-8 -*- 

from numpy import array
from collections import Counter, OrderedDict

import json
import sys
import numpy as np
import random
import math

class Word2VecCoding:

    #hidden layer size
    # output representation for each word embedding
    LAYER1_SIZE = 30
    # context length
    # ?  not sure how it work
    WINDOW = 2
    # learing rate
    ALPHA_INIT = 0.025
    # random depreate high-frequency words
    SAMPLE = 0.001
    # number of negative sampling
    NEGATIVE = 10
    # number of iteration 
    ITE = 2

    '''
    def __init__(self):
        #hidden layer size
        # output representation for each word embedding
        self.LAYER1_SIZE = 30
        # context length
        # ?  not sure how it work
        self.WINDOW = 2
        # learing rate
        self.ALPHA_INIT = 0.025
        # random depreate high-frequency words
        self.SAMPLE = 0.001
        # number of negative sampling
        self.NEGATIVE = 10
        # number of iteration 
        self.ITE = 2
    ''' 

    def LearnVocabFromTrainFile(self):

        
        # https://docs.python.org/3/howto/unicode.html
        #trainingFile = open("..\\TrainingFiles\\poem_small.txt", encoding='utf-8')
        trainingFile = open("..\\TrainingFiles\\poem.txt", encoding='utf-8')
        #trainingFile = open("..\\TrainingFiles\\poem.txt")

        # using numpy to calculate word cnt 
        volcabularyToCnt = Counter()
        '''
        # strip remove heading / trailing extra characters
        # here utf-8 is for chinese
        # https://pymotw.com/2/collections/counter.html
        # http://www.runoob.com/python/att-string-strip.html
        for line in trainingFile.readlines():
            print (line.encode("utf-8").decode('UTF-8'))
            for word in line.decode("utf-8").strip().split():
            #for word in line.strip().split():
                print (word)
                volcabularyToCnt.update(word)
        '''

        for line in trainingFile:
            #print (line.encode('UTF-8').decode('UTF-8'))
            # not sure why needs this. this is observed through output
            line = line.strip().replace('\ufeff', '')
            for word in line.encode('UTF-8').decode('UTF-8').split():
                #print (word.encode('UTF-8').decode('UTF-8'))
                volcabularyToCnt.update(word)

        '''
        # this supposes printing chinese but office computer keeps fail
        #print (trainingFile.readline().encode('UTF-8').decode('UTF-8'))

        #print (volcabularyToCnt)

        # for extra it fail
        # but for output debug window it successes
        text = '測試'
        print (text)
        '''


        # vcount_list  => volcabularyCntPairListAfterFilter
        # reverse means descending order
        # using sorting order to give a word a id 
        # depending on criteria, filtering  words under  ? cnt
        # here is 5
        # for 1 error rate is 0.17 error

        # c.items()                       # convert to a list of (elem, cnt) pairs

        volcabularyCntPairListAfterFilter = sorted(filter(lambda x: x[1] >= 5, volcabularyToCnt.items()),
                                            reverse = True, key = lambda x: x[1])


        print (volcabularyCntPairListAfterFilter)

        # vocab_dict = volcabularyToIdByFreqDesc
        # build dictionary, one id for one word
        # using orderedDict to get 
        #   enumerate will generate
        # ((0, ("a", 2)), (1, ("b", 1))......)
        volcabularyToIdByFreqDesc = OrderedDict(map(lambda x: (x[1][0], x[0]), enumerate(volcabularyCntPairListAfterFilter)))

        print (volcabularyToIdByFreqDesc)



        # transform volcabularyCntPairListAfterFilter into ordered dictionary for lookup
        # vocab_freq_dict => volcabularyToCntByFreqDesc 
        # ? why here need to order, might be aligned with volcabularyToIdByFreqDesc with id
        volcabularyToCntByFreqDesc = OrderedDict(map(lambda x: (x[0], x[1]),   volcabularyCntPairListAfterFilter))

        print (volcabularyToCntByFreqDesc)

        trainingFile.close()
        print ('inside function')

        # output volcabulary and id for reference
        volcabularyToIdFile = None
        try:
            volcabularyToIdFile = open("..\\TrainingFiles\\volcabularyToId.txt", 'w',  encoding='utf-8')
            for volcabulary, id in volcabularyToIdByFreqDesc.items():
                volcabularyToIdFile.write(volcabulary +"\t" + str(id) + "\n")
        except:
            print ("unknown exception, something wrong while output volcabulary and id as pair")
        finally:
            volcabularyToIdFile.close()

        return volcabularyToIdByFreqDesc, volcabularyToCntByFreqDesc

    # tricky part for hanlding negative sampling
    # ? 0.75 is a magic number
    # ? duplicate word will be different entries here, not sure the purpose
    def InitUnigramTable(self, volcabularyToCntByFreqDesc):

        # get id with normalized freq pairs

        # table_freq_list => volcabularyIdToCntWithAdjustment
        ##volcabularyIdToCntWithAdjustment = map(lambda x: (x[0], int(x[1][1] * 0.75)), enumerate(volcabularyToCntByFreqDesc.items()))
        # i prefer to change it to dicionary
        # here ut us ** not *
        # python ** expotentila
        volcabularyIdToCntWithAdjustment = OrderedDict(map(lambda x: (x[0], int(x[1][1] ** 0.75)), enumerate(volcabularyToCntByFreqDesc.items())))


        print (volcabularyIdToCntWithAdjustment.items())
        
        #table_size => totalCntAFterAdjustment
        totalCntAFterAdjustment = 0
        # using map object
        #for volcabularyIdAndCnt in volcabularyIdToCntWithAdjustment:
        #    totalCntAFterAdjustment += volcabularyIdAndCnt[1]
        # using dictionary 
        for id, cnt in volcabularyIdToCntWithAdjustment.items() :
            totalCntAFterAdjustment += cnt
        
        # table => unigramTable
        #unigramTable = np.zeros(totalCntAFterAdjustment).astype(int)
        unigramTable = np.zeros(totalCntAFterAdjustment, dtype=int)

        ##unigramTable = np.zeros((3, totalCntAFterAdjustment), dtype=int)

        offset = 0

        # using map
        #for volcabularyIdAndCnt in volcabularyIdToCntWithAdjustment:
        #    unigramTable[offset:offset+ volcabularyIdAndCnt[1]] = volcabularyIdAndCnt[0]
        #    offset += volcabularyIdAndCnt[1]

        # using dictionary
        for id, cnt in volcabularyIdToCntWithAdjustment.items() :
            unigramTable[offset:offset+ cnt] = id
            offset += cnt





        print (unigramTable)

        return unigramTable


    def train(self, volcabularyToIdByFreqDesc, volcabularyToCntByFreqDesc, unigramTable):


        # ? where to use this 
        # randomly abandon some words
        totalWordsCnt = 0
        for volcabulary, cnt in volcabularyToCntByFreqDesc.items() :
            totalWordsCnt += cnt;

        print("totalWordsCnt = %10d" % (totalWordsCnt))

        # to decide dimentions
        vocabSize = len(volcabularyToCntByFreqDesc)

        print("vocabSize = %10d" % (vocabSize))


        # initialize weight 
        # ? why this kind of randomization
        # syn0 => inputToHiddenLayerWeight
        # each unique has one dimention
        #inputToHiddenLayerWeight = np.random.rand(vocabSize, LAYER1_SIZE)
        inputToHiddenLayerWeight = (0.5 - np.random.rand(vocabSize, self.LAYER1_SIZE)) / self.LAYER1_SIZE



        # ? why this is zero
        # syn1 => HiddenLayerToOutputWeight
        # ? how to initialize
        HiddenLayerToOutputWeight = np.zeros((self.LAYER1_SIZE, vocabSize))


        # train_words => trainWordsCnt
        # becasue words with high frequency might subject to be forfeited
        # here is to record words will survive
        trainWordsCnt = 0

        # for output and tracking  purpose
        pCount = 0
        avgErr = 0.
        errCount = 0

        for local_iter in range(self.ITE):
            print ("local_iter: %d" % (local_iter))

            # repopen the same file again
            #? might be in the future just open one time
            #trainingFile = open("..\\TrainingFiles\\poem_small.txt", encoding='utf-8')
            trainingFile = open("..\\TrainingFiles\\poem.txt", encoding='utf-8')
            lineNumber = 0
            try:
                for line in trainingFile:
                    ##line = line.strip().replace('\ufeff', '')
                    # sen => selectedWordListFromSentence
                    selectedWordIdListFromSentence = []

                    #print("lineNumber = %10d" % (lineNumber))
                    lineNumber+=1

                    for word in line.encode('UTF-8').decode('UTF-8').split():

                        # one time trains one sentence

                        # some words has filted by volcabularyCntPairListAfterFilter based on filtering count
                        # if cnt too small and not exist in volcabularyToIdByFreqDesc then do not train
                        if word in volcabularyToIdByFreqDesc:
                            # last_word => wordId
                            wordId = volcabularyToIdByFreqDesc[word]
                        
                            # cn => cntForWordId
                            cntForWordId = volcabularyToCntByFreqDesc[word]

                            # generate a random number to decide

                            # calculate probably with each word
                            # ? do not know how to come up with
                            # words with high frequency are more likely to be abandaned
                            ran = (math.sqrt(cntForWordId / float(self.SAMPLE * totalWordsCnt + 1))) * (self.SAMPLE * totalWordsCnt) / cntForWordId

                            if ran < random.random():
                                continue

                            trainWordsCnt+=1
                            selectedWordIdListFromSentence.append(wordId)

                    # adjust learning rate every sentence
                    # ? not sure how it work
                    alpha = self.ALPHA_INIT * (1  -  trainWordsCnt/ float(self.ITE * totalWordsCnt +1) )

                    # ? magic number
                    # not making  learning rate to small
                    if alpha < self.ALPHA_INIT * 0.0001:
                        alpha = self.ALPHA_INIT * 0.0001



                    # for a sentence , start learning for a selected words
                    # a => wordIndexInSentence
                    # word => curWordId
                    for wordIndexInSentence, curWordId in enumerate(selectedWordIdListFromSentence):
                    #for wordIndexInSentence in range(len(selectedWordIdListFromSentence)):

                        # adjust window size
                        # b => adjustWindowSize
                        adjustWindowSize = random.randint(1, self.WINDOW)

                        # c => windowIndex
                        # ? what about windowIndex is negative here
                        # it will be dealt with inside
                        #for windowIndex in range (wordIndexInSentence -adjustWindowSize,  wordIndexInSentence - adjustWindowSize +1):
                        for windowIndex in range (wordIndexInSentence -adjustWindowSize,  wordIndexInSentence + adjustWindowSize +1):

                            # make sure validate word index in range
                            # also  windowIndex not the same as current checking wordIndexInSentence
                            if windowIndex >=0 and windowIndex <len(selectedWordIdListFromSentence) and windowIndex != wordIndexInSentence:


                                # last_word => ? lastWordId
                                # ? might be renaming it to a better name
                                # what does last word mean
                                # even list you can still using array-like to access
                                lastwordId = selectedWordIdListFromSentence[windowIndex]



                                # h_err => hiddenLayerUpdateBuffer
                                # ? dimentioan not sure
                                # it is decided by backword propagation
                                hiddenLayerUpdateBuffer = np.zeros((self.LAYER1_SIZE))


                                # negative sampling
                                # negcount => negIndex
                                for negIndex in range(self.NEGATIVE):

                                    # if zero, do positvie sampleing checking
                                    if (negIndex == 0):
                                        # target_word => targetWordId
                                        targetWordId = curWordId
                                        label = 1
                                    else:
                                        # get negative sample randomly from unigramTable
                                        # unigramTable has unique ids but different ids might be the same words
                                        # ? might be having duplication because of reasons

                                        # ? my speculation
                                        # also, assuming randomization so no word will be picked up more than one time for a NEGATIVE cycle


                                        label = 0
                                        while True:
                                            # range is included
                                            targetWordId = unigramTable[random.randint(0, len(unigramTable)-1)]

                                            # negative sample is not one following immedidately
                                            # instead, words not presenting in the sentence are negative sample
                                            # different from what i saw
                                            if targetWordId not in selectedWordIdListFromSentence:
                                                break;



                                    # get model predict result
                                    # how to come up word representation for one-hot encoding
                                    # using last_word => ? wordId
                                    # it also decdiew which V (from inputToHiddenLayerWeight) and which W  (from HiddenLayerToOutputWeight)
                                    # idk , Vk1....Vk layer and w1 .....Wk layer

                                    # sigmoid
                                    # Y for word id : lastwordId to word id :   targetWordId
                                    # o_pred => predForPositiion

                                    #print (inputToHiddenLayerWeight[lastwordId, :])
                                    #print (HiddenLayerToOutputWeight[:, targetWordId])
                                    #print (-np.dot(inputToHiddenLayerWeight[lastwordId, :], HiddenLayerToOutputWeight[:, targetWordId]))
                                    #print (np.exp(-np.dot(inputToHiddenLayerWeight[lastwordId, :], HiddenLayerToOutputWeight[:, targetWordId])))

                                    predForPositiion = 1 / (1 + np.exp(-np.dot(inputToHiddenLayerWeight[lastwordId, :], HiddenLayerToOutputWeight[:, targetWordId])))

                                
                                    # for positve example, label = 1
                                    # for negative example, label = 0

                                    # o_err 
                                    # skip this variable
                                
                                    # using hiddenLayerUpdate, old W HiddenLayerToOutputWeight to accumluate possible update for V
                                    # this will be update to V after positive example and negative sample per sentence are done
                                    # scalar multiply vector
                                    hiddenLayerUpdateBuffer += (predForPositiion - label) * HiddenLayerToOutputWeight[:, targetWordId]

                                    # becasue of randomization , supposes targetwordId will not duplicate in the same NEGATIVE cycle
                                    # so, no need to use another space to store HiddenLayerToOutputWeight[targetWordId] 's new update
                                    # instead, just use originla space to update

                                    # for debug
                                    # the following are doing the same operation 
                                    #print (inputToHiddenLayerWeight[lastwordId, :])
                                    #print (inputToHiddenLayerWeight[lastwordId])

                                    #print (HiddenLayerToOutputWeight[:, targetWordId])
                                    #print (inputToHiddenLayerWeight[lastwordId, :])


                                    #print (HiddenLayerToOutputWeight[:, targetWordId] + inputToHiddenLayerWeight[lastwordId, :])

                                    #HiddenLayerToOutputWeight[:, targetWordId] -= alpha * (predForPositiion - label) * inputToHiddenLayerWeight[lastwordId]
                                    HiddenLayerToOutputWeight[:, targetWordId] -= alpha * (predForPositiion - label) * inputToHiddenLayerWeight[lastwordId, :]

                                    #print (HiddenLayerToOutputWeight[:, targetWordId])

                                    # avg_err => avgErr
                                    # ? not sure the purpose of it
                                    # only record absolute difference
                                    avgErr += abs(predForPositiion - label)


                                    # err_count => errCount
                                    # ? not sure the purpose of it since it add one anywy
                                    errCount += 1
                            
                                # update hidden layer by hiddenLayerUpdateBuffer
                                inputToHiddenLayerWeight[lastwordId, :] -= alpha * hiddenLayerUpdateBuffer


                                # print out result
                                # for output information
                                # ? it will reset after 10000 do not know why
                                # prevent too much output
                                # p_count => pCount
                                pCount +=1
                                
                                if pCount % 10000 == 0:
                                    #print("iter = %3d, alpha : %.6f, train words = %d, Average Error = %.6f" % \
                                    #    (local_iter, alpha, 100 *trainWordsCnt, avgErr / float(errCount)))
                                    print("iter = %3d, alpha : %.6f, train words = %d, Average Error = %.6f  lineNumber = %10d" % \
                                        (local_iter, alpha, 100 *trainWordsCnt, avgErr / float(errCount), lineNumber))

                                    # ? not sure why need to reset here
                                    # i think for a 10000 update, need to monitor status
                                    avgErr = 0.0
                                    # ? internet code only reset avgErr but not errCount
                                    # thus, errCount keeps decreasing  but it makes no sense
                                    # in order to align with internet solution, do not reset here
                                    #errCount = 0

                                '''
                                print("iter = %3d, alpha : %.6f, train words = %d, Average Error = %.6f" % \
                                    (local_iter, alpha, 100 *trainWordsCnt, avgErr / float(errCount)))
                                if pCount % 10000 == 0:
                                    avgErr = 0
                                    errCount = 0
                                '''

            except:
                print ("unknown exception, something wrong with routine")
            finally:
                trainingFile.close()

            try:
                # store models per iteration
                # model_name => modelName
                modelName = "w2v_model_blog_%s.json" % (local_iter)
                print ("save model: %s" % (modelName))

                # no chinese, no need utf 8
                # fm => modelFile
                modelFile = open("..\\TrainingFiles\\" +modelName, 'w')
                modelFile.write(json.dumps(inputToHiddenLayerWeight.tolist(), indent=4))
            finally:
                modelFile.close()

#####################################
# internet
#####################################
    #def trainInter(vocab_dict, vocab_freq_dict, table):
    def trainInter(self, volcabularyToIdByFreqDesc, volcabularyToCntByFreqDesc, table):
        
        totalWordsCnt = sum([x[1] for x in volcabularyToCntByFreqDesc.items()])
        vocabSize = len(volcabularyToIdByFreqDesc)

        # 參數設定

        '''
        layer1_size = 30 # hidden layer 的大小，即向量大小

        window = 2 # 上下文寬度的上限

        alpha_init = 0.025 # learning rate

        sample = 0.001 # 用來隨機丟棄高頻字用

        negative = 10 # negative sampling 的數量

        ite = 2 # iteration 次數
        '''

    
        # Weights 初始化

        # inputToHiddenLayerWeight : input layer 到 hidden layer 之間的 weights ，用隨機值初始化

        # HiddenLayerToOutputWeight : hidden layer 到 output layer 之間的 weights ，用0初始化

        inputToHiddenLayerWeight = (0.5 - np.random.rand(vocabSize, self.LAYER1_SIZE)) / self.LAYER1_SIZE 
        HiddenLayerToOutputWeight = np.zeros((self.LAYER1_SIZE, vocabSize))
    
        # 印出進度用

        trainWordsCnt = 0 # 總共訓練了幾個字

        pCount = 0
        avgErr = 0.
        errCount = 0
    
        for local_iter in range(self.ITE):
            print ("local_iter: %d" % (local_iter))
            f = open("..\\TrainingFiles\\poem.txt", encoding='utf-8')
            for line in f.readlines():
            
                #用來暫存要訓練的字，一次訓練一個句子

                selectedWordListFromSentence = []
            
                # 取出要被訓練的字

                #for word_raw in line.decode("utf-8").strip().split():
                for word_raw in line.encode('UTF-8').decode('UTF-8').split():
                    wordId = volcabularyToIdByFreqDesc.get(word_raw, -1)
                
                    # 丟棄字典中沒有的字（頻率太低）

                    if wordId == -1:
                        continue
                    cntForWordId = volcabularyToCntByFreqDesc.get(word_raw)
                    ran = (math.sqrt(cntForWordId / float(self.SAMPLE * totalWordsCnt + 1))) * (self.SAMPLE * totalWordsCnt) / cntForWordId
                
                    # 根據字的頻率，隨機丟棄，頻率越高的字，越有機會被丟棄

                    if ran < random.random():
                        continue
                    trainWordsCnt += 1
                
                    # 將要被訓練的字加到 selectedWordListFromSentence

                    selectedWordListFromSentence.append(wordId)
                
                # 根據訓練過的字數，調整 learning rate

                alpha = self.ALPHA_INIT * (1 - trainWordsCnt / float(self.ITE * totalWordsCnt + 1))
                if alpha < self.ALPHA_INIT * 0.0001:
                    alpha = self.ALPHA_INIT * 0.0001
                
                # 逐一訓練 selectedWordListFromSentence 中的字

                for wordIndexInSentence, word in enumerate(selectedWordListFromSentence):
            
                        # 隨機調整 window 大小

                    b = random.randint(1, self.WINDOW)
                    for c in range(wordIndexInSentence - b, wordIndexInSentence + b + 1):
                    
                        # input 為 window 範圍中，上下文的某一字

                        if c < 0 or c == wordIndexInSentence or c >= len(selectedWordListFromSentence):
                            continue
                        wordId = selectedWordListFromSentence[c]
                                        
                        # h_err 暫存 hidden layer 的 error 用

                        h_err = np.zeros((self.LAYER1_SIZE))
                    
                        # 進行 negative sampling

                        for negcount in range(self.NEGATIVE):
                    
                            # positive example，從 selectedWordListFromSentence 中取得，模型要輸出 1

                            if negcount == 0:
                                target_word = word
                                label = 1
                        
                            # negative example，從 table 中抽樣，模型要輸出 0 

                            else:
                                while True:
                                    target_word = table[random.randint(0, len(table) - 1)]
                                    if target_word not in selectedWordListFromSentence:
                                        break
                                label = 0
                        
                            # 模型預測結果

                            o_pred = 1 / (1 + np.exp(- np.dot(inputToHiddenLayerWeight[wordId, :], HiddenLayerToOutputWeight[:, target_word])))
                        
                            # 預測結果和標準答案的差距

                            o_err = o_pred - label
                        
                            # backward propagation

                            # 此部分請參照 word2vec part2 的公式推導結果

                        
                            # 1.將 error 傳遞到 hidden layer                        

                            h_err += o_err * HiddenLayerToOutputWeight[:, target_word]
                        
                            # 2.更新 HiddenLayerToOutputWeight

                            HiddenLayerToOutputWeight[:, target_word] -= alpha * o_err * inputToHiddenLayerWeight[wordId]
                            avgErr += abs(o_err)
                            errCount += 1
                    
                        # 3.更新 inputToHiddenLayerWeight

                        inputToHiddenLayerWeight[wordId, :] -= alpha * h_err
                    
                        # 印出目前結果

                        pCount += 1
                        if pCount % 10000 == 0:
                            #print "Iter: %s, Alpha %s, Train Words %s, Average Error: %s" \
                            #      % (local_iter, alpha, 100 * trainWordsCnt, avgErr / float(errCount))
                            print("iter = %3d, alpha : %.6f, train words = %d, Average Error = %.6f" % \
                                (local_iter, alpha, 100 *trainWordsCnt, avgErr / float(errCount)))

                            # ? here reset average err for new calculation
                            # but errCount not reset
                            # does not make sense
                            avgErr = 0.
                            # this line is useless so comment it 
                            # errCount == 0.
                            print ("errCount: %.10f" % (errCount))
                        
            # 每一個 iteration 儲存一次訓練完的模型

            model_name = "w2v_model_blog_%s.json" % (local_iter)
            print ("save model: %s" % (model_name))
            fm = open("..\\TrainingFiles\\" +model_name, 'w')
            fm.write(json.dumps(inputToHiddenLayerWeight.tolist(), indent=4))
            fm.close()
#####################################
# internet
#####################################

    def getTop(self, word, volcabularyToIdByFreqDesc, w2vModel, idToVolcabularyByFreqDesc):

        # add prevent
        if word not in volcabularyToIdByFreqDesc:
            print ("query word: %s does not exist in training files" % (word))

        # wid = wordId
        wordId = volcabularyToIdByFreqDesc[word]

        # dimention becomoe layers X 1 after expand
        #print (np.expand_dims(w2vModel[wordId], axis=1))

        # calculate cosine similarity
        # for np dot 
        # [vacab size X 30 ] [30 X 1 after expand dims] = [vocab X 1]
        # dot_result = dotResult
        dotResult = np.dot(w2vModel, np.expand_dims(w2vModel[wordId], axis=1))

        #print (dotResult)
        
        # ? why transpose, 
        # becasue using axis = 0 to do sum
        # [vocab  x layer size ]
        # [layer size * vocab size ]
        #print (np.power(w2v_model.T, 2))
        
        # [1 X vocab size ]
        norm = np.sqrt(np.sum(np.power(w2vModel.T, 2), axis=0))

        #print (norm)

        # cosine similarity
        # norm [1 X vocab size ]
        # norm[wordId] : scalar
        # dot_result[:, 0] : [vocab X 1]

        #print (norm*norm[wordId])
        #print (dotResult[:, 0])

        # ? how can this get calculated
        # because 1-d array will be row array default even though it is column
        # [1 X vocab ] 
        #cosine_result => cosineResult
        cosineResult = np.divide(dotResult[:, 0], norm*norm[wordId])

        #print (cosineResult)

        # sort by cosineResult in decreasing order
        # also filter the same wordId
        # id , cosineResult for a vocabulary

        # final_result => finalResult
        finalResult = sorted(filter(lambda x:x[0] != wordId, 
                          [(x[0], x[1]) for x in enumerate(cosineResult)]),
                          key =lambda x: x[1], reverse=True)

        print ("query word: %s" % (word))

        # print top 5 closest
        # index 0 to 4
        for id, cosineValue in finalResult[:5]:
            print ("closest word: %s [%.6f]" % (idToVolcabularyByFreqDesc[id], cosineValue))



if __name__ == '__main__':


    print ('test')

    ###############################
    # training and store model
    ###############################
    
    volcabularyToIdByFreqDesc, volcabularyToCntByFreqDesc = Word2VecCoding().LearnVocabFromTrainFile()

    unigramTable = Word2VecCoding().InitUnigramTable(volcabularyToCntByFreqDesc)

    Word2VecCoding().train(volcabularyToIdByFreqDesc, volcabularyToCntByFreqDesc, unigramTable)
    

    ###############################
    # training and store model internet
    ###############################
    '''  
    volcabularyToIdByFreqDesc, volcabularyToCntByFreqDesc = Word2VecCoding().LearnVocabFromTrainFile()

    unigramTable = Word2VecCoding().InitUnigramTable(volcabularyToCntByFreqDesc)

    # globla internet function
    #trainInter(volcabularyToIdByFreqDesc, volcabularyToCntByFreqDesc, unigramTable)
    # change to class function
    Word2VecCoding().trainInter(volcabularyToIdByFreqDesc, volcabularyToCntByFreqDesc, unigramTable)
    '''

    ###############################
    # get model and get top closest synonyms
    ###############################

    '''
    # f2 -> trainedModelFile
    trainedModelName = "w2v_model_blog_%s.json" % (Word2VecCoding().ITE-1)

    trainedModelFile = None
    w2vModel = None
    try:
        # will read brackets and contents at the saime time
        #trainedModelFile = open("..\\TrainingFiles\\" + trainedModelName, "r")
        # for internet model
        trainedModelFile = open("..\\TrainingFiles\\wiki_model_source_code_103103302018\\" + trainedModelName, "r")

        #w2v_model => w2vModel
        w2vModel = np.array(json.loads("".join(trainedModelFile.readlines())))

        #i = 0
        #for line in trainedModelFile:
        #    i+=1
        #    print (line)
        #    if i == 10:
        #        break
    finally:
        trainedModelFile.close()


    # becasue 
    volcabularyToIdByFreqDesc, volcabularyToCntByFreqDesc = Word2VecCoding().LearnVocabFromTrainFile()

    # vocab_dict_reversed  => idToVolcabularyByFreqDesc
    # create inverted index and id is also unique as well
    # ? is it sorted by id?  no it just use the order from input array
    idToVolcabularyByFreqDesc = OrderedDict(map(lambda x: (x[1], x[0]), volcabularyToIdByFreqDesc.items()))

    #print (volcabularyToIdByFreqDesc)
    #print (idToVolcabularyByFreqDesc)


    Word2VecCoding().getTop(u"山", volcabularyToIdByFreqDesc, w2vModel, idToVolcabularyByFreqDesc)
    '''








    


