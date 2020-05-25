import re;
import nltk; #nltk must be installed
from nltk.stem import PorterStemmer
import collections
import math
from operator import itemgetter
from collections import Counter
class termInfo:     #This class is created for keeping necessary information
    def __init__(self, docID, term, wi,dti,qti):
        self.docID = docID
        self.term = term
        self.wi=wi
        self.dti=dti
        self.qti = qti

porter=PorterStemmer() #For stemming words

paragraphs=[]
f=open('200_content.txt',encoding="utf8").read()
f=f.split('\n')
for line in f:
    paragraphs.append(line)
N=len(paragraphs) # N is the number of documents in the collection

headlineForDictionary="Term, Document Frequency, Offset"  #Dictionary headline
headlineForPostings="Term, DocID, Raw Term Frequency, Weighted Term Frequency"
dictionaryF= open('dictionary.txt', 'w')
#dictionaryF.write(headlineForDictionary)
postings=open('postings.txt', 'w')
#postings.write(headlineForPostings)

inverted_index = collections.defaultdict(set)
i=1 #This is a counter for document ID. Once we move to another line, i will increase by one.
termList=[] #This list of tuples is for keeping all terms
nonDuplicateTerms=[]
for line in paragraphs:  #start reading the document
    document=line;
    document=document.lower(); #make all characters lowercase
    document = re.sub(r'[^\w\s]', '', document)  # For removing punctuations
    tokens = document.split()
    for words in tokens:
        words=porter.stem(words); #stem every word
        inverted_index[words].add(i) #add terms and their docIDs to inverted_index
        termList.append((words,i));  #add terms and docIDs in a list of tuples
        if words not in nonDuplicateTerms:
            nonDuplicateTerms.append(words) #adds the term to list only if term is not already in the list.
    i=i+1
termList=sorted(termList)  #sort the term list. It sorts for elements and then their docIDs
nonDuplicateTerms=sorted(nonDuplicateTerms)

counter=1 #this counter is for keeping line numbers in the postings list.
for words in nonDuplicateTerms:
    #len(inverted_index[words]) gives the doc frequency because it doesn't keep same documents if a term is appeared more than once in the same document
    element= words +" " +str(len(inverted_index[words])) +" " + str(counter) +"\n"
    counter=counter+len(inverted_index[words]) #Since every term will appear the number of its doc frequency,when we add its doc frequency to counter, we reach the next element.
    dictionaryF.write(element)

listForDuplicates = []  # if an element is appeared more than 1 time in the same document, we should make sure it only appears one time in postings list.
for tuples in termList:
    # Tuples first element is the term and second element is doc ID.
    # The count method counts how many times same tuple appeared in the list.
    # If two tuple's term and DocID is same, they are in the same document which increases term Frequency in this document.
    tfw=1+(math.log10(termList.count(tuples)))
    element = tuples[0] + " "+str(  tuples[1] )+" "  + str(termList.count(tuples)) +" "+ str(tfw)+"\n"
    if element not in listForDuplicates:
        postings.write(element)
        listForDuplicates.append(element);  # add the element to this list to prevent duplicate lines

postings.close()
dictionaryF.close()
print("dictionary.txt and postings.txt is constructed. Now, calculating doc len and normalized doc len")
DocLen=[]
NormalizedDocLen=[]
for i in range(0,(N+1)) :
    DocLen.append(0)
for i in range(0,(N+1)):
    NormalizedDocLen.append(0)
#calculate doc lens based on term frequencies
listForDuplicate = []
for tuples in termList:
    tf = termList.count(tuples)
    element=tuples[0]+str(tuples[1])
    if element not in listForDuplicate:
        DocLen[tuples[1]]=DocLen[tuples[1]]+tf
        listForDuplicate.append(element);

#calculate normolized doc lens
listForDuplicat = []
for tuples in termList:
    tfw = 1 + (math.log10(termList.count(tuples)))
    element=tuples[0]+str(tuples[1])
    if element not in listForDuplicat:
        NormalizedDocLen[tuples[1]]=NormalizedDocLen[tuples[1]]+(tfw*tfw)
        listForDuplicat.append(element);

for i in range (0,len(NormalizedDocLen)):
    NormalizedDocLen[i]=math.sqrt(NormalizedDocLen[i])

total=0
for i in DocLen:
    total=total+i
avdl=total/N #average document length

queries=input("Enter the queries \n")  #Take queries from the user.
queries=queries.split() #split the input based on spaces.
i=0
queryList=[] #This list keeps all words seperate which is given by user
queryArray=[] # This array keeps every different query as one element in the list.
for words in queries:
    # If a word starts wiht character '('
    # We understand that this index gives the start
    # of the query and at the next index, query starts
    if (words[0]=="("):
        queryList.append(i)
    i=i+1
counter=1
k=0
line="" #initiate lines as an empty string


#Following loops takes words between two words and its indexes were determined in the previous step.
# Take words between two start points, unites if query term includes more than one word
# and keep them in the list: queryArray
while(k<(len(queryList)-1)):
    num=queryList[k+1]-queryList[k]-1
    for t in range(num):
        line=line+" "+ queries[counter]+ " "
        counter=counter+1
    queryArray.append(line)
    line=""
    counter =counter+1
    k=k+1

#Previous while loop takes queries between two indexes, so it cannot include the last one.
#Following loop is for keeping the last query
line=""
num=i-queryList[k]-1
for t in range(num):
    line = line + " " + queries[counter] + " "
    counter = counter + 1
queryArray.append(line)


#Following implementation processes the stemming for query terms and keeps them in queryArr list.
queryArr=[]
x=""
for elements in queryArray:
    line=elements
    line=line.lower()
    tr=line.split()
    for words in tr:
        words = re.sub(r'[^\w\s]', '', words)
        words = porter.stem(words)
        x=x+" "+words
    queryArr.append(x)
    x=""



#Computing cos similarity and write to the file:
queryOrd=1 #for keeping order query to write to the file
for query in queryArr:
    name = "cosquery" + str(queryOrd) + "result.txt" #determine file name
    cosOut = open(name, "w")
    cosOut.write("For query: ")
    cosOut.write(query)
    cosOut.write("\n")
    queries = query.split() #split every query's terms
    queryLen=0
    queryTermWeights=[]
    for term in queries:  #calculate each term's term frequency in the query
        termTf=query.count(term)
        qtf=1+math.log10(termTf)
        idf=0
        dft=0
        f=open("dictionary.txt","r")
        for line in f:
            doc=line.split()
            if(term==doc[0]):
                dft=int(doc[1])  #In the dictionary, take document frequency of the term
        f.close()
        idf=math.log10(N/dft) #idf weight of the term
        queryTermWeights.append(qtf*idf)
        res=qtf*idf
        s="Term "+ term+ "  Query tf-idf weight="+ str(res)+"\n"
        cosOut.write(s)
    docResults=[]
    docResults.clear()
    for i in range (0,N+1):
        docResults.append(0)
    docId=0
    for term in queries:
        termNum=0
        offset=0
        df=0
        f = open("dictionary.txt", "r")
        for line in f:
            doc = line.split()
            if (term == doc[0]):
                offset = int(doc[2])-1 #take the offset and doc freq of the term
                df=int(doc[1])
        f.close()
        f = open("postings.txt", "r")
        lineCounter = 0
        termFreq = 0
        for line in f:
            doc = line.split()
            if (lineCounter>(offset-1) and lineCounter<(offset+df)):
                termFreq = float(doc[3]) #take term freq of the term in the specified document. termFreq is the last item in the postings list
                id=int(doc[1])
                idf = math.log10(N /df)
                tf_idf=termFreq*idf #calculate tf-idf weight of term in the document.
                result=tf_idf*queryTermWeights[termNum] #multiply term's query and document tf-idf weights
                docResults[id]=docResults[id]+result  #add each query term's tf-idf weight multiplications to this result list. docId=index
            lineCounter = lineCounter + 1

        termNum+=1
    counter=0
    lastResults=[]
    for i in docResults:
        if(i!=0):
            i=i/NormalizedDocLen[counter] #divide the results to normalized doc lengths
            lastResults.append([i,counter])#add docID and similarities to a new list
        counter=counter+1
    lastResults.sort(reverse=True)

    c=0
    for t in lastResults:
        if(c<10):
            strng="Similarity: "+str(t[0])+" DocId: " + str(t[1])+ "\n"
            cosOut.write(strng)  #write docId and similarity information to the file.
        c=c+1
    queryOrd=queryOrd+1


#Compute Okapi similarity and write to the file:

queryOrd = 1 #for keeping query order
for query in queryArr:
    name = "okaquery" + str(queryOrd) + "result.txt" #create file name
    okaOut = open(name, "w")
    okaOut.write("For query: ")
    okaOut.write(query)
    okaOut.write("\n")
    queries = query.split()
    okapresult = []  #every index in this list is docID
    for i in range(1, N+1):
        okapresult.append(0)  #initialize with 0
    for id in range(1, N):
        for term in queries:
            qtfi = 0
            dti=0
            for t in queries:  #calculate term's tf in the query
                if (term == t):
                    qtfi = qtfi + 1
            qti = (1.2 + 1) * qtfi / (1.2 + qtfi)
            f = open("dictionary.txt", "r")
            offset=0
            df=0
            for line in f:
                doc = line.split()
                if (term == doc[0]):  #extract df and offset from the dictionary
                    offset = int(doc[2]) - 1
                    df = int(doc[1])
            f.close()
            wi=math.log10((N-df+0.5)/(df+0.5))
            tfi=0
            f = open("postings.txt", "r")
            lineCounter = 0
            termFreq = 0
            for line in f:
                doc = line.split()
                if (lineCounter > (offset - 1) and lineCounter<(offset+df)):
                    if(int(doc[1])==id ):
                        tfi=int(doc[2])  #extract tf of the term in the document.
                        dti=((1.2+1*tfi)/(1.2*((1-0.75)+0.75*DocLen[id]/N)+tfi))
                        okap = qti * wi * dti #calculate okapi
                        okapresult[id]=okapresult[id]+okap #add query's every term calculation to the result array. index i means document id=i
                lineCounter=lineCounter+1
        f.close()
    count=0
    finalresults=[]
    for i in okapresult:
        if(i!=0):
            finalresults.append([okapresult[count],count]) #add results to a new list as [similarty, docID] tuple
        count = count + 1
    finalresults.sort(reverse=True)
    #write information to the file:
    c = 0
    for t in finalresults:
        if (c < 10):
            strng = "Similarity: " + str(t[0]) + " DocId: " + str(t[1]) + "\n"
            okaOut.write(strng)  #write information to the file
        c = c + 1
    okapresult.clear()
    queryOrd=queryOrd+1








































