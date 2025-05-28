import os
import requests
import time
def SystemUI():
    print(' ----by 甲虫壳 qq:1729239418')
    print('|--------------------------------------|')
    print('|            文献检索程序           |')
    print('|输入关键词生成一个带有文献pdf下载链接的网页|')
    print('|      打开网页可根据需要直接下载文献     |')
    print('|--------------------------------------|')
def big_ContentGet(keyword,pageMark,yearSelect):
    weblist = WebsitePage(keyword, pageMark, yearSelect)  # 处理getdata里页数的内容，返回所有页数的网址
    for i in range(len(weblist)):
        print('>正在获取>%s  页面内容……<第%d页-总%d页' % (weblist[i], i + 1, pageMark))
        if JournalHome(filePath, weblist[i]) == 'Yes':  # 获取每个页面网址的内容
            file = open(filePath + 'content.txt', 'r', encoding='utf-8')
            data = file.read()
            file.close()
            if len(data.split('\n')) > 1:  # 有内容才对他进行数据处理,没内容就是空白文件
                articleTitle, articleDownload, articleOrigin, articlePuber, articleTime, articleCite, articleSummery = ContentGet(
                    filePath)
                DataSave(keyword, filePath, articleTitle, articleDownload, articleOrigin, articlePuber, articleTime,
                         articleCite, articleSummery)  # 输入参数是具体文章的网址，得到标题，文章编号，发表时间，文章类型，摘要，关键词等信息，并将它们储存在文件中
    print('|^_^|已获得%d个文献数据,内容保存在%sAlldata.txt中' % (pageMark * 10, filePath))
def WebsitePage(keyword,pageMark,yearSelect):#getdata是爬取的内容
    weblist = []        #储存网站不同页数网址的列表
    firstStr='https://sc.panda321.com/scholar?start='
    threeStr='&q='
    fourStr='&btnG='
    #https://sc.panda321.com/scholar?start=10&q=Supercapacitor&hl=zh-CN&as_sdt=0,5&as_ylo=2002&as_yhi=2018
    if yearSelect=='pass':
        yearStr=''
    elif '-' in yearSelect:
        yearStr='&as_ylo='+yearSelect.split('-')[0]+'&as_yhi='+yearSelect.split('-')[1]
    else:
        yearStr='&as_ylo='+yearSelect
    for i in range(pageMark):
        weblist.append(firstStr+str(i*10)+threeStr+keyword+fourStr+yearStr)
    return (weblist)
def JournalHome(filePath,website):   #获取杂志文章主页的内容，用于后面页数的分析
    header1={
    'authority':'sc.panda321.com',
    'method':'GET',
    'path':website[23:],
    'scheme':'https',
    'accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-encoding':'gzip,deflate,br',
    'accept-language':'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'cache-control':'max-age=0',
    'cookie':'_ga=GA1.2.1653995190.1611765489;GSP=LM=1611930908:S=nyJZa_ZUNMlMya_n;UM_distinctid=1774e91d798959-\
0bd43e28cf10ca-50391c45-1fa400-1774e91d799c34;__gads=ID=930b916a1451f7a3-226c487bd4c500f6:T=1611930918:RT=\
1611930918:S=ALNI_MZ7gxXD4j6U_w34cF3TmNecYpylfg;NID=208=O6hi6KjCGYh5OMBq18gWuTLyP-xI4hVgNh7Jdwf2lR3bb0Ie3d7Jn8ulm3S_\
OnMB4e8JEur23Ukys_VBd9dEPcJp8ujPy4DP01FB9FRLpLx8ZPb5gQ_S2ps5_VwnYDY8gmSZuvY5MkCslTnCOjuv5F0098jXEcYBjQTuo_I0SUY;_gid=GA1.2.\
2035594247.1612497151;CNZZDATA1279080935=577313372-1611927622-https%253A%252F%252Fsc.panda321.com%252F%7C1612493887\
;_gat_gtag_UA_126288799_1=1',
    'sec-fetch-dest':'document',
    'sec-fetch-mode':'navigate',
    'sec-fetch-site':'none',
    'upgrade-insecure-requests':'1',
    'user-agent':'Mozilla/5.0(WindowsNT10.0;Win64;x64)AppleWebKit/537.36(KHTML,likeGecko)Chrome/88.0.4324.96Safari/537.36Edg/88.0.705.56'
    }
    header2 = {
        'authority': 'sc.panda321.com',
        'method': 'GET',
        'path': website[23:],
        'scheme': 'https',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'accept-encoding': 'gzip,deflate,br',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'cache-control':'max-age=0',
        'cookie': '_ga=GA1.2.1653995190.1611765489;GSP=LM=1611930908:S=nyJZa_ZUNMlMya_n;UM_distinctid=1774e91d798959-\
    0bd43e28cf10ca-50391c45-1fa400-1774e91d799c34;__gads=ID=930b916a1451f7a3-226c487bd4c500f6:T=1611930918:RT=\
    1611930918:S=ALNI_MZ7gxXD4j6U_w34cF3TmNecYpylfg;NID=208=O6hi6KjCGYh5OMBq18gWuTLyP-xI4hVgNh7Jdwf2lR3bb0Ie3d7Jn8ulm3S_\
    OnMB4e8JEur23Ukys_VBd9dEPcJp8ujPy4DP01FB9FRLpLx8ZPb5gQ_S2ps5_VwnYDY8gmSZuvY5MkCslTnCOjuv5F0098jXEcYBjQTuo_I0SUY;_gid=GA1.2.\
    2035594247.1612497151;CNZZDATA1279080935=577313372-1611927622-https%253A%252F%252Fsc.panda321.com%252F%7C1612493887\
    ;_gat_gtag_UA_126288799_1=1',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'upgrade-insecure-requests': '1',
        'user-agent':'Mozilla/5.0(WindowsNT10.0;Win64;x64)AppleWebKit/537.36(KHTML,likeGecko)Chrome/88.0.4324.146Safari/537.36'
    }
    #获取的网页内容以wb+的方式写入，打开以utf-8的方式打开，注意后面文件打开的方式都得这样
    foo=open(filePath+'website.txt','a',encoding='utf-8')#将处理过的网站写保存在website.txt中
    foo.write(website)
    foo.write('\n')
    foo.close()
    judge=40#判断获取文件的行数
    i=0
    try:
        while judge>39 and i<10:  #应对防爬
            if i%2==1:
                text = requests.get(website,headers=header1,timeout=30)
            else:
                text = requests.get(website, headers=header2,timeout=30)
            fo = open(filePath + 'content.txt', 'wb+', )  # 方便检查错误
            fo.write(text.content)
            fo.close()
            file = open(filePath + 'content.txt', 'r', encoding='utf-8')
            data = file.read()
            file.close()
            i=i+1
            judge=len(data.split('\n'))
            print('\r正在进行%d次访问....|~_~|'%i)
            time.sleep(1)
    except (requests.exceptions.ReadTimeout, UnicodeDecodeError, requests.exceptions.ConnectionError):
        pass
    #如果爬取7次都失败，那就跳过这个网页，同时将跳过的网页以某种方式告诉并且保存在某个文件夹中
    if judge>39 and i==10:
        file = open(filePath + 'content.txt', 'w', encoding='utf-8')
        file.write('')
        file.close()
        file1 = open(filePath + 'unGetWeb.txt', 'a', encoding='utf-8')
        file1.write(website)
        file1.write('\n')
        file1.close()
        y_nMark='No'
        print(website+'访问10次都失败，将跳过此网页处理下一网页，同时该网址保存在unGetWeb.txt中')
    else:
        y_nMark='Yes'
        print('_______________访问成功|^_^|')
    return(y_nMark)
def ContentGet(filePath):
    articleID=[]
    articleTitle=[]
    articleDownload=[]
    articleSummery=[]
    articleTime=[]
    articleCite=[]
    articleOrigin=[]
    articlePuber=[]
    filterList=[]
    file = open(filePath + 'content.txt', 'r', encoding='utf-8')
    data = file.read()
    file.close()
    datalist = data.split('a id')
    for strs in datalist[1:]:
        if 'cites' in strs:
            startIndex = strs.find(' href=')
            endIndex = strs[startIndex + 7:].find('"') + startIndex + 7  # '" href="'字符占了7个位置
            articleID.append(strs[startIndex-13:startIndex-1])
            articleDownload.append(strs[startIndex + 7:endIndex])

            startIndex = strs[endIndex:].find('>') + endIndex
            endIndex = strs[startIndex:].find('</a>') + startIndex
            articleTitle.append(strs[startIndex + 1:endIndex])

            startIndex = strs[endIndex:].find(' -') + endIndex
            endIndex = strs[startIndex:].find('</div>') + startIndex
            tempIndex = strs[startIndex + 6:endIndex][::-1].find('-')
            articleOrigin.append(strs[startIndex + 6:endIndex][::-1][0:tempIndex - 1][::-1])
            articleTime.append(strs[startIndex + 6:endIndex][::-1][tempIndex + 2:tempIndex + 6][::-1])
            if articleOrigin[-1] in journalInpo:
                tempIndex1=strs[startIndex + 6:endIndex][::-1][tempIndex + 1:].find('-')
                articlePuber.append(strs[startIndex + 6:endIndex][::-1][tempIndex+8:tempIndex1+tempIndex][::-1])
            else:
                articlePuber.append('None')

            startIndex = strs[endIndex:].find('gs_rs') + endIndex
            endIndex = strs[startIndex:].find('</div>') + startIndex
            articleSummery.append(strs[startIndex + 7:endIndex].replace('\n',''))

            startIndex = strs[endIndex:].find('cites') + endIndex
            endIndex = strs[startIndex:].find('</a>') + startIndex
            tempstr = strs[startIndex + 5:endIndex][::-1]
            if len(tempstr) > 3:
                l = 1
                while tempstr[l] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    l = l + 1
                articleCite.append(tempstr[:l][::-1])
            else:
                pass
        else:
            startIndex = strs.find(' href=')
            endIndex = strs[startIndex + 7:].find('"') + startIndex + 7  # '" href="'字符占了7个位置
            filterList.append(strs[startIndex - 13:startIndex - 1])
            filterList.append(strs[startIndex + 7:endIndex])
    for i in range(len(articleID)):
        if articleID[i] in filterList and articleDownload[::-1][:4]!='fdp.':
            a=filterList.index(articleID[i])
            articleDownload[i]=filterList[a+1]
    return(articleTitle, articleDownload,articleOrigin,articlePuber,articleTime, articleCite, articleSummery)
def DataSave(dataName,filePath,articleTitle, articleDownload,articleOrigin,articlePuber,articleTime, articleCite, articleSummery):
    if dataName+'.txt' in os.listdir(filePath):
        fo=open(filePath+dataName+'.txt','r',encoding='utf-8')
        data=fo.read().split('\n')
        fo.close()
        if len(data)>2:
            ordernumber=int(data[len(data)-2].split('<>')[0].rstrip())
            DataSaveSub(dataName,filePath,articleTitle, articleDownload,articleOrigin,articlePuber,articleTime, articleCite, articleSummery,startNumber=ordernumber+1,endnNumber=ordernumber+len(articleCite)+1)
        else:
            DataSaveSub(dataName,filePath,articleTitle, articleDownload,articleOrigin,articlePuber,articleTime, articleCite, articleSummery,startNumber=0,endnNumber=len(articleCite))
    else:
        fo = open(filePath + dataName+'.txt', 'w+',encoding='utf-8')
        fo.write('order        <>title        <>web        <>origin        <>journal        <>time        <>cite        <>summery')
        fo.write('\n')
        fo.close()
        DataSaveSub(dataName,filePath,articleTitle, articleDownload,articleOrigin,articlePuber,articleTime, articleCite, articleSummery,startNumber=0,endnNumber=len(articleCite))
def DataSaveSub(dataName,filePath,articleTitle, articleDownload,articleOrigin,articlePuber,articleTime, articleCite, articleSummery,startNumber,endnNumber):
    fo = open(filePath + dataName+'.txt', 'a', encoding='utf-8')
    for i in range(startNumber, endnNumber):
        fo.write(str(i))
        fo.write('   <>')
        fo.write(articleTitle[i - startNumber])
        fo.write('     <>')
        fo.write(articleDownload[i - startNumber])
        fo.write('     <>')
        fo.write(articlePuber[i - startNumber])
        fo.write('     <>')
        fo.write(articleOrigin[i - startNumber])
        fo.write('     <>')
        fo.write(articleTime[i - startNumber])
        fo.write('     <>')
        fo.write(articleCite[i - startNumber])
        fo.write('     <>')
        fo.write(articleSummery[i - startNumber].replace('<br/>',''))
        fo.write('\n')
    fo.close()
def DataOpen(dataName,filePath):
    JudgeLine = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    articleID = []
    articleTitle = []
    articleDownload = []
    articleOrigin = []
    articlePuber = []
    articleTime = []
    articleCite = []
    articleSummery = []
    fo = open(filePath + dataName+'.txt', 'w+', encoding='utf-8')
    fo.readline()
    alldata = fo.read().split('\n')
    alldata.pop()
    for strs in alldata:
        if strs[0] in JudgeLine:
            templist=strs.split('<>')
            articleID.append(templist[0].rstrip())
            articleTitle.append(templist[1].rstrip())
            articleDownload.append(templist[2].rstrip())
            articleOrigin.append(templist[3].rstrip())
            articlePuber.append(templist[4].rstrip())
            articleTime.append(templist[5].rstrip())
            articleCite.append((templist[6].rstrip()))
            articleSummery.append(templist[7].rstrip())
        else:
            pass
    return (articleID,articleTitle, articleDownload, articleOrigin, articlePuber, articleTime, articleCite, articleSummery)
def Fitlerdata(Sci_hub_Down,SingleDoiweb,SingleDoiwebSub,articleID,articleDownload,articlePuber):
    articleDownload_1=[]
    for i in range(len(articleID)):
        articleDownload_1.append('None')
    inDirctDownload=[]
    inDirctDownloadSub=[]
    for i in range(len(articleDownload)):
        if articleDownload[i][len(articleDownload[i])-4:] == '.pdf':  # 网址以.pdf结尾可直接访问pdf文件
           pass
        elif articleDownload[i][len(articleDownload[i])-4:] != '.pdf' \
                and '/10.' in articleDownload[i]:  # 从网站本身找DOI
            startIndex = articleDownload[i].find('/10.')
            endIndex = articleDownload[i][startIndex + 1:].find('/')
            endIndex_judge = articleDownload[i][startIndex + 1:].rfind('/')
            if endIndex == endIndex_judge:  # 最后两组是DOI
                articleDownload_1[i]=Sci_hub_Down + articleDownload[i][startIndex + 1:] + '.pdf'
            else:  # 中间两组是DOI
                endIndex_next = endIndex = articleDownload[i][startIndex + endIndex + 2:].find('/')
                articleDownload_1[i]=Sci_hub_Down +articleDownload[i][startIndex + 1:startIndex + endIndex + endIndex_next] + '.pdf'
                '''
        elif articleDownload[i][len(articleDownload[i]) - 4:] != '.pdf' \
                and '/10.' not in articleDownload[i] and articlePuber[i] in SingleDoiweb:  # 某些DOI只有一段的
            i = SingleDoiweb.index(articlePuber[i].split('/')[-1])
            articleDownload_1[i]=Sci_hub_Down + SingleDoiwebSub[i] + articleDownload[i].split('/')[-1] + '.pdf'
'''
        else:  # 某些需要特殊手段获得，如访问sci-hub，
            inDirctDownload.append(articleDownload[i])
            inDirctDownloadSub.append(articleID[i])
    return(inDirctDownload,inDirctDownloadSub,articleDownload_1)
def GetDown_Link(filePath,Sci_hub, inDirctDownload, inDirctDownloadSub, Sci_hub_Down,articleID):
    print('|-------------------------------------------|')
    print('|-------正在从云端获取文献DOI值------------|')
    print('|----已智能帮您获得%d个DOI，剩余%d个请等待—----|' %(len(articleID)-len(inDirctDownloadSub),len(inDirctDownload)))
    print('|-------------------------------------------|')
    inDirctDownload_new=[]
    inDirctDownloadSub_new=[]
    for i in range(len(inDirctDownload)):
        inDirctDownload[i]=Sci_hub+inDirctDownload[i]
    for i in range(len(inDirctDownload)):
        print('>剩余%d个，总%d....顺序号%s' %(len(inDirctDownload)-i,len(inDirctDownload),inDirctDownloadSub[i]))
        try:
                print('正在爬取web:%s|~_~|' % inDirctDownload[i][19:])
                text = requests.get(inDirctDownload[i], timeout=15)
                fo = open(filePath + 'SciContent.txt', 'wb+')
                fo.write(text.content)
                fo.close()
                foo = open(filePath + 'SciContent.txt', 'r', encoding='utf-8')
                data = foo.read()
                foo.close()
                if '</title>' in data:
                    endIndex = data.find('</title>')
                    startIndex = data[:endIndex][::-1].find('|')
                    inDirctDownload_new.append(Sci_hub_Down + data[endIndex - startIndex + 1:endIndex] + '.pdf')
                    inDirctDownloadSub_new.append(inDirctDownloadSub[i])
                else:
                    print(inDirctDownload[i][20:], '请求错误！请手动打开!对应序号和网站名保存在unOpen_web.txt下')
                    fooo = open(filePath + 'unOpen_web.txt', 'a', encoding='utf-8')
                    fooo.write(inDirctDownload[i][20:])
                    fooo.write('          <>')
                    fooo.write(inDirctDownloadSub[i])
                    fooo.write('\n')
                    fooo.close()
        except (requests.exceptions.ReadTimeout, UnicodeDecodeError, requests.exceptions.ConnectionError):
            print(inDirctDownload[i][20:], '请求超时!对应序号和网站名保存在unOpen_web.txt下')
            fooo = open(filePath + 'unOpen_web.txt', 'a', encoding='utf-8')
            fooo.write(inDirctDownloadSub[i][20:])
            fooo.write('          <>')
            fooo.write(inDirctDownload[-1])
            fooo.write('\n')
            fooo.close()
            pass
    return(inDirctDownload_new,inDirctDownloadSub_new)
def NewWeblist(inDirctDownload,inDirctDownloadSub,articleDownload,articleID,articleDownload_1):
    for i in range(len(inDirctDownload)):
        index = articleID.index(inDirctDownloadSub[i])
        articleDownload_1[index]=inDirctDownload[i]
    return (articleID,articleDownload_1)
def ProduceHtml(keyword,filePath,articleID,articleTitle, articleDownload,articleDownload_1, articleOrigin, articlePuber, articleTime, articleCite, articleSummery):
    fo=open(filePath+keyword+'.html','a',encoding='utf-8')
    fo.write('<html>\n   <head>\n      <mate charset="utf-8>\n      <title>by 甲虫壳</title>\n   </head>\n')
    fo.write('   <body>\n      <center>\n        <h2><font color="bule" size="7"> %s </font></br></h2>\n      </center>\n' %(keyword))
    for i in range(len(articleID)):
        fo.write('      <b><font color="gray" size="3"> %s -</font></b>\n' %(articleID[i]))
        if articleDownload_1[i] == 'None':
            fo.write('<a href=" %s " target="_blank"><font color="bule" size="5.5"> %s </font></b></br></b></a>\n' %(articleDownload[i],articleTitle[i]))
        else:
            fo.write('<a href=" %s " target="_blank"><font color="red" size="5.5"> %s __</font></b></b></a>\n' % (
                articleDownload_1[i], '[PDF]'))
            fo.write('<font size="5.5"> %s </font></b></b></a>\n' % ('   '))
            fo.write('<a href=" %s " target="_blank"><font color="bule" size="5.5"> %s </font></b></br></b></a>\n' % (
            articleDownload[i], articleTitle[i]))
        fo.write('<b><font color="gray" size="3">-----From: %s -%s </br>\n' %(articleOrigin[i],articlePuber[i]))
        fo.write(' <b><font color="black" size="4.5">Summary: %s </font></b></br></b>\n' %(articleSummery[i]))
        fo.write(' <b><font color="gray" size="3">-----Cites: %s -year: %s </font></b></br></br></b>\n' %(articleCite[i],articleTime[i]))
        fo.write('\n')
        fo.write('\n')
    fo.write('      </body>\n   </html>\n')
    fo.close()
    print('镜像网页以生成，请用浏览器打开%s中的browse.html查看|^_^|'%(filePath))
    print('________________by 甲虫壳________________')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    SystemUI()
    #keyword=input('>请输入\033[31m英文关键词\033[0m,多个关键词请用"\033[31m + \033[0m",输入完毕回车确认:')
    #yearSelect=input('>是否选择发表时间,\033[31m时间范围\033[0m请输入如"2018-2021",\033[31m自某时间以来\033[0m直接输入起始年份,不需要直接输入,"\033[31m pass\033[0m "\n:')
    keyword = input('>请输入英文关键词,多个关键词请用\'+\',输入完毕回车确认:')
    yearSelect = input('>是否选择发表时间,时间范围请输入如"2018-2021",自某时间以来直接输入\'起始年份\',不需要直接输入,"pass"\n:')
    filePath=os.getcwd()+'\\'
    pageMark=input('请输入需要检索的数量(1000个大约需要40min):')
    pageMark=int(pageMark)//10  #每页内容10个
    #里面有很多其他期刊
    print('程序正在进行第一阶段操作，总三个阶段')
    journalInpo=['rsc.org','pubs.rsc.org','ACS Publications','Wiley Online Library','nature.com','science.sciencemag.org','Taylor & Francis','Springer','ieeexplore.ieee.org','dl.acm.org','iopscience.iop.org','Elsevier']
    #这个程序通过谷歌学术获得所有内容
    big_ContentGet(keyword, pageMark, yearSelect)
    print('程序正在进行第二阶段操作，总三个阶段')
    #下面获取pdf下载链接，并储存在文件中
    Sci_hub = 'https://sci-hub.se/'  # SCI HUB下载的链接前缀
    Sci_hub_Down = 'https://sci.bban.top/pdf/'  # SCI HUB直接获得文件的链接
    SingleDoiweb = ['rsc.org', 'Taylor&Francis', 'pubs.rsc.org', 'nature.com', 'JSTOR', 'science.sciencemag.org']
    SingleDoiwebSub = ['10.1039/', '10.1080/', '10.1039/', '10.1038/', '10.2307/', '10.1126/']
    #打开文件获得所有数据
    articleID,articleTitle, articleDownload, articleOrigin, articlePuber, articleTime, articleCite, articleSummery=DataOpen(keyword,filePath)
    #初步获得DOI和下载链接
    inDirctDownload, inDirctDownloadSub,articleDownload_1=Fitlerdata(Sci_hub_Down,SingleDoiweb,SingleDoiwebSub,articleID,articleDownload,articlePuber)
    #利用sci-hub获取DOI
    inDirctDownload,inDirctDownloadSub=GetDown_Link(filePath,Sci_hub, inDirctDownload, inDirctDownloadSub, Sci_hub_Down,articleID)
    #上面一步获取的进行替换
    articleID,articleDownload_1=NewWeblist(inDirctDownload,inDirctDownloadSub,articleDownload,articleID,articleDownload_1)
    print(articleDownload_1)
    #将所有信息做成一个html
    print('程序正在进行第三阶段操作，总三个阶段')
    print('|^_^|,|^_^|,|^_^|,|^_^|,|^_^|,|^_^|,|^_^|,|^_^|,|^_^|')
    ProduceHtml(keyword,filePath,articleID,articleTitle, articleDownload,articleDownload_1, articleOrigin, articlePuber, articleTime, articleCite, articleSummery)
    print('程序结束！')
    input("Press <enter>")
