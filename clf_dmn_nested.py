import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import xml.etree.ElementTree as et
import graphviz
from collections import defaultdict
import random
import string
import layered_classifier


class xmlDmn:
    def __init__(self, dmnName, decisionId=None, prevDecisionId=None):
        self.tree = et.parse(dmnName)
        self.xmlRoot = self.tree.getroot()  
        self.namespace = self.xmlRoot.tag.split("}")[0].strip("{")
        self.namespace2 = "http://bpmn.io/schema/dmn/biodi/1.0"
        self.ns = {"xmlns" : self.namespace, "biodi": self.namespace2}
        if decisionId is not None:
            self.decision = self.xmlRoot.find(".//*[@id='%s']"%(decisionId), namespaces=self.ns)
            self.decisionTableElement = self.decision.find(".//xmlns:decisionTable", namespaces=self.ns)  
            self.decId = decisionId
        else:
            self.decisionTableElement, self.decId = self.createDecision(self.xmlRoot, prevDecisionId)

    def createDecision(self, root, prevDec):
        xmlRoot = root
        prevPosition = xmlRoot.find(".//*[@id='%s']/xmlns:extensionElements/biodi:bounds"%(prevDec),self.ns)
        newDecisionId = idGen("Decision_")
        newDecision = et.SubElement(xmlRoot,"{%s}decision"%(self.namespace), attrib={"id":newDecisionId,"name":idGen("SecondLevel_")})
        newExEle = et.SubElement(newDecision,"{%s}extensionElements"%(self.namespace))
        newBounds = et.SubElement(newExEle,"{%s}bounds"%(self.namespace2), attrib={"x": str(int(prevPosition.attrib["x"])+300), "y": "80","width":"180","height":"80" })
        newDecTable = et.SubElement(newDecision,"{%s}decisionTable"%(self.namespace), attrib={"id":idGen("decisionTable_")})
        
        return newDecTable, newDecisionId

    def createConnectionForTables(self, firstTable, secondTable):
        """
        Input tables (ids of tables) which you want to connect (from -> to),
        save as firstTable and secondTable
        """
        start = firstTable
        end = secondTable
        startDecision = self.xmlRoot.find(".//*[@id='%s']"%(start), namespaces=self.ns)
        startBounds = startDecision.find(".//xmlns:extensionElements/biodi:bounds", namespaces = self.ns)
        endDecision = self.xmlRoot.find(".//*[@id='%s']"%(end), namespaces=self.ns)
        endBounds = endDecision.find(".//xmlns:extensionElements/biodi:bounds", namespaces = self.ns)
        endExtEle = endDecision.find(".//xmlns:extensionElements", namespaces=self.ns)

        newEdge = et.SubElement(endExtEle, "{%s}edge"%(self.namespace2),attrib={"source":start})
        
        startWaypoint = et.SubElement(newEdge, "{%s}waypoints"%(self.namespace2), attrib={"x": str( int(startBounds.attrib["x"])+int(startBounds.attrib["width"]) ), "y":str(int(startBounds.attrib["y"])+40 ) } )
        endWaypoint = et.SubElement(newEdge, "{%s}waypoints"%(self.namespace2), attrib={"x":str( int(endBounds.attrib["x"]) ), "y": str( int(startBounds.attrib["y"])+40 ) }  )
        newInfReq = et.SubElement(endDecision,"{%s}informationRequirement"%(self.namespace))
        newReqDec = et.SubElement(newInfReq,"{%s}requiredDecision"%(self.namespace), attrib={"href":"#%s"%(start)})

    def printDecisionTable(self):
        """
        For debugging
        """
        for element in self.decisionTableElement:
            if self.namespace in element.tag:
                print(element.tag.split("}")[-1], ":", element.attrib)
    
    def clearDecisionTable(self, decId):
        """
        Clears dmn table and prepares it for new input
        """
        for element in  self.xmlRoot.findall(".//xmlns:decisionTable/", namespaces=self.ns):
            print(element)
            if element.tag == "{%s}input"%(self.namespace):
                self.decisionTableElement.remove(element)
            elif element.tag == "{%s}output"%(self.namespace):
                self.decisionTableElement.remove(element)
            elif element.tag == "{%s}rule"%(self.namespace):
                self.decisionTableElement.remove(element)
    
    def generateTableColumns(self,names, colTypes):
        outputName = names[-1]
        inputNames = names[:-1]
        for name in inputNames:
            newInput = et.SubElement(self.decisionTableElement, "{%s}input"%(self.namespace), attrib= {"id": idGen("input_")})
            if colTypes[name] == "int64" or colTypes[name] == "float64" or colTypes[name] == "int32":
                newInputExpression = et.SubElement(newInput, "{%s}inputExpression"%(self.namespace), attrib={"id": idGen("inputExpression_"),"typeRef":"double"}) 
            elif colTypes[name] == "object":
                newInputExpression = et.SubElement(newInput, "{%s}inputExpression"%(self.namespace), attrib={"id": idGen("inputExpression_"),"typeRef":"string"})
            elif colTypes[name] == "bool":
                newInputExpression = et.SubElement(newInput, "{%s}inputExpression"%(self.namespace), attrib={"id": idGen("inputExpression_"),"typeRef":"boolean"})
            else:
                newInputExpression = et.SubElement(newInput, "{%s}inputExpression"%(self.namespace), attrib={"id": idGen("inputExpression_"),"typeRef":"date"})
            newText = et.SubElement(newInputExpression, "{%s}text"%(self.namespace))
            newText.text = name
        et.SubElement(self.decisionTableElement, "{%s}output"%(self.namespace), attrib={"id": idGen("output_"),"name": outputName,"typeRef":"string"})
        et.SubElement(self.decisionTableElement, "{%s}output"%(self.namespace), attrib={"id": idGen("output_"),"name": "Decision","typeRef":"string"})
        
    def createRuleCell(self,newRule):
        """
        Input parent element, save as newRule
        """
        newInEntry = et.SubElement(newRule, "{%s}inputEntry"%(self.namespace), attrib={"id":idGen("UnaryTests_")})
        newInText = et.SubElement(newInEntry,"{%s}text"%(self.namespace))
        return newInText
    
    def generateTableRows(self, mlDict, annotation=None):
        """
        Input created dictionary containing decisions and logic, save as mlDict
        Second input are annotations 
        Creates new rules based on mlDict    
        """
        for className,v in mlDict:
            newRule = et.SubElement(self.decisionTableElement, "{%s}rule"%(self.namespace), attrib={"id":idGen("DecisionRule_")})
            #newAnnot = et.SubElement(newRule, "{%s}description"%(self.namespace))
            #newAnnot.text = str(annotation)
            for featureName in v:
                tSignList = list(v[featureName].keys())
                if len(v[featureName]) == 0:
                    self.createRuleCell(newRule)               
                    continue
                if len(v[featureName]) == 1:    
                    if tSignList[0] == "not":
                        elInList = len(v[featureName][tSignList[0]])
                        notText = "not(\"{}\"),"
                        notText = (elInList*notText)
                        self.createRuleCell(newRule).text = notText[:-1].format(*v[featureName][tSignList[0]])
                    elif tSignList[0] == "is":
                        elInList = len(v[featureName][tSignList[0]])
                        isText = "\"{}\","
                        isText = (elInList*isText)
                        self.createRuleCell(newRule).text = isText[:-1].format(*v[featureName][tSignList[0]]) 
                    else:
                        self.createRuleCell(newRule).text = "{} {:.2f}".format(tSignList[0],v[featureName][tSignList[0]])

                if len(v[featureName]) == 2:
                    if tSignList[0] == "not":
                        elInListNot = len(v[featureName][tSignList[0]]) 
                        elInListIs = len(v[featureName][tSignList[1]])
                        notText = "not(\"{}\"),"
                        notText = (elInListNot*notText)
                        isText = "\"{}\","
                        isText = (elInListIs*isText)
                        self.createRuleCell(newRule).text = (isText+notText[:-1]).format(*v[featureName][tSignList[1]],*v[featureName][tSignList[0]])

                    if  tSignList[0] == "is":
                        elInListNot = len(v[featureName][tSignList[1]]) 
                        elInListIs = len(v[featureName][tSignList[0]])
                        notText = "not(\"{}\"),"
                        notText = (elInListNot*notText)
                        isText = "\"{}\","
                        isText = (elInListIs*isText)
                        self.createRuleCell(newRule).text = (isText+notText[:-1]).format(*v[featureName][tSignList[0]],*v[featureName][tSignList[1]])

                    if tSignList[0] == "<=":
                        if v[featureName][tSignList[0]] > v[featureName][tSignList[1]]:
                            self.createRuleCell(newRule).text = "]{:.2f}..{:.2f}]".format(v[featureName][tSignList[1]],v[featureName][tSignList[0]])
                        else:
                            self.createRuleCell(newRule)
                            continue
                    if tSignList[0] == ">":
                        if v[featureName][tSignList[0]] < v[featureName][tSignList[1]]:
                            self.createRuleCell(newRule).text = "]{:.2f}..{:.2f}]".format(v[featureName][tSignList[0]],v[featureName][tSignList[1]])
                        else:
                            self.createRuleCell(newRule)
                            continue
                    
            newOutEntry = et.SubElement(newRule, "{%s}outputEntry"%(self.namespace), attrib={"id":idGen("LiteralExpression_")})
            newOutText = et.SubElement(newOutEntry, "{%s}text"%(self.namespace))
            newOutText.text = str(className)

            newOutEntry = et.SubElement(newRule, "{%s}outputEntry"%(self.namespace), attrib={"id":idGen("LiteralExpression_")})
            newOutText = et.SubElement(newOutEntry, "{%s}text"%(self.namespace))
            newOutText.text = str(annotation)

    def writeTree(self, outName=None):
        """
        Input name of file where you want to save newly created dmn, save as outName
        """
        if outName is None:
            outName = "new.dmn"
            self.tree.write(outName)
        else:
            self.tree.write(outName)




def percentage(percent, whole):
    return (percent * whole) / 100.0

def idGen(text):
    """Simple ID generator"""
    return text + ''.join(random.choices(string.ascii_lowercase + string.digits, k=7))

def removeNestedLists(origList):
    flatten = []
    for i in origList:
        if type(i) != list:
            flatten.append(i)
        else:
            subFlatten = removeNestedLists(i)
            flatten += subFlatten
    return flatten 

def prepareData(fileName, targetClass=-1):
    """
    One-Hot encoding za categorical data
    """
    df = pd.read_csv(fileName)
    targetColumnName = df.columns[targetClass]
    targetColumnData = df[targetColumnName]
    
    df = df.drop(targetColumnName,1)
    df = pd.get_dummies(df)
    df[targetColumnName] = targetColumnData
    
    return df

def visualizeTree(classifier, fileName, features, classNames = None):
    """
    Tree visualization
    Input model , save as classifier
    Input file name, save as fileName
    Input name of features, save as features
    Input class names , save classNames
    """
    clf = classifier
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=features, class_names= classNames, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(fileName)  

def getFeautureImportance(oldData, X, target = -1):
    """
    Input csv data , save as oldData
	Input number of GBC weak learners, save as X
    Input location of targeted column in csv file (deafult is last column in csv file),
    save as target
    """
    df = prepareData(oldData, targetClass=target)
    dfTarget = df[df.columns[target]]
    dfData = df[df.columns.drop(df.columns[target])]
    print("Selecting important features...")
    clf = GradientBoostingClassifier(n_estimators=X)
    clf.fit(dfData, dfTarget)
    
    feature_importances = pd.DataFrame(zip(dfData.columns, clf.feature_importances_)).sort_values(by=1, ascending=False)
    final_columns = feature_importances[feature_importances[1] > 0.1*np.mean(feature_importances[1])]
    print("DONE!")
    final_columns_dict = final_columns.to_dict(orient="list")
    bestFeatureSplits = dict()
    for key, value in zip(final_columns_dict[0],final_columns_dict[1]):
        bestFeatureSplits[key]=value
    
    return bestFeatureSplits


def createClassifier(oldData, bestFeatureSplits, dmnName, decisionId = None,target = -1, clearTable = True, nextClassifer=False, prevDecId=None, createConnection=False):
    """
	Input combination of features (split points), save as bestFeatureSplits
    Input csv data , save as oldData
    Input location of column in dataframe which contains targeted classes, by the default its the last column,
    save as target
    """
    preDF = prepareData(oldData, targetClass=target)
    

    targetColumnName = preDF.columns[target]
    
    if nextClassifer==False:
        newDmnObject = xmlDmn(dmnName, decisionId)
    else:
        newDmnObject = xmlDmn(dmnName, decisionId=None, prevDecisionId=prevDecId)
    
    featuresForClf = list()
    print("Features for CLF : ")
    for column,value in bestFeatureSplits.items():
        if value > 0.02:
            featuresForClf.append(column)
            print("\t-",column,":",value)

    featuresForClf.append(targetColumnName)

    df = preDF[featuresForClf]
    dfTarget = df[df.columns[target]]
    dfData = df[df.columns.drop(df.columns[target])]
    dfFeature = df.columns.drop(df.columns[target])
    X_train, X_test, y_train, y_test = train_test_split(dfData, dfTarget, test_size=0.1)
    
    if int(percentage(2,len(X_train))) < 1:
        print("Minimum samples in leaf is less then 1\nExit ...")
        return
    else:
        minInLeafs = int(percentage(2,len(X_train)))
    
    clf = tree.DecisionTreeClassifier(min_samples_leaf=minInLeafs).fit(X_train,y_train)
    print("Accuaracy on the testing set : {:.3f}".format(clf.score(X_test,y_test)))
    print("Minimum samples in leaf : ",int(percentage(2,len(X_train))))
    
    visualizeTree(clf,fileName=idGen("new_"),features=dfFeature)

    featureNames = [dfFeature[i] for i in clf.tree_.feature]  
    leafIds = clf.apply(dfData)
    leftChildren = clf.tree_.children_left
    rightChildren = clf.tree_.children_right
    decPath = clf.decision_path(dfData)
    threshold = clf.tree_.threshold
    leafImpurity = clf.tree_.impurity

    #Prepare features for dmn
    helperSet = set()
    for feature in featuresForClf[:-1]:
        helperSet.add(feature.split("_")[0])
    featuresForDmn = list(helperSet)
    featuresForDmn.append(featuresForClf[-1])

    #decisionDictionary = defaultdict(list)
    df = df.rename(columns=lambda x: x.split("_")[0])

    #Prepare columns type
    colType = dict()
    for element in featuresForDmn:
        if type(df.dtypes[element]) != np.dtype:
            colType[element] = "object"
        elif df.dtypes[element] == "uint8":
            colType[element] = "object"
        else:
            colType[element] = df.dtypes[element] 
    
    if clearTable == True:
        newDmnObject.clearDecisionTable(decisionId)

    newDmnObject.generateTableColumns(featuresForDmn,colType)

    for i in tqdm(set(leafIds)):  
        samplesInNode = decPath.getcol(i).copy()
        rows = samplesInNode.nonzero() [0]
        sampleId = rows[0]  
        nodeIndex = decPath.indices[decPath.indptr[sampleId]:decPath.indptr[sampleId+1]]
        className = clf.classes_[np.argmax(clf.tree_.value[i])]
        inputOutput = defaultdict(dict)   
        for value in featuresForDmn[:-1]:   
               inputOutput[className][value] = {}  
        for index, nodeId in enumerate(nodeIndex): 
            nodeFeature = featureNames[nodeIndex[index-1]]
            nodeThreshold = threshold[nodeIndex[index-1]] 
            if len(nodeFeature.split("_"))>1:
                if nodeId in set(leftChildren):
                    try:
                        inputOutput[className][nodeFeature.split("_")[0]]["not"].append(nodeFeature.split("_")[1])
                    except KeyError:
                        inputOutput[className][nodeFeature.split("_")[0]]["not"] = [nodeFeature.split("_")[1]]
                
                if nodeId in set(rightChildren):
                    try:

                        inputOutput[className][nodeFeature.split("_")[0]]["is"].append(nodeFeature.split("_")[1])
                    except KeyError:
                        inputOutput[className][nodeFeature.split("_")[0]]["is"] = [nodeFeature.split("_")[1]]
            else:               
                if nodeId in set(leftChildren):
                    inputOutput[className][nodeFeature]["<="] = nodeThreshold
                if nodeId in set(rightChildren):
                    inputOutput[className][nodeFeature][">"] = nodeThreshold
        newDmnObject.generateTableRows(inputOutput.items(),calculateAnnotation(leafImpurity[i]))
    
    tableID = newDmnObject.decId

    if createConnection:
        newDmnObject.createConnectionForTables(prevDecId, tableID)
    
    newDmnObject.writeTree()
    newDF = selectDataForNextTable(leafIds,decPath,leafImpurity, oldData)

    return newDF, tableID


def calculateAnnotation(value):
    if value == 0:
        return "Continue process"
    elif value > 0 and value <= 0.3:
        return "Ask expert"
    else:
        return "For the second level"


def selectDataForNextTable(leafs, decisionPath , impurity, oldData):
    """
	Input leafs,decision path, impurity and old data
    Output is new dataframe for "Second Level"
    """
    print("Selecting data for next level table ...")
    sampleBox = []
    for i in tqdm(set(leafs)):
        if impurity[i] > 0.3:
            samplesInNode = decisionPath.getcol(i).copy()
            rows = samplesInNode.nonzero() [0]
            sampleBox.append(rows.tolist())
    print("Number of leafs for next level : ",len(sampleBox))
    sampleBox = removeNestedLists(sampleBox)
    print("Number of samples in next csv : ", len(sampleBox))
    df = pd.read_csv(oldData)
    newDataFrame = df.iloc[sampleBox]
    return newDataFrame

def createNextTable(selectedData, dmnFile, num, previousTable):
    """
	Input is smaller dataset which consist only of selected data, save as selectedData
	Input previous table id which is needed for connecting tables, save as previousTable
	Input a number of weak learners for GBC , save as num
    """
    selectedData.to_csv("newData.csv", index=False)
    bestSplits = getFeautureImportance("newData.csv", X=num)
    newDf, tableID = createClassifier("newData.csv",bestSplits,dmnFile,prevDecId=previousTable, nextClassifer=True, clearTable=False, createConnection=True)

    return newDf, tableID

if __name__ == "__main__":
    
    bestSplits = getFeautureImportance("creditData.csv", X = 200)
    newDF,table1 = createClassifier("creditData.csv", bestSplits, "test.dmn", decisionId="Decision_13nychf")
    newDF2, table2 = createNextTable(newDF, "new.dmn",num=200,previousTable=table1)
    newDF3, table3 = createNextTable(newDF2, "new.dmn",num=200, previousTable=table2)
    newDF4, table4 = createNextTable(newDF3,"new.dmn",num=200, previousTable=table3)
    #newDF5, table5 = createNextTable(newDF4,"new.dmn",num=200, previousTable=table4)
    #newDF6, table6 = createNextTable(newDF5,"new.dmn",num=200, previousTable=table5)
    #newDF7, table7 = createNextTable(newDF6,"new.dmn",num=200, previousTable=table6)
