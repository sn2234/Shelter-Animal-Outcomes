import math
import pandas as pd
import numpy as np
from sklearn import preprocessing

def parseAge(ageStr):
    if type(ageStr) is float:
        return ageStr

    parts = ageStr.lower().split()
    if len(parts) != 2:
        raise Exception("Unable to parse age: {0}".format(ageStr))

    ageDays = 1
    if parts[1].find("day") != -1:
        ageDays = 1
    elif parts[1].find("week") != -1:
        ageDays = 7
    elif parts[1].find("month") != -1:
        ageDays = 30
    elif parts[1].find("year") != -1:
        ageDays = 365
    else:
        raise Exception("Unable to parse age: {0}".format(ageStr))

    return float(parts[0])*ageDays

class DataLoader:

    colEncoders = {}

    def loadTrainData(self, fileName):
        rawData = pd.read_csv(fileName)

        # Select columns
        selectedColumns = rawData[['Name', 'DateTime', 'OutcomeType',
           'AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'Breed', 'Color']]

        # Parse dates
        selectedColumns['DateTime'] = pd.to_datetime(selectedColumns['DateTime'])
        #And convert timestamps to floats
        selectedColumns['DateTime'] = selectedColumns['DateTime'].map(lambda x: x.timestamp())

        # Parse age and fill missing entries with mean value
        selectedColumns.loc[:, ('AgeuponOutcome')] = selectedColumns['AgeuponOutcome'].map(parseAge)

        imp = preprocessing.Imputer()
        selectedColumns.loc[:, ('AgeuponOutcome')] = imp.fit_transform(selectedColumns['AgeuponOutcome'].reshape(-1, 1))


        # Fill other missing values - hack
        selectedColumns = selectedColumns.fillna('NA')

        for iCol in range(len(selectedColumns.columns)):
            if selectedColumns.dtypes[iCol].kind == 'O':
                col = selectedColumns.columns[iCol]
                enc = preprocessing.LabelEncoder()
                if col != 'OutcomeType':
                    selectedColumns[col] =  enc.fit_transform(selectedColumns[col]).astype(float)
                self.colEncoders[col] = enc

        preprocessing.scale(selectedColumns[['Name', 'DateTime', 'AnimalType', 'SexuponOutcome',
                                'AgeuponOutcome', 'Breed', 'Color']], copy =  False)

        return (selectedColumns[['Name', 'DateTime', 'AnimalType', 'SexuponOutcome',
                                'AgeuponOutcome', 'Breed', 'Color']].values,
                selectedColumns['OutcomeType'].values)
