import csv


#
def total_vote_count(data):
    total_votes = {label: data[label]['VVD'] +
                          data[label]['CDA'] +
                          data[label]['PVV'] +
                          data[label]['D66'] +
                          data[label]['SP'] +
                          data[label]['GROENLINKS'] +
                          data[label]['PvdA'] +
                          data[label]['50PLUS'] +
                          data[label]['CU'] +
                          data[label]['PvdD'] +
                          data[label]['SGP'] +
                          data[label]['FvD'] for label in data.keys()}
    return total_votes


#
def read_data(
        votesfile='C:/Users/Max\Google Drive/Datasets/Dutch 2017 Parliament Elections/Municipal_results_parliament_2017.csv',
        geofile='C:/Users/Max/Google Drive/Datasets/Dutch 2017 Parliament Elections/Municipal_locations_2019.csv'):
    print('Reading municipality data')
    geo_data = dict()

    with open(geofile, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            geo_data[row['NAAM']] = {'Provincie': row['Provincie'],
                                     'x': float(row['Lon'].replace(',', '.')),
                                     'y': float(row['Lat'].replace(',', '.'))}
    data = dict()

    print('Reading voting data')
    with open(votesfile, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            label = row['Gemeente']
            if label in geo_data:
                data[label] = {'VVD': int(row['VVD']),
                               'CDA': int(row['CDA']),
                               'PVV': int(row['PVV']),
                               'D66': int(row['D66']),
                               'SP': int(row['SP']),
                               'GROENLINKS': int(row['GROENLINKS']),
                               'PvdA': int(row['PvdA']),
                               'CU': int(row['CU']),
                               '50PLUS': int(row['50PLUS']),
                               'PvdD': int(row['PvdD']),
                               'SGP': int(row['SGP']),
                               'FvD': int(row['FvD']),
                               'Provincie': geo_data[label]['Provincie'],
                               'Lon': geo_data[label]['x'],
                               'Lat': geo_data[label]['y']
                               }
    return data
#

