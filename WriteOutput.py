from __future__ import division
import pdb as check
import csv

def ReadCSV(filenameCSV, col):
    field_names = []
    with open(filenameCSV, 'r') as csvfile:
        csvreader = csv.reader(csvfile,  delimiter=',')
        for row in csvreader:
            field_names.append(row[col])
    return field_names

def WriteCSV(fileName, BigBig):
    filenameCSV = fileName[1:-4] + '_grid.csv'
    field_names = ReadCSV('./FieldsToWrite.csv', 0)
    with open(filenameCSV, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile,  delimiter=',')
        writer = csv.DictWriter(csvfile, fieldnames = field_names)
        writer.writeheader()
        for i in range(len(BigBig['x'])):
            csvwriter.writerow([BigBig[field_name][i] for field_name in field_names])

    print 'Calculations written successfully to file: ' +  filenameCSV

    return filenameCSV
