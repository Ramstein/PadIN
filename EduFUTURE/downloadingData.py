!pip install pydrive wget
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from google.colab import auth
from oauth2client.client import GoogleCredentials

import wget
from zipfile import ZipFile

from EduFUTURE.authenticateToGdrive import AuthenticateTOGdrive

class DownloadingData:


    def download(self, url, extractionPath, drive, fileName):
        if fileName != None:
            id = url.split('=')[len(url.split('=')) - 1]

            last_weight_file = drive.CreateFile({'id': id})
            last_weight_file.GetContentFile(fileName)

            with ZipFile(fileName, 'r') as zipObj:    # extracts the file in sample_data/examples-master/cpp & dcgan & imagenet
                zipObj.extractall(extractionPath)

        if fileName == None:
            wget.download(url, 'sample_data')
            fileName = wget.detect_filename(url=url)
            print(fileName)

            # Create a ZipFile Object and load sample.zip in it
            with ZipFile('sample_data/' + fileName, 'r') as zipObj:  # extracts in the sample_data/PetImages/Cat & Dog
                # Extract all the contents of zip file in different directory
                zipObj.extractall(extractionPath)


    if __name__ == '__main__':
        # url = 'https://drive.google.com/open?id=1mcM9I-Z7NXESXVs0zLAbithwrWUNrmAj'
        url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'

        fileName = 'examples-master.zip'  '''for gdrive file the name of the file is not inferrable by the detect_filename()'''
        extractionPath = 'sample_data/'
        drive = authenticate()
        download(url=url, fileName=fileName, extractionPath=extractionPath, gDrive=False, drive=drive)