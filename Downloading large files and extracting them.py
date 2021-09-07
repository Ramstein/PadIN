import wget

from zipfile import ZipFile
from google_drive_downloader import GoogleDriveDownloader as gdd



print('Beginning file downlaod with wget module')

# url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'


url = 'https://drive.google.com/open?id=1u6axu8yXQXCcgvIZ9nb47j07TBsQyEor'
try:
    wget.download(url, 'C:/Users/zeeshan/PycharmProjects/')
except:
    print('wget module failed, downloading by google_drive_downlaoder')
    try:
        id = url.split('=')[len(url.split('='))-1]
        gdd.download_file_from_google_drive(file_id=id,
                                            dest_path='C:/Users/zeeshan/PycharmProjects/PadIN 1.0',
                                            unzip=False)
        print('file downloaded')
    except:print('downloading failed.')



# https://drive.google.com/open?id=1VRPhH52Co6aOZq54_UfJc_YjXmZTAP0n

def main():
    print('1. Extract all files in ZIP to current directory')
    # Create a ZipFile Object and load sample.zip in it
    with ZipFile('sampleDir.zip', 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall()

    print('2. Extract all files in ZIP to different directory')

    # Create a ZipFile Object and load sample.zip in it
    with ZipFile('sampleDir.zip', 'r') as zipObj:
        # Extract all the contents of zip file in different directory
        zipObj.extractall('temp')

    print('3. Extract single file from ZIP')

    # Create a ZipFile Object and load sample.zip in it
    with ZipFile('sampleDir.zip', 'r') as zipObj:
        # Get a list of all archived file names from the zip
        listOfFileNames = zipObj.namelist()
        # Iterate over the file names
        for fileName in listOfFileNames:
            # Check filename endswith csv
            if fileName.endswith('.csv'):
                # Extract a single file from zip
                zipObj.extract(fileName, 'temp_csv')


# if __name__ == '__main__':
#      main()

