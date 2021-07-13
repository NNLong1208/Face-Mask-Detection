import gdown
import zipfile

def dowload():
    url = 'https://drive.google.com/uc?export=download&id=10Jm4ztCeV9dqVMVGzLP9B3iOyUb2EKLJ'
    output = 'models.zip'
    gdown.download(url, output, quiet=False)
    with zipfile.ZipFile('models.zip', 'r') as zip_ref:
        zip_ref.extractall('./')