import gdown
import zipfile

def dowload():
    url = 'https://drive.google.com/uc?export=download&id=19hkj39T7hmJZArct1v-VfIjhV3rk0gyo'
    output = 'models.zip'
    gdown.download(url, output, quiet=False)
    with zipfile.ZipFile('models.zip', 'r') as zip_ref:
        zip_ref.extractall('./')