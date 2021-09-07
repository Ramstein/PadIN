from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import pickle
import datetime



class UploadModel:
    def uploadFineModel(self, drive,max, min, accIndex, lossIndex, date, CBF):
        print("Uploading the fine model to drive.")
        val_acc_upload = open(
            '{}_{}_{}_{}_{}_{}'.format(accIndex, round(max, 4), CBF[0], CBF[1], CBF[2], f":{date:%Y-%m-%d-%Hh%Mm%Ss}"),
            'rb')
        val_acc_up = pickle.load(val_acc_upload)

        model_acc = drive.CreateFile({"title": '{}_{}_{}_{}_{}_{}'.format(accIndex, round(min, 4), CBF[0], CBF[1], CBF[2],
                                                                           f":{date:%Y-%m-%d-%Hh%Mm%Ss}")})
        model_acc.SetContentFile(val_acc_up)
        model_acc.Upload()
        accFileLink = 'https://drive.google.com/open?id='+str(model_acc.get('id'))

        val_loss_upload = open(
            '{}_{}_{}_{}_{}_{}'.format(lossIndex, round(min, 4), CBF[0], CBF[1], CBF[2], f":{date:%Y-%m-%d-%Hh%Mm%Ss}"),
            'rb')
        val_loss_up = pickle.load(val_loss_upload)
        model_loss = drive.CreateFile({"title": '{}_{}_{}_{}_{}_{}'.format(lossIndex, round(min, 4), CBF[0], CBF[1], CBF[2], f":{date:%Y-%m-%d-%Hh%Mm%Ss}")})
        model_loss.SetContentFile(val_loss_up)
        model_loss.Upload()
        lossFileLink = 'https://drive.google.com/open?id='+str(model_loss.get('id'))

        print('val_acc model link: ', accFileLink)
        print('val_loss model link: ', lossFileLink)

