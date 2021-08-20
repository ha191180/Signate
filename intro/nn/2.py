# ライブラリのインポート
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from PIL import Image, ImageFilter

DEVICE= "cpu"

# モデル訓練用関数
def train_model(model, criterion, optimizer, num_epochs=5,is_saved = False):
    best_acc = 0.0

    # エポック数だけ下記工程の繰り返し
    for epoch in range(num_epochs):

        for phase in ['train', 'val']:
            print('{}:フェイズ'.format(phase))

            # 訓練フェイズと検証フェイズの切り替え
            if phase == 'train':
                model.train() 
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0

            # dataloadersからバッチサイズだけデータ取り出し、下記工程（1−5）の繰り返し
            for i,(inputs, labels) in enumerate(image_dataloaders[phase]):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # 1. optimizerの勾配初期化
                optimizer.zero_grad()
                    
                # 2.モデルに入力データをinputし、outputを取り出す
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                # 3. outputと正解ラベルから、lossを算出
                loss = criterion(outputs, labels)
                print('   loaders:{}回目'.format(i+1)  ,'   loss:{}'.format(loss))

                if phase == 'train':                        
                    # 4. 誤差逆伝播法により勾配の算出
                    loss.backward()
                    # 5. optimizerのパラメータ更新
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # C. 今までのエポックでの精度よりも高い場合はモデルの保存
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                if(is_saved):
                    torch.save(model.state_dict(), './original_model_{}.pth'.format(epoch))

    print('Best val Acc: {:4f}'.format(best_acc))
 



    
    