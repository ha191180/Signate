from PIL import Image
from torchvision import transforms, datasets, models
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import os
import zipfile


# 画像の読み込み
# img = Image.open('cast_def_0_1055.jpeg')


# zip解凍用の関数定義
def unzip_dataset(INPATH,OUTPATH):
    with zipfile.ZipFile(INPATH) as zf:
        zf.extractall(OUTPATH)

# zip解凍(※数秒から数十秒かかる場合があります)
unzip_dataset(INPATH='./image_data.zip',OUTPATH='./')

# osを使って、ファイルの一覧確認
print(os.listdir('./image_data/train/ok'))


# transformsの定義
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


'''
各フォルダに学習用と検証用のデータが揃い、かつ、クラスごと(=ラベルごと)にフォルダが分かれている場合はdatasets.ImageFolder()を用います。
引数には読み込みたい画像データがあるフォルダのパスと、先ほど定義したtransformsを指定します。
datasetの構造についてもtransform同様に学習と検証をわけて記載します。
これにより画像がクラスごと（=フォルダごと）に読み込まれ、transformsに定義したデータ処理を行なってくれます。
今回のフォルダ構造が学習と検証、またokとngで分けていた理由の1つとなります。
'''

# dataset作成
image_datasets = {
    'train': datasets.ImageFolder('./image_data/train',data_transforms['train']),
    'val': datasets.ImageFolder('./image_data/val',data_transforms['val'])
}

'''
Dataloaderは、datasetsから画像データをバッチごとに取り出すことを目的に使われます。
バッチ毎にデータを取り出すことによって、学習を行う計算機のスペックに合わせて学習を行うことが可能となります。

Dataloaderはtorch.utils.data.DataLoaderを用い、メインで用いる引数の詳細は以下となります。

引数dataset: loaderで読み込むデータセット
引数batch_size: バッチサイズ
引数shuffle: データをshuffleするか（True もしくは False）
引数num_workers: 実行プロセス数（デフォルトは0）
引数drop_last: 残りデータ数がバッチサイズに満たない場合にスキップするか（True もしくは False）

Dataloaderからのデータの取り出しにはfor文を使います。
for文により、1loopする度に、loaderからバッチサイズ分のinputデータとlabelデータを取り出すことができます。

Ex: 
for i ,(inputs,labels) in enumerate(image_dataloaders['train']):
  print(inputs)
  print(labels)

'''



# dataloaders作成
image_dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=4,shuffle=True, num_workers=0, drop_last=True),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=4,shuffle=False, num_workers=0, drop_last=True),
}


'''
【torchvisionに実装済みの代表的画像分類モデル】

AlexNet
VGG
ResNet
SqueezeNet
DenseNet
Inception v3
GoogLeNet
ShuffleNet v2
MobileNet v2
ResNeXt
Wide ResNet
MNASNet
etc...
'''


# 事前学習無のResNet18作成
model_ft = models.resnet18(pretrained=False)

# 定数設定
device = "cpu"
TARGET_NUM = 2

# モデル作成関数の定義
def get_model(target_num,isPretrained=False):
    """for get model"""
    
    model_ft = models.resnet18(pretrained=isPretrained)
    model_ft.fc = nn.Linear(512, target_num)
    model_ft = model_ft.to(device)
    return model_ft


model = get_model(TARGET_NUM, False)

'''
optimizerを作成します。
学習率lrは比較的小さめの0.001、モーメンタムmomentumは0.9に設定します。
学習率は0.1などの大きいものに設定するとうまく学習しない可能性があるのでまずは0.001とし、学習の塩梅を確認しながら必要があれば再設定していきましょう。
モーメンタムはパラメータの更新をより慣性的なものにするという狙いがあります。
optimizerの勾配の初期化を行う際は、optimizer.zero_grad()、ネットワークのパラメータの更新はoptimizer.step()で行います。
'''

# 最適化関数定義
optimizer = optim.SGD(model_ft.parameters(),lr=0.001, momentum=0.9)

'''
精度の高い予測モデルを作成するためには、まずモデルの予測精度を評価する必要があります。
予測精度を評価する際に用いるのがloss（損失）関数です。
loss（損失）関数は予測と実際の値のズレの大きさを表す関数でモデルの予測精度を評価します。
loss（損失）関数の値が小さければより正確なモデルと言えます。ネットワークを学習させる過程でうまくlossが小さくなっているかが学習がうまくいっているのかの指標となります。

鋳造製品の欠陥検出は正常品であるか、欠陥品であるかの分類問題となるため交差エントロピー誤差を用います。
'''

# loss関数定義
criterion = nn.CrossEntropyLoss()

'''
Chart
# エポック数だけ下記工程の繰り返し
    # 訓練フェイズ
    A. モデルを訓練モードに変更
        ＃ dataloadersからバッチサイズだけデータ取り出し、下記工程（1−5）の繰り返し
        1. optimizerの勾配初期化
        2. モデルに入力データをinputし、outputを取り出す
        3. outputと正解ラベルから、lossを算出
        4. 誤差逆伝播法により勾配の算出
        5. optimizerのパラメータ更新

    # 推論フェイズ        
    B. モデルを推論モードに変更
        ＃ dataloadersからバッチサイズだけデータ取り出し、下記工程（1、２）の繰り返し
        1. optimizerの勾配初期化
        2. モデルに入力データをinputし、outputを取り出す
        3. outputと正解ラベルから、lossを算出

    C. 今までのエポックでの精度よりも高い場合はモデルの保存
'''

# モデル学習用関数
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
                inputs = inputs.to(DEVICE) #--> Picture array data
                labels = labels.to(DEVICE) #--> labels data like [0,1,1,1]

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



### FOR PREDICTION ###

unzip_dataset(INPATH='./test_data.zip',OUTPATH='./')

# DataFrame作成
df_test = pd.DataFrame(data=os.listdir('./test_data/'))

# カラム名変更
df_test = df_test.rename(columns={0: 'filename'})

# targetカラム作成
df_test['target'] = 0
df_test.loc[df_test['filename'].str.contains('ok'),'target'] = 1

# csv保存
df_test.to_csv('df_test.csv', index=False)

# 推論用のtransforms作成
test_transforms = transforms.Compose([
               transforms.Resize(256),
               transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

])

'''
__len__関数にて、データセットの大きさを定義します。今回であれば10枚の画像を持っていることを定義しましょう。
また、#3：__getitem__関数の定義では、obj[i]のようにインデックスで指定されたときにコールされる関数となります。
なので、__getitem__では画像1枚1枚をPillowを用いて読み込み、その1枚1枚に対してtransformerで定義したデータ前処理を適用しましょう。
今回は推論用のデータセットなのでラベル（正常品であるか欠陥品であるか）は付与せず、各々のファイル名を返す処理を記載しています。
'''

# dataset作成
class Test_Datasets(Dataset):

    def __init__(self, data_transform):
        
        self.df = pd.read_csv('./df_test.csv',names=['filename','target'])
        self.data_transform = data_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        
        file = self.df['filename'][index]
        image = Image.open('./test_data/'+ file)
        image = self.data_transform(image)

        return image,file

# datasetのインスタンス作成
test_dataset = Test_Datasets(data_transform=test_transforms)

# dataloader作成
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=1 , 
                                                shuffle=False, 
                                                num_workers=0, 
                                                drop_last=True )


'''
torch.load()を用い、map_location=lambda storage, loc: storageを用いてることでモデルの各層のパラメータを各層ごとにロードしています。
'''

best_model = get_model(target_num=2)

best_model.load_state_dict(torch.load('./original_model_33.pth', map_location=lambda storage, loc: storage), strict=True)

print(best_model)

pred = []

# データの取り出し
for i,(inputs, labels) in enumerate(test_dataloader):

    inputs = inputs.to(DEVICE)

    # 学習済みモデルを推論モードに設定
    best_model.eval()

    # 学習済みモデルにデータをインプットし、推論をさせる
    outputs = best_model(inputs)

    # アウトプットから推定されたラベルを取得
    _, preds = torch.max(outputs, 1)

    # 事前に用意したリストに推論結果(0 or 1)を格納
    pred.append(preds.item())

df_test['pred'] = pred





