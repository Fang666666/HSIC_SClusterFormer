import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time
from util import print_results
from models.SClusterFormer import SClusterFormer


def resolve_dict(hp):
    return hp['run_times'], hp['patch_size'], hp['train_ratio'], hp['pca_components'], hp['emap_components'], hp['BATCH_SIZE_TRAIN'], hp['epochs'], hp['cuda']

def loadData():
    data = sio.loadmat('./data/PaviaU.mat')['paviaU']
    labels = sio.loadmat('./data//PaviaU_gt.mat')['paviaU_gt']

    # data = sio.loadmat('./data/WHU_Hi_HanChuan.mat')['WHU_Hi_HanChuan']
    # labels = sio.loadmat('./data/WHU_Hi_HanChuan_gt.mat')['WHU_Hi_HanChuan_gt']

    # data = sio.loadmat('./data/WHU_Hi_HongHu.mat')['WHU_Hi_HongHu']
    # labels = sio.loadmat('./data/WHU_Hi_HongHu_gt.mat')['WHU_Hi_HongHu_gt']

    return data, labels

def loadEMAPData():
    emapdata = sio.loadmat('./EMAP/EMAP_Data/PaviaU_EMAP.mat')['Feature_E']

    # emapdata = sio.loadmat('./EMAP/EMAP_Data/HanChuan_EMAP.mat')['Feature_E']

    # emapdata = sio.loadmat('./EMAP/EMAP_Data/HongHu_EMAP.mat')['Feature_E']

    return emapdata

def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX


def global_standardization(feature):
    mean_value = np.mean(feature)
    std_value = np.std(feature)
    normalized_feature = (feature - mean_value) / std_value
    return normalized_feature


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, trainRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        # test_size=testRatio,
                                                        train_size=trainRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test


def create_data_loader(patch_size, train_ratio, pca_components, emap_components, BATCH_SIZE_TRAIN):
    X, y = loadData()
    X_emap = loadEMAPData()
    CLASSES_NUM = int(np.max(y))
    patch_size = patch_size

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X1 = applyPCA(X, numComponents=pca_components)
    X_emap = global_standardization(np.expand_dims(X_emap, axis=2))
    X_pca = np.concatenate([X1, X_emap], axis=2)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... create data cubes ... ...')
    X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, train_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)
    pca_components = pca_components + emap_components

    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    X = TestDS(X, y)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               num_workers=4,
                                               drop_last=True,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=BATCH_SIZE_TRAIN,
                                              shuffle=False,
                                              num_workers=4,
                                              drop_last=True
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                  batch_size=BATCH_SIZE_TRAIN,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  drop_last=True
                                                  )

    return train_loader, test_loader, all_data_loader, y, CLASSES_NUM


class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def train(train_loader, epochs, patch_size, pca_components, emap_components, class_num, cuda):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = cuda

    net = SClusterFormer(img_size=patch_size, pca_components=pca_components,
                                                      emap_components=emap_components,
                                                      num_classes=class_num,
                                                      n_groups=[16, 16, 16], depths=[2, 1, 1], patchsize=patch_size)
    net.to(device)
    net = torch.nn.DataParallel(net, device_ids=device_ids)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    total_loss = 0

    for epoch in range(epochs):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device_ids[0]), target.to(device_ids[0])
            outputs = net(data)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
                                                                         loss.item()))
        if total_loss / (epoch + 1) < 0.415:
            break
    print('Finished Training')

    return net, device


def test(device, net, test_loader):
    count = 0
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def acc_reports(y_test, y_pred_test):
    try:
        target_names = ['1', '2', '3', '4', '5', '6', '7',
                        '8', '9', '10', '11', '12', '13', '14', '15', '16']
        classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    except:
        try:
            target_names = ['1', '2', '3', '4', '5', '6', '7',
                            '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
            classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
        except:
            try:
                target_names = ['1', '2', '3', '4', '5', '6', '7',
                                '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
                classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
            except:
                target_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
                classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)



    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa, confusion, each_acc, aa, kappa


def get_classification_map_labels(y_pred, y):
    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                cls_labels[i][j] = y_pred[k] + 1
                k += 1

    return cls_labels

def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 1101:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 0:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 2:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 3:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 4:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 5:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 6:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 7:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 8:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 9:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 11:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 13:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 14:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 15:
            y[index] = np.array([0, 168, 132]) / 255.
        if item == 16:
            y[index] = np.array([128, 128, 165]) / 255.
        if item == 17:
            y[index] = np.array([168, 128, 128]) / 255.
        if item == 18:
            y[index] = np.array([255, 0, 165]) / 255.
        if item == 19:
            y[index] = np.array([0, 165, 255]) / 255.
        if item == 20:
            y[index] = np.array([128, 255, 0]) / 255.
        if item == 21:
            y[index] = np.array([255, 168, 128]) / 255.

    return y

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

def loop_train_test(hyper_parameter):
    run_times, patch_size, train_ratio, pca_components, emap_components, BATCH_SIZE_TRAIN, epochs, cuda = resolve_dict(hyper_parameter)

    OA = []
    AA = []
    KAPPA = []
    TRAINING_TIME = []
    TESTING_TIME = []
    _, y = loadData()
    CLASSES_NUM = int(np.max(y))
    ELEMENT_ACC = np.zeros((run_times, CLASSES_NUM))

    for run_i in range(0, run_times):
        print('round:', run_i + 1)
        train_loader, test_loader, all_data_loader, y_all, class_num = create_data_loader(patch_size, train_ratio,
                                                                                          pca_components,
                                                                                          emap_components,
                                                                                          BATCH_SIZE_TRAIN)
        print('>' * 10, "Start Training", '<' * 10)
        tic1 = time.perf_counter()
        net, device = train(train_loader=train_loader, epochs=epochs, patch_size=patch_size, pca_components=pca_components, emap_components=emap_components, class_num=class_num, cuda=cuda)

        toc1 = time.perf_counter()
        tic2 = time.perf_counter()
        print('>' * 10, "Start Testing", '<' * 10)
        toc2 = time.perf_counter()
        # 评价指标
        y_pred_test, y_test = test(device, net, test_loader)
        classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
        print('SClusterFormer')
        print('OA: ', oa)
        print('AA: ', aa)
        print('Kappa: ', kappa)

        OA.append(oa)
        AA.append(aa)
        KAPPA.append(kappa)
        TRAINING_TIME.append(toc1 - tic1)
        TESTING_TIME.append(toc2 - tic2)
        ELEMENT_ACC[run_i, :] = each_acc

        print('-------Save the result in mat format--------')

        y_pred, y_new = test(device, net, all_data_loader)
        y_part = np.array(y_all[42560:])
        y_pred = np.append(y_pred,y_part)
        X, y = loadData()
        cls_labels = get_classification_map_labels(y_pred, y)

        x = np.ravel(cls_labels)
        for i in range(len(x)):
            if x[i] == 0:
                x[i] = 23
        x = x[:] - 1

        gt = y.flatten()
        for i in range(len(gt)):
            if gt[i] == 0:
                gt[i] = 23
        gt = gt[:] - 1

        y_list = list_to_colormap(x)
        y_gt = list_to_colormap(gt)

        y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
        gt_re = np.reshape(y_gt, (y.shape[0],y.shape[1], 3))
        classification_map(y_re, y, 300,
                           # 'classification_maps/' + 'HH-SClusterFormer_0.005.eps')
                           # 'classification_maps/' + 'HC-SClusterFormer_0.005.eps')
                           'classification_maps/' + 'PU-SClusterFormer_0.05.eps')
        classification_map(y_re, y, 300,
                           # 'classification_maps/'+'HH-SClusterFormer_0.005.png')
                           # 'classification_maps/'+'HC-SClusterFormer_0.005.png')
                           'classification_maps/'+'PU-SClusterFormer_0.05.png')
        classification_map(gt_re, y, 300,
                           # 'classification_maps/' + 'HH_gt.png')
                           # 'classification_maps/' + 'HC_gt.png')
                           'classification_maps/' + 'PU_gt.png')
        print('------Get classification maps successful-------')

    print_results(class_num, np.array(OA), np.array(AA), np.array(KAPPA), np.array(ELEMENT_ACC),
                  np.array(TRAINING_TIME), np.array(TESTING_TIME))
