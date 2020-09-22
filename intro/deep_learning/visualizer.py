from matplotlib import pyplot as plt

def learning_plot(history, epochs):
    fig = plt.figure(figsize=(15,5))
    # Lossの可視化
    plt.subplot(1,2,1)
    plt.plot(range(1,epochs+1), history.history['loss'])
    plt.plot(range(1,epochs+1), history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.xticks(range(1,epochs+1))
    plt.ylabel('loss')
    plt.legend(['train', 'val'], loc='upper right')
    
    # 正解率(accuracy)の可視化
    plt.subplot(1,2,2)
    plt.plot(range(1,epochs+1), history.history['acc'])
    plt.plot(range(1,epochs+1), history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.xticks(range(1,epochs+1))
    plt.ylabel('accuracy')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()