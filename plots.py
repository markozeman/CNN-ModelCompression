import matplotlib.pyplot as plt


def show_grayscale_image(img):
    """
    Show grayscale image (1 channel).

    :param img: image to plot
    :return: None
    """
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


def show_image(img):
    """
    Show coloured image (3 channels).

    :param img: image to plot
    :return: None
    """
    plt.imshow(img)
    plt.show()


def plot_fit_history(history):
    """
    Plot changes of loss and accuracy during training.

    :param history: History object
    :return: None
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model accuracy and loss during training')
    plt.ylabel('Accuracy / Loss')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy (train)', 'Accuracy (test)', 'Loss (train)', 'Loss (test)'])
    plt.show()


