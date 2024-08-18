import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(8, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(10), [str(i) for i in range(10)])
    plt.yticks(np.arange(10), [str(i) for i in range(10)])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = confusion_matrix.max() / 2
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(confusion_matrix[i, j]), horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()

def calculate_metrics(confusion_matrix):
    precision = np.zeros(10)
    recall = np.zeros(10)
    f1_score = np.zeros(10)

    for i in range(10):
        TP = confusion_matrix[i, i]
        FP = confusion_matrix[:, i].sum() - TP
        FN = confusion_matrix[i, :].sum() - TP

        precision[i] = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall[i] = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

    return precision, recall, f1_score

def plot_metrics(precision, recall, f1_score):
    classes = np.arange(10)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.bar(classes, precision, color='blue')
    plt.ylim(0, 1)
    plt.title('Precision per Class')
    plt.xlabel('Class')
    plt.ylabel('Precision')

    plt.subplot(1, 3, 2)
    plt.bar(classes, recall, color='green')
    plt.ylim(0, 1)
    plt.title('Recall per Class')
    plt.xlabel('Class')
    plt.ylabel('Recall')

    plt.subplot(1, 3, 3)
    plt.bar(classes, f1_score, color='red')
    plt.ylim(0, 1)
    plt.title('F1-Score per Class')
    plt.xlabel('Class')
    plt.ylabel('F1-Score')

    plt.tight_layout()
    plt.show()

