import time

# Configurations
import model_settings
model_settings.args.lr = 0.001
epochs_num = 3000

# Training ComparE
from train import train_model_Com, test1
# Train ComParE model
t_total = time.time()
for epoch in range(epochs_num):
    train_model_Com(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
# Testing
test1()



# Configuration
import model_settings
model_settings.args.hidden1 = 64
model_settings.args.hidden2 = 32
model_settings.args.hidden3 = 8
model_settings.args.lr = 0.0005
epochs_num = 10000
from train import train_model_phon, test2

# Train Phonation model
t_total = time.time()
for epoch in range(epochs_num):
    train_model_phon(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test1(), test2()

# In[2]
# Draw Confusion matrix
from utils import plot_confusion_matrix
from data_preprocessing import label_matrix, features, features_phon, adj,labels, idx_test
from train import model, model_phon
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

# Calculate sensitivity, specificity, fi
def get_model_metrics(y_true, y_pred):
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 计算灵敏度和特异性
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    # 计算 F1 分数
    f1 = f1_score(y_true, y_pred)
    
    return sensitivity, specificity, f1


y_true = label_matrix[idx_test]

plt.subplot(1,2,1)
output_Com = model(features, adj)
y_pred = output_Com[idx_test].max(1)[1].type_as(labels)
plot_confusion_matrix(y_true= y_true, y_pred= y_pred)
sensitivity, specificity, f1 = get_model_metrics(y_true, y_pred)
print('Model ComparE')
print('sensitivity: {:04f}'.format(sensitivity),
      'specificity: {:.4f}'.format(specificity),
      'F1 Score: {:.4f}'.format(f1)               )


plt.subplot(1,2,2)
output_phon = model_phon(features_phon, adj)
y_pred = output_phon[idx_test].max(1)[1].type_as(labels)
plot_confusion_matrix(y_true= y_true, y_pred= y_pred)
sensitivity, specificity, f1 = get_model_metrics(y_true, y_pred)
print('Model Phonation')
print('sensitivity: {:04f}'.format(sensitivity),
      'specificity: {:.4f}'.format(specificity),
      'F1 Score: {:.4f}'.format(f1)               )

plt.show()
