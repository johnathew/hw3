import matplotlib.pyplot as plt
import numpy as np

testing_loss = [1.045, 1.009, 1.22, 1.22, 0.930, 0.952, 1.13, 1.074, 1.275, 0.66]
std_dev_loss = np.std(testing_loss)


testing_accuracy = [0, 66.40, 68.50, 62.21, 60.87, 72.50, 69, 72, 66.42, 71, 73.2]
std_testing_accuracy = np.std(testing_accuracy)
training_accuracy = [0, 46, 34.76, 51.6, 26, 66.41, 36, 99, 65, 70, 48.7]
std_training_accuracy = np.std(training_accuracy)

ci = 1.96 * std_dev_loss / np.sqrt(len(testing_loss))
ci_testing_accuracy = 1.96 * std_testing_accuracy / np.sqrt(len(testing_accuracy))
ci_training_accuracy = 1.96 * std_training_accuracy / np.sqrt(len(training_accuracy))

testing_accuracy.sort()
training_accuracy.sort()
testing_loss.sort(reverse=True)


plt.plot( testing_accuracy, label='Testing Accuracy', color='blue')
plt.fill_between(np.arange(0, len(testing_accuracy)), np.array(testing_accuracy) - ci_testing_accuracy, np.array(testing_accuracy) + ci_testing_accuracy, color='blue', alpha=0.2)
plt.fill_between(np.arange(0, len(training_accuracy)), np.array(training_accuracy) - ci_training_accuracy, np.array(training_accuracy) + ci_training_accuracy, color='red', alpha=0.2)
plt.plot(training_accuracy, label='Training Accuracy', color='red')
plt.xlabel('Experiment')
plt.ylabel('Value')
plt.title('Loss and Accuracy')
plt.legend()
plt.show()  

plt.plot(testing_loss, label='Testing Loss', color='green')
plt.fill_between(np.arange(0, len(testing_loss)), np.array(testing_loss) - ci, np.array(testing_loss) + ci, color='green', alpha=0.2)
plt.xlabel('Experiment')
plt.ylabel('Value')
plt.title('Loss')
plt.legend()
plt.show()  