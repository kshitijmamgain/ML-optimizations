import numpy as np
from sklearn.metrics import (auc, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, 
                             precision_recall_curve, classification_report)

import matplotlib.pyplot as plt

class Model_Evaluation():
    """
    Class to run evalutaions and assess model performances

    """
    def __init__(self):
        """
        place holder for the test_y and pred to be initialized later on 

        """
        self.test_y = None
        self.pred_y = None
        self.threshold = None
        

    def set_label_scores(self, pred_y, test_y, threshold=0.5 ):
        """
        Sets the train and test data
        Parameters:

        pred: <np array><float>  Prediction probabilities
        test_y: <np array>
        threshold: <float default as 0.5> Threshold to decide classes

        """
        self.pred_y = pred_y
        self.test_y = test_y
        self.threshold = threshold


    def get_metrics(self,
                    roc_filename,
                    pr_filename,
                    fpr_fnr_filename,
                    algo="Catboost"):
        
        """This function generates the evaluation report for the model

        roc_filename <str>: Name of roc_file
        pr_filename <str>: name of the pr_filename
        fpr_fnr_filename <str> name of the fpr_fnr filename
        threshold <float> value used for the cutpoints to define classes 
        algo: <str> Algorithm which is being tested
        
        returns:
        metric_results <dict> dictionary of results

        """
        metric_results = dict()
        precision, recall, _ = precision_recall_curve(self.test_y, self.pred_y)
        pred_y_bin = np.where(self.pred_y > self.threshold, 1, 0)

        metric_results['algo'] = algo

        # do the calculations
        metric_results['pr-auc'] = auc(recall, precision)
        metric_results['class_report'] = classification_report(self.test_y, pred_y_bin)
        metric_results['conf_metrics'] = confusion_matrix(self.test_y, pred_y_bin)
        metric_results['roc_auc'] = roc_auc_score(self.test_y, self.pred_y)


        # start the graphs
        metric_results['roc_curve_path'] = \
        roc_filename if self.roc(roc_filename, metric_results['roc_auc']) else "not generated"
        
        metric_results['pr_curve_path'] = \
        pr_filename if self.prcurve(pr_filename, metric_results['pr-auc']) else "not generated"
        
        metric_results['fpr_fnr_path'] = \
        fpr_fnr_filename if self.fpr_fnr(fpr_fnr_filename) else "not generated"

        return metric_results



    def roc(self, filename, roc_auc):
        """calculates and draw the ROC curve 
        
        filename <str> filename save the figure
        roc_auc <float> AUC value
        
        Returns True if the figure is saved
        """
        
        fpr, tpr, _  = roc_curve(y_true=self.test_y, y_score=self.pred_y)
        plt.figure(figsize=(16, 8))

        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' %roc_auc, alpha=0.5)

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title('Receiver operating characteristic', fontsize=20)
        plt.legend(loc="lower right", fontsize=16)
        plt.savefig(filename)

        return True
        
    def prcurve(self, filename, pr_auc):
        """
        output the precision recall curve for an instance
        filename <str> filename save the figure
        pr_auc <float> AUC value
        
        Returns True if the figure is saved
        """
        
        precision, recall, _ = precision_recall_curve(self.test_y, self.pred_y)

        # plot the precision-recall curves
        no_skill = len(self.test_y[self.test_y==1]) / len(self.test_y)
        plt.figure(figsize = (16,8))
        plt.plot([0, 1], [no_skill, no_skill], color='navy', linestyle='--',
                 alpha=0.5)
        plt.plot(recall, precision, color='darkorange',
                 label='ROC curve (area = %0.2f)'% pr_auc, alpha=0.5)
        # axis labels
        plt.title('Precision Recall Curve', size = 20)
        plt.xlabel('Recall', fontsize=16)
        plt.ylabel('Precision', fontsize=16)
        plt.grid(True)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # show the legend
        plt.legend(fontsize=16)
        plt.savefig(filename)

        return True



    def fpr_fnr(self, filename):
        """
        A class method to output the fpr_fnr curve for an instance
        filename <str> filename save the figure
        
        pr_auc <float> AUC value
        
        Returns True if the figure is saved
        
        """
        fpr, tpr, _  = roc_curve(y_true=self.test_y, y_score=self.pred_y)
        plt.figure(figsize = (16,8))
        plt.plot(_, fpr, color='blue', lw=2, label='FPR', alpha=0.5)
        plt.plot(_, 1-tpr, color='green', lw=2, label='FNR', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.xlabel('Threshold', fontsize=16)
        plt.ylabel('Error Rate', fontsize=16)
        plt.title('FPR-FNR curves', fontsize=20)
        plt.legend(loc="lower left", fontsize=16)
        plt.savefig(filename)
        return True