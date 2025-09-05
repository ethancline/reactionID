import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


import reactionModel

def plot_and_test_model_BCE(model, losses=None, valid_losses=None, num_epochs=2500, device=None, straws=None, truth=None, lr=None):
    _,ax = plt.subplots(ncols=2, nrows=2, figsize=(10,10),constrained_layout=True)
    if losses is not None:
        ax[0,0].plot(range(0,num_epochs),losses,color='r',label='Training Loss Curve')
        a = None
        print(f'Minimum training loss: {min(losses):.5f}')
        if valid_losses is not None:
            ax[0,0].plot(range(0,num_epochs),valid_losses,color='b',label='Validation Loss Curve')
            print(f'Minimum validation loss: {min(valid_losses):.5f}')
        if lr is not None:
            a = ax[0,0].twinx()
            a.set_ylabel('Learning Rate',color='g')
            a.plot(range(0,num_epochs),lr,color='g',label='Learning Rate', alpha=0.5, linestyle='--')
            a.tick_params(axis='y', labelcolor='g')
            a.yaxis.set_major_formatter('{:.3g}'.format)
        if a is not None:
            lines, labels = ax[0,0].get_legend_handles_labels()
            lines2, labels2 = a.get_legend_handles_labels()
            ax[0,0].legend(lines + lines2, labels + labels2,fontsize=7.5)
        else:
            ax[0,0].legend()

    ax[0,0].set_title('Loss vs Epoch')
    ax[0,0].set_xlabel('Epoch')
    ax[0,0].set_ylabel('Loss')
    ax[0,0].set_yscale('log')
    model.eval()
    input_data = reactionModel.inputLineData(data_values=torch.tensor(straws, dtype=torch.float32), line_parameters=torch.tensor(truth, dtype=torch.float32))
    test_loader = DataLoader(input_data, batch_size=256, shuffle=True) # Doesn't matter for validation besides batch size
    predicted, truth = [], []
    raw_output, sigmoid_output = [], []
    hitradii = []
    print("Validating model...")
    with torch.no_grad():
        tot_loss = 0.0
        tot_samples = 0
        for batch_data, batch_values in test_loader:
            batch_data, batch_values = batch_data.to(device), batch_values.to(device)
            output = model(batch_data)
            batch_size = batch_values.size(0)
            loss = nn.BCEWithLogitsLoss()(output, batch_values.unsqueeze(1)).item() * batch_size
            tot_loss += loss
            tot_samples += batch_size
            batch_values = torch.flatten(batch_values)
            mask = (batch_values != -1)
            output = torch.flatten(output)
            hitradii.append(batch_data[:, 1].reshape(batch_size, 1).flatten()[mask])
            raw_output.append(output[mask])
            output = torch.sigmoid(output)  
            sigmoid_output.append(output[mask])
            output = (output > 0.5).float() 
            predicted.append(output[mask])
            truth.append(batch_values[mask])
        print(f'Average Loss: {tot_loss/tot_samples:.5f}')
    predicted, truth = torch.cat(predicted).cpu().numpy(), torch.cat(truth).cpu().numpy()
    raw_output, sigmoid_output = torch.cat(raw_output).cpu().numpy(), torch.cat(sigmoid_output).cpu().numpy()
    hitradii = torch.cat(hitradii).cpu().numpy()

    correct_mask = truth == predicted

    same_side = np.where((correct_mask), 1, 0)
    print(f'Percent on the correct side: {np.sum(same_side)/len(same_side)*100.:.3f}')

    h = ax[0,1].hist2d(truth, predicted, bins=2, cmin=1)
    ax[0,1].set_title(f'Truth vs Predicted Sides')
    ax[0,1].set_xlabel('Truth Side')
    ax[0,1].set_ylabel('Predicted Side')
    plt.colorbar(h[3], ax=ax[0,1], label='Entries')

    ax[1,0].hist(truth, bins=3, label='Truth', alpha=0.5, histtype='step')
    ax[1,0].hist(predicted, bins=3, label='Predicted', alpha=0.5, histtype='step')
    ax[1,0].set_title('Truth and Predicted Distributions')
    ax[1,0].set_xlabel('Value')
    ax[1,0].set_ylabel('Counts')
    ax[1,0].set_xticks([0.25, 0.75])
    ax[1,0].set_xticklabels(['Right', 'Left'])
    ax[1,0].legend()

    ax[1,1].hist(raw_output, bins=50, label=fr'$\mu={np.mean(raw_output):.3f}$'+'\n'+fr'$\sigma={np.std(raw_output):.3f}$')
    ax[1,1].set_title('Logit Distribution')
    ax[1,1].set_xlabel('Logit Value')
    ax[1,1].set_ylabel('Counts')
    ax[1,1].xaxis.set_major_formatter('{:.3g}'.format)
    ax[1,1].legend()

    plt.savefig("bce_loss_residuals_10.pdf", bbox_inches="tight", dpi=300)

    _, ax2 = plt.subplots(ncols=2, nrows=2, figsize=(10,10), constrained_layout=True)

    ax2[0,0].hist(sigmoid_output, bins=30, label=fr'$\mu={np.mean(sigmoid_output):.3f}$'+'\n'+fr'$\sigma={np.std(sigmoid_output):.3f}$')
    ax2[0,0].set_title('Sigmoid Distribution')
    ax2[0,0].set_xlabel('Sigmoid Value')
    ax2[0,0].set_ylabel('Counts')
    ax2[0,0].xaxis.set_major_formatter('{:.3g}'.format)
    ax2[0,0].legend()

    ax2[0,1].hist(truth - predicted, bins=3, label=fr'$\mu={np.mean(truth - predicted):.3f}$'+'\n'+fr'$\sigma={np.std(truth - predicted):.3f}$')
    ax2[0,1].set_title('Residuals')
    ax2[0,1].set_xlabel('Truth - Predicted')
    ax2[0,1].set_ylabel('Counts')
    ax2[0,1].xaxis.set_major_formatter('{:.3g}'.format)
    ax2[0,1].legend()

    fpr, tpr, _ = roc_curve(truth, sigmoid_output)
    roc_auc = auc(fpr, tpr)
    
    ax2[1,0].plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
    ax2[1,0].fill_between(fpr, tpr, alpha=0.1)
    ax2[1,0].plot([0, 1], [0, 1], 'k--',alpha=0.5,label='Random Guessing')
    x = np.array([0, 0, 1])
    y = np.array([0, 1, 1])
    ax2[1,0].plot(x, y,color='r',alpha=0.5, label='Perfect')
    ax2[1,0].set_title('ROC Curve')
    ax2[1,0].set_xlabel('False Positive Rate')
    ax2[1,0].set_ylabel('True Positive Rate')
    ax2[1,0].legend(loc='lower right')

    precision, recall, _ = precision_recall_curve(truth, sigmoid_output)
    avg_precision = average_precision_score(truth, sigmoid_output)
    ax2[1,1].plot(recall, precision, label=f'Avg Precision = {avg_precision:.3f}')
    ax2[1,1].fill_between(recall, precision, alpha=0.1)
    ax2[1,1].plot([0,0.5],[1,0.5], 'k--', alpha=0.5, label='Random Guessing')
    x = np.array([0, 1, 1])
    y = np.array([1, 1, 0.5])
    ax2[1,1].plot(x, y, color='r', alpha=0.5, label='Perfect')
    ax2[1,1].set_title('Precision-Recall Curve')
    ax2[1,1].set_xlabel('Recall')
    ax2[1,1].set_ylabel('Precision')
    ax2[1,1].legend()

    plt.savefig("bce_sigmoid_10.pdf", bbox_inches="tight", dpi=300)

    _, ax3 = plt.subplots(ncols=2, nrows=2, figsize=(10,10), constrained_layout=True)

    n_xbins = 30
    x_bins = np.linspace(np.min(raw_output), np.max(raw_output), n_xbins)
    y_bins = [-0.5, 0.5, 1.5] 
    h = ax3[0,0].hist2d(raw_output, (correct_mask).astype(int), bins=[x_bins, y_bins], cmap='viridis', cmin=1)
    ax3[0,0].set_title('Logit vs Correct Prediction')
    ax3[0,0].set_xlabel('Logit Value')
    ax3[0,0].set_ylabel('Correct Prediction')
    ax3[0,0].set_yticks([0,1])
    ax3[0,0].set_yticklabels(['False', 'True'])
    plt.colorbar(h[3], ax=ax3[0,0], label='Counts')

    bad_indices, good_indices = np.where(truth != predicted)[0], np.where(correct_mask)[0]
    ax3[0,1].hist(raw_output[bad_indices], bins=30, label='Wrong Predictions', alpha=0.5, histtype='step')
    ax3[0,1].hist(raw_output[good_indices], bins=30, label='Correct Predictions', alpha=0.5, histtype='step')
    ax3[0,1].axvline(x=-2.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax3[0,1].axvline(x=2.5, color='red', linestyle='--', alpha=0.5)
    ax3[0,1].set_title('Logit Distribution by Prediction')
    ax3[0,1].set_xlabel('Logit Value')
    ax3[0,1].set_ylabel('Counts')
    ax3[0,1].legend()
    
    ax3[1,0].hist(hitradii, bins=30, label="Hit Radii", alpha=0.5, histtype='step')
    ax3[1,0].hist(hitradii[bad_indices], bins=30, label='Wrong Predictions', alpha=0.5, histtype='step')
    ax3[1,0].hist(hitradii[good_indices], bins=30, label='Correct Predictions', alpha=0.5, histtype='step')
    ax3[1,0].set_title('Hit Radius Distribution by Prediction')
    ax3[1,0].set_xlabel('Hit Radius (mm)')
    ax3[1,0].set_ylabel('Counts')
    ax3[1,0].legend()

    h = ax3[1,1].hist2d(hitradii, raw_output, bins=30, cmap='viridis', cmin=1)
    ax3[1,1].set_title('Logit vs Hit Radius')
    ax3[1,1].set_xlabel('Hit Radius (mm)')
    ax3[1,1].set_ylabel('Logit Value')
    ax3[1,1].xaxis.set_major_formatter('{:.3g}'.format)
    plt.colorbar(h[3], ax=ax3[1,1], label='Counts')

    plt.savefig("bce_logits_10.pdf", bbox_inches="tight", dpi=300)

    # _, ax4 = plt.subplots(ncols=2, nrows=2, figsize=(10,10), constrained_layout=True)

    # lessThan1mm = np.where(hitradii < 1.0)[0]
    # moreThan1mm = np.where(hitradii >= 1.0)[0]
    # lessThan25mm = np.where(hitradii < 2.5)[0]
    # moreThan25mm = np.where(hitradii >= 2.5)[0]

    # moreThan25logit = np.where(np.abs(raw_output) >= 2.5)[0]

    # ax4[0,0].hist(raw_output[lessThan1mm], bins=30, label='Hit Radius < 1mm', alpha=0.5, histtype='step')
    # ax4[0,0].hist(raw_output[moreThan1mm], bins=30, label='Hit Radius >= 1mm', alpha=0.5, histtype='step')
    # ax4[0,0].axvline(x=-2.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
    # ax4[0,0].axvline(x=2.5, color='red', linestyle='--', alpha=0.5)
    # ax4[0,0].set_title('Logit Distribution by Hit Radius')
    # ax4[0,0].set_xlabel('Logit Value')
    # ax4[0,0].set_ylabel('Counts')
    # ax4[0,0].legend()

    # h = ax4[0,1].hist2d(raw_output[lessThan1mm], (truth[lessThan1mm] == predicted[lessThan1mm]).astype(int), bins=[x_bins, y_bins], cmap='viridis', cmin=1)
    # ax4[0,1].set_title('Logit vs Correct Prediction (Hit Radius < 1mm)')
    # ax4[0,1].set_xlabel('Logit Value')
    # ax4[0,1].set_ylabel('Correct Prediction')
    # ax4[0,1].set_yticks([0,1])
    # ax4[0,1].set_yticklabels(['False', 'True'])
    # plt.colorbar(h[3], ax=ax4[0,1], label='Counts')

    # h = ax4[1,0].hist2d(raw_output[moreThan1mm], (truth[moreThan1mm] == predicted[moreThan1mm]).astype(int), bins=[x_bins, y_bins], cmap='viridis', cmin=1)
    # ax4[1,0].set_title('Logit vs Correct Prediction (Hit Radius >= 1mm)')
    # ax4[1,0].set_xlabel('Logit Value')
    # ax4[1,0].set_ylabel('Correct Prediction')
    # ax4[1,0].set_yticks([0,1])
    # ax4[1,0].set_yticklabels(['False', 'True'])
    # plt.colorbar(h[3], ax=ax4[1,0], label='Counts')

    # h = ax4[1,1].hist2d(raw_output[moreThan25mm], (truth[moreThan25mm] == predicted[moreThan25mm]).astype(int), bins=[x_bins, y_bins], cmap='viridis', cmin=1)
    # ax4[1,1].set_title('Logit vs Hit Radius (Hit Radius >= 2.5mm)')
    # ax4[1,1].set_xlabel('Logit Value')
    # ax4[1,1].set_ylabel('Correct Prediction')  
    # ax4[1,1].set_yticks([0,1])
    # ax4[1,1].set_yticklabels(['False', 'True'])
    # plt.colorbar(h[3], ax=ax4[1,1], label='Counts')

    # plt.savefig('bce_hit_radii_10.pdf', bbox_inches='tight', dpi=300)

    # print(f'Percent correct for >= 1mm: {np.sum((truth[moreThan1mm] == predicted[moreThan1mm]).astype(int))/len(moreThan1mm)*100:.3f}')
    # print(f'Percent correct for < 1mm: {np.sum((truth[lessThan1mm] == predicted[lessThan1mm]).astype(int))/len(lessThan1mm)*100:.3f}')
    # print(f'Percent correct for >= 2.5mm: {np.sum((truth[moreThan25mm] == predicted[moreThan25mm]).astype(int))/len(moreThan25mm)*100:.3f}')
    # print(f'Percent correct for < 2.5mm: {np.sum((truth[lessThan25mm] == predicted[lessThan25mm]).astype(int))/len(lessThan25mm)*100:.3f}')

    # percentOver5logit = np.sum(np.abs(raw_output) > 5.0) / len(raw_output) * 100
    # print(f'Percent of logits with abs value > 5: {percentOver5logit:.3f}')
    # percentOver5logitCorrect = np.sum(np.abs(raw_output[correct_mask]) > 5.0) / np.sum(np.abs(raw_output) > 5.0) * 100
    # print(f'Percent logits with abs value > 5 on correct side: {percentOver5logitCorrect:.3f}')

    # over25_mask = np.abs(raw_output) > 2.5
    # moreThan1mm_mask = hitradii >= 1.0
    
    # percentOver25logit = np.sum(over25_mask & moreThan1mm_mask) / len(raw_output) * 100
    # print(f'Percent of logits with abs value > 2.5 and radius >= 1 mm: {percentOver25logit:.3f}')
    # correct_and_over25_mask = over25_mask & moreThan1mm_mask & correct_mask
    # percentOver25logitCorrect = np.sum(correct_and_over25_mask) / np.sum(over25_mask & moreThan1mm_mask) * 100
    # print(f'Percent logits with abs value > 2.5 and radius >= 1 mm on correct side: {percentOver25logitCorrect:.3f}')

    # plt.tight_layout()
    plt.show()