from sklearn import metrics
import torch
import numpy as np
import random

def validation(model, dev_loader, criterion, loss, device):

    model.eval()
    loss_total = 0

    # Calculate Metrics         
    labels_all = torch.empty((0))
    predicted_all = torch.empty((0))
    ids_all_v = torch.empty((0))

    # Validation data
    with torch.no_grad():     
      # Iterate through validation dataset
        for batch in dev_loader:
            for m in range(len(batch)):
              batch[m] = batch[m].to(device)
            
            batch_inputs = batch[:-2]
            point_ids_v = batch[-2]
            labels_v = batch[-1]
            # Forward pass only to get logits/output
            outputs = model(*batch_inputs)

            # Get predictions from the maximum value
            if loss == 'cross_entropy':
                outputs_softmax = torch.log_softmax(outputs, dim = 1)
                _, predicted_v = torch.max(outputs_softmax, dim = 1) 
            else:
                _, predicted_v = torch.max(outputs.data, 1)
                
            loss_v = criterion(outputs, labels_v)

            loss_v = loss_v.cpu().detach()
            labels_v = labels_v.cpu().detach()
            point_ids_v = point_ids_v.cpu().detach()
            predicted_v = predicted_v.cpu().detach()
            loss_total += loss_v.item()

            # Total correct predictions
            labels_all = torch.cat([labels_all, labels_v])
            predicted_all = torch.cat([predicted_all, predicted_v])
            ids_all_v = torch.cat([ids_all_v, point_ids_v])
            del batch, batch_inputs

        f1_v = 100 * metrics.f1_score(labels_all, predicted_all, average = 'macro')

        return f1_v, labels_all, predicted_all, ids_all_v, loss_total


def train(model, train_loader, criterion, optimizer, epoch, num_epochs, gradient_acc, device):
    model.train()
    loss_total_train=0
    
    for iter, batch in enumerate(train_loader):
        for m in range(len(batch)):
          batch[m] = batch[m].to(device)
        batch_inputs = batch[:-2]
        point_ids = batch[-2]
        labels = batch[-1]
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        # Forward pass to get output/logits
        outputs = model(*batch_inputs)
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        #apply gradient accumulation if needed
        loss = loss/gradient_acc
        # Getting gradients w.r.t. parameters
        loss.backward()
        # Updating parameters
        if (iter + 1) % gradient_acc == 0:
          optimizer.step()
        #add up the loss
        loss = loss.cpu().detach()
        loss_total_train += loss.item() * gradient_acc
        # Show training progress
        if (iter % 800 == 0):
          print('[{}/{}, {}/{}] loss: {:.8}'.format(epoch, num_epochs, iter, len(train_loader), loss.item()))
        del batch, batch_inputs, outputs

    return loss_total_train

def test(model, test_loader, loss, device):
      model.eval()

      labels_all = torch.empty((0))
      predicted_all = torch.empty((0))
      probs_all = torch.empty((0))
      ids_all_t = torch.empty((0))

      #Test data
      with torch.no_grad():     
            # Iterate through test dataset
            for batch in test_loader:
                  for m in range(len(batch)):
                    batch[m] = batch[m].to(device)
                  batch_inputs = batch[:-2]
                  point_ids_t = batch[-2]
                  labels_t = batch[-1]
                  # Forward pass only to get logits/output
                  outputs_t = model(*batch_inputs)
                  
                  # Get predictions from the maximum value
                  if (loss == 'cross_entropy'):
                    outputs_t_softmax = torch.log_softmax(outputs_t, dim = 1)
                    _, predicted_t = torch.max(outputs_t_softmax, dim = 1) 
                  else:
                    outputs_t_softmax = outputs_t.data
                    _, predicted_t = torch.max(outputs_t_softmax, 1)

                  # Total correct predictions
                  predicted_t = predicted_t.cpu().detach()
                  labels_t = labels_t.cpu().detach()
                  probs_t = outputs_t_softmax.cpu().detach()
                  point_ids_t = point_ids_t.cpu().detach()
                  labels_all = torch.cat([labels_all, labels_t])
                  predicted_all = torch.cat([predicted_all, predicted_t])
                  probs_all = torch.cat([probs_all, probs_t])
                  ids_all_t = torch.cat([ids_all_t, point_ids_t])
                  del batch

      f1_t = 100 * metrics.f1_score(labels_all, predicted_all, average = 'macro')

      return f1_t, labels_all, predicted_all, probs_all, ids_all_t

def set_seed(seed):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)