def run_epoch(data_iter, model, loss_compute, epoch_num, is_train=True):
    """
    Standard Training and Logging Function
    """
    # TODO: Implement the training loop for one epoch, including loss calculation and logging to W&B
    pass

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    Task 3.3: Implement Greedy Decoding function for inference
    """
    # TODO: Implement token-by-token greedy decoding to get predicted translation.
    raise NotImplementedError


def run_training_experiment():
    """ Setup for W&B and base variables """
    # TODO: Initialize W&B, set up dataset, dataloader, model, optimizer, loss function, and learning rate scheduler
    pass

def evaluate_bleu(model, test_dataloader, tgt_vocab):
    """ Evaluate translations using python bleu metric """
    # TODO: loop test set, greedily decode each sequence and calculate BLEU score
    pass

if __name__ == "__main__":
    # Call run_training_experiment() and train the baseline architecture
    pass