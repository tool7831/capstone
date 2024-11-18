class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=False):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement before stopping.
            min_delta (float): Minimum improvement to qualify as a new best.
            verbose (bool): Whether to print information about early stopping.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print("Initial best score set.")

        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print("Improvement detected, resetting counter.")