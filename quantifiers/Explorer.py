import torch

class Explorer:
    def __init__(self):
        self.history = []  # Each entry: (B, O, D)

    '''
    Returns: A tuple of two (b,) tensors
    	Step distance: How different this tensor is to the last one inputted
    	Novelty: How different this tensor is to all tensors its seen before
    	On first call returns all zeros
    '''
    def __call__(self, x):
        x = x.detach() # (B, O, D)
        B, O, D = x.shape

        # Step Distance
        if len(self.history) == 0:
            step_dist = torch.zeros(B, device=x.device)
        else:
            prev = self.history[-1] # (B, O, D)
            step_dist = torch.norm(x - prev, dim=(1, 2)) # Frobenius Norm

        # Novelty
        if len(self.history) == 0:
            novelty = torch.zeros(B, device=x.device)
        else:
            novelty = torch.full((B,), float('inf'), device=x.device) # (B,)

            # Compare each index b against its own past values
            for past in self.history:
                diff = x - past # (B, O, D)
                dists = torch.norm(diff, dim=(1, 2)) # (B,)
                novelty = torch.minimum(novelty, dists)

        self.history.append(x)
        return step_dist, novelty
