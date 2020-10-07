# Pixel-wise Triplet Learning for Accurate Boundary Discrimination in Medical Image Segmentation
Codes and demo of PTL for CVPR 2021 Review.

# How to use
The PTL loss can be embbed into any segmentation network as an additional learning branch.
## Definition
emb_size : the hidden size of final segmentation features

batch_size : the size of mini-batch

H : height of final segmentation output

W : width of final segmentation output

## Input
mask // shape=(batch_size, H, W) // the ground truth, it always have only two classes of foreground (1) or background (0)

probs_fg // shape=(batch_size, 1, H, W) // the prediction probability of foreground pixels

probs_bg // shape=(batch_size, 1, H, W) // the prediction probability of background pixels

embs // shape=(batch_size, emb_size, H, W) // the final segmentation features

loss_func_triplet = torch.nn.TripletMarginLoss(margin=10.0) // the pre-defined triplet loss

## Usage
```
Dice_loss = Dice_loss(mask, probs_fg)

PTL_loss_fg = get_triplet_loss(mask, probs_fg, embs, loss_func_triplet) // PTL for foreground

PTL_loss_bg = get_triplet_loss(mask, probs_bg, embs, loss_func_triplet) // PTL for background

final_loss = Dice_loss + 0.005 * (PTL_loss_fg + PTL_loss_bg)

final_loss.backward()
```
