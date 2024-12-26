def dice_score(pred_mask, gt_mask):

    common = (pred_mask == gt_mask).sum()  

    pred_mask_size = pred_mask.size()[0]*pred_mask.size()[1]*pred_mask.size()[2]*pred_mask.size()[3]
    gt_mask_size = gt_mask.size()[0]*gt_mask.size()[1]*gt_mask.size()[2]*gt_mask.size()[3]
    
    accuracy = (2*common/(pred_mask_size+gt_mask_size)).item()
    
    return accuracy