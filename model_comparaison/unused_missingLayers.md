This is what the shell returns when executing the script with
`models/bobsl_i3d.pth.tar` or `models/bsl1k_i3d.pth.tar`.

```shell
=> loading checkpoint '../models/bsl1k_i3d.pth.tar'
Unused from pretrain module.logits.conv3d.weight, pretrain: torch.Size([1064, 1024, 1, 1, 1])
Unused from pretrain module.logits.conv3d.bias, pretrain: torch.Size([1064])
Unused from pretrain module.Conv3d_1a_7x7.conv3d.weight, pretrain: torch.Size([64, 3, 7, 7, 7])
Unused from pretrain module.Conv3d_1a_7x7.bn.weight, pretrain: torch.Size([64])
Unused from pretrain module.Conv3d_1a_7x7.bn.bias, pretrain: torch.Size([64])
Unused from pretrain module.Conv3d_1a_7x7.bn.running_mean, pretrain: torch.Size([64])
Unused from pretrain module.Conv3d_1a_7x7.bn.running_var, pretrain: torch.Size([64])
Unused from pretrain module.Conv3d_1a_7x7.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Conv3d_2b_1x1.conv3d.weight, pretrain: torch.Size([64, 64, 1, 1, 1])
Unused from pretrain module.Conv3d_2b_1x1.bn.weight, pretrain: torch.Size([64])
Unused from pretrain module.Conv3d_2b_1x1.bn.bias, pretrain: torch.Size([64])
Unused from pretrain module.Conv3d_2b_1x1.bn.running_mean, pretrain: torch.Size([64])
Unused from pretrain module.Conv3d_2b_1x1.bn.running_var, pretrain: torch.Size([64])
Unused from pretrain module.Conv3d_2b_1x1.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Conv3d_2c_3x3.conv3d.weight, pretrain: torch.Size([192, 64, 3, 3, 3])
Unused from pretrain module.Conv3d_2c_3x3.bn.weight, pretrain: torch.Size([192])
Unused from pretrain module.Conv3d_2c_3x3.bn.bias, pretrain: torch.Size([192])
Unused from pretrain module.Conv3d_2c_3x3.bn.running_mean, pretrain: torch.Size([192])
Unused from pretrain module.Conv3d_2c_3x3.bn.running_var, pretrain: torch.Size([192])
Unused from pretrain module.Conv3d_2c_3x3.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_3b.b0.conv3d.weight, pretrain: torch.Size([64, 192, 1, 1, 1])
Unused from pretrain module.Mixed_3b.b0.bn.weight, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_3b.b0.bn.bias, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_3b.b0.bn.running_mean, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_3b.b0.bn.running_var, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_3b.b0.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_3b.b1a.conv3d.weight, pretrain: torch.Size([96, 192, 1, 1, 1])
Unused from pretrain module.Mixed_3b.b1a.bn.weight, pretrain: torch.Size([96])
Unused from pretrain module.Mixed_3b.b1a.bn.bias, pretrain: torch.Size([96])
Unused from pretrain module.Mixed_3b.b1a.bn.running_mean, pretrain: torch.Size([96])
Unused from pretrain module.Mixed_3b.b1a.bn.running_var, pretrain: torch.Size([96])
Unused from pretrain module.Mixed_3b.b1a.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_3b.b1b.conv3d.weight, pretrain: torch.Size([128, 96, 3, 3, 3])
Unused from pretrain module.Mixed_3b.b1b.bn.weight, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_3b.b1b.bn.bias, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_3b.b1b.bn.running_mean, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_3b.b1b.bn.running_var, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_3b.b1b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_3b.b2a.conv3d.weight, pretrain: torch.Size([16, 192, 1, 1, 1])
Unused from pretrain module.Mixed_3b.b2a.bn.weight, pretrain: torch.Size([16])
Unused from pretrain module.Mixed_3b.b2a.bn.bias, pretrain: torch.Size([16])
Unused from pretrain module.Mixed_3b.b2a.bn.running_mean, pretrain: torch.Size([16])
Unused from pretrain module.Mixed_3b.b2a.bn.running_var, pretrain: torch.Size([16])
Unused from pretrain module.Mixed_3b.b2a.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_3b.b2b.conv3d.weight, pretrain: torch.Size([32, 16, 3, 3, 3])
Unused from pretrain module.Mixed_3b.b2b.bn.weight, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_3b.b2b.bn.bias, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_3b.b2b.bn.running_mean, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_3b.b2b.bn.running_var, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_3b.b2b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_3b.b3b.conv3d.weight, pretrain: torch.Size([32, 192, 1, 1, 1])
Unused from pretrain module.Mixed_3b.b3b.bn.weight, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_3b.b3b.bn.bias, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_3b.b3b.bn.running_mean, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_3b.b3b.bn.running_var, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_3b.b3b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_3c.b0.conv3d.weight, pretrain: torch.Size([128, 256, 1, 1, 1])
Unused from pretrain module.Mixed_3c.b0.bn.weight, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_3c.b0.bn.bias, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_3c.b0.bn.running_mean, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_3c.b0.bn.running_var, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_3c.b0.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_3c.b1a.conv3d.weight, pretrain: torch.Size([128, 256, 1, 1, 1])
Unused from pretrain module.Mixed_3c.b1a.bn.weight, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_3c.b1a.bn.bias, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_3c.b1a.bn.running_mean, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_3c.b1a.bn.running_var, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_3c.b1a.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_3c.b1b.conv3d.weight, pretrain: torch.Size([192, 128, 3, 3, 3])
Unused from pretrain module.Mixed_3c.b1b.bn.weight, pretrain: torch.Size([192])
Unused from pretrain module.Mixed_3c.b1b.bn.bias, pretrain: torch.Size([192])
Unused from pretrain module.Mixed_3c.b1b.bn.running_mean, pretrain: torch.Size([192])
Unused from pretrain module.Mixed_3c.b1b.bn.running_var, pretrain: torch.Size([192])
Unused from pretrain module.Mixed_3c.b1b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_3c.b2a.conv3d.weight, pretrain: torch.Size([32, 256, 1, 1, 1])
Unused from pretrain module.Mixed_3c.b2a.bn.weight, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_3c.b2a.bn.bias, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_3c.b2a.bn.running_mean, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_3c.b2a.bn.running_var, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_3c.b2a.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_3c.b2b.conv3d.weight, pretrain: torch.Size([96, 32, 3, 3, 3])
Unused from pretrain module.Mixed_3c.b2b.bn.weight, pretrain: torch.Size([96])
Unused from pretrain module.Mixed_3c.b2b.bn.bias, pretrain: torch.Size([96])
Unused from pretrain module.Mixed_3c.b2b.bn.running_mean, pretrain: torch.Size([96])
Unused from pretrain module.Mixed_3c.b2b.bn.running_var, pretrain: torch.Size([96])
Unused from pretrain module.Mixed_3c.b2b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_3c.b3b.conv3d.weight, pretrain: torch.Size([64, 256, 1, 1, 1])
Unused from pretrain module.Mixed_3c.b3b.bn.weight, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_3c.b3b.bn.bias, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_3c.b3b.bn.running_mean, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_3c.b3b.bn.running_var, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_3c.b3b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4b.b0.conv3d.weight, pretrain: torch.Size([192, 480, 1, 1, 1])
Unused from pretrain module.Mixed_4b.b0.bn.weight, pretrain: torch.Size([192])
Unused from pretrain module.Mixed_4b.b0.bn.bias, pretrain: torch.Size([192])
Unused from pretrain module.Mixed_4b.b0.bn.running_mean, pretrain: torch.Size([192])
Unused from pretrain module.Mixed_4b.b0.bn.running_var, pretrain: torch.Size([192])
Unused from pretrain module.Mixed_4b.b0.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4b.b1a.conv3d.weight, pretrain: torch.Size([96, 480, 1, 1, 1])
Unused from pretrain module.Mixed_4b.b1a.bn.weight, pretrain: torch.Size([96])
Unused from pretrain module.Mixed_4b.b1a.bn.bias, pretrain: torch.Size([96])
Unused from pretrain module.Mixed_4b.b1a.bn.running_mean, pretrain: torch.Size([96])
Unused from pretrain module.Mixed_4b.b1a.bn.running_var, pretrain: torch.Size([96])
Unused from pretrain module.Mixed_4b.b1a.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4b.b1b.conv3d.weight, pretrain: torch.Size([208, 96, 3, 3, 3])
Unused from pretrain module.Mixed_4b.b1b.bn.weight, pretrain: torch.Size([208])
Unused from pretrain module.Mixed_4b.b1b.bn.bias, pretrain: torch.Size([208])
Unused from pretrain module.Mixed_4b.b1b.bn.running_mean, pretrain: torch.Size([208])
Unused from pretrain module.Mixed_4b.b1b.bn.running_var, pretrain: torch.Size([208])
Unused from pretrain module.Mixed_4b.b1b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4b.b2a.conv3d.weight, pretrain: torch.Size([16, 480, 1, 1, 1])
Unused from pretrain module.Mixed_4b.b2a.bn.weight, pretrain: torch.Size([16])
Unused from pretrain module.Mixed_4b.b2a.bn.bias, pretrain: torch.Size([16])
Unused from pretrain module.Mixed_4b.b2a.bn.running_mean, pretrain: torch.Size([16])
Unused from pretrain module.Mixed_4b.b2a.bn.running_var, pretrain: torch.Size([16])
Unused from pretrain module.Mixed_4b.b2a.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4b.b2b.conv3d.weight, pretrain: torch.Size([48, 16, 3, 3, 3])
Unused from pretrain module.Mixed_4b.b2b.bn.weight, pretrain: torch.Size([48])
Unused from pretrain module.Mixed_4b.b2b.bn.bias, pretrain: torch.Size([48])
Unused from pretrain module.Mixed_4b.b2b.bn.running_mean, pretrain: torch.Size([48])
Unused from pretrain module.Mixed_4b.b2b.bn.running_var, pretrain: torch.Size([48])
Unused from pretrain module.Mixed_4b.b2b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4b.b3b.conv3d.weight, pretrain: torch.Size([64, 480, 1, 1, 1])
Unused from pretrain module.Mixed_4b.b3b.bn.weight, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4b.b3b.bn.bias, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4b.b3b.bn.running_mean, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4b.b3b.bn.running_var, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4b.b3b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4c.b0.conv3d.weight, pretrain: torch.Size([160, 512, 1, 1, 1])
Unused from pretrain module.Mixed_4c.b0.bn.weight, pretrain: torch.Size([160])
Unused from pretrain module.Mixed_4c.b0.bn.bias, pretrain: torch.Size([160])
Unused from pretrain module.Mixed_4c.b0.bn.running_mean, pretrain: torch.Size([160])
Unused from pretrain module.Mixed_4c.b0.bn.running_var, pretrain: torch.Size([160])
Unused from pretrain module.Mixed_4c.b0.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4c.b1a.conv3d.weight, pretrain: torch.Size([112, 512, 1, 1, 1])
Unused from pretrain module.Mixed_4c.b1a.bn.weight, pretrain: torch.Size([112])
Unused from pretrain module.Mixed_4c.b1a.bn.bias, pretrain: torch.Size([112])
Unused from pretrain module.Mixed_4c.b1a.bn.running_mean, pretrain: torch.Size([112])
Unused from pretrain module.Mixed_4c.b1a.bn.running_var, pretrain: torch.Size([112])
Unused from pretrain module.Mixed_4c.b1a.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4c.b1b.conv3d.weight, pretrain: torch.Size([224, 112, 3, 3, 3])
Unused from pretrain module.Mixed_4c.b1b.bn.weight, pretrain: torch.Size([224])
Unused from pretrain module.Mixed_4c.b1b.bn.bias, pretrain: torch.Size([224])
Unused from pretrain module.Mixed_4c.b1b.bn.running_mean, pretrain: torch.Size([224])
Unused from pretrain module.Mixed_4c.b1b.bn.running_var, pretrain: torch.Size([224])
Unused from pretrain module.Mixed_4c.b1b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4c.b2a.conv3d.weight, pretrain: torch.Size([24, 512, 1, 1, 1])
Unused from pretrain module.Mixed_4c.b2a.bn.weight, pretrain: torch.Size([24])
Unused from pretrain module.Mixed_4c.b2a.bn.bias, pretrain: torch.Size([24])
Unused from pretrain module.Mixed_4c.b2a.bn.running_mean, pretrain: torch.Size([24])
Unused from pretrain module.Mixed_4c.b2a.bn.running_var, pretrain: torch.Size([24])
Unused from pretrain module.Mixed_4c.b2a.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4c.b2b.conv3d.weight, pretrain: torch.Size([64, 24, 3, 3, 3])
Unused from pretrain module.Mixed_4c.b2b.bn.weight, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4c.b2b.bn.bias, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4c.b2b.bn.running_mean, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4c.b2b.bn.running_var, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4c.b2b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4c.b3b.conv3d.weight, pretrain: torch.Size([64, 512, 1, 1, 1])
Unused from pretrain module.Mixed_4c.b3b.bn.weight, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4c.b3b.bn.bias, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4c.b3b.bn.running_mean, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4c.b3b.bn.running_var, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4c.b3b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4d.b0.conv3d.weight, pretrain: torch.Size([128, 512, 1, 1, 1])
Unused from pretrain module.Mixed_4d.b0.bn.weight, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_4d.b0.bn.bias, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_4d.b0.bn.running_mean, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_4d.b0.bn.running_var, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_4d.b0.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4d.b1a.conv3d.weight, pretrain: torch.Size([128, 512, 1, 1, 1])
Unused from pretrain module.Mixed_4d.b1a.bn.weight, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_4d.b1a.bn.bias, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_4d.b1a.bn.running_mean, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_4d.b1a.bn.running_var, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_4d.b1a.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4d.b1b.conv3d.weight, pretrain: torch.Size([256, 128, 3, 3, 3])
Unused from pretrain module.Mixed_4d.b1b.bn.weight, pretrain: torch.Size([256])
Unused from pretrain module.Mixed_4d.b1b.bn.bias, pretrain: torch.Size([256])
Unused from pretrain module.Mixed_4d.b1b.bn.running_mean, pretrain: torch.Size([256])
Unused from pretrain module.Mixed_4d.b1b.bn.running_var, pretrain: torch.Size([256])
Unused from pretrain module.Mixed_4d.b1b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4d.b2a.conv3d.weight, pretrain: torch.Size([24, 512, 1, 1, 1])
Unused from pretrain module.Mixed_4d.b2a.bn.weight, pretrain: torch.Size([24])
Unused from pretrain module.Mixed_4d.b2a.bn.bias, pretrain: torch.Size([24])
Unused from pretrain module.Mixed_4d.b2a.bn.running_mean, pretrain: torch.Size([24])
Unused from pretrain module.Mixed_4d.b2a.bn.running_var, pretrain: torch.Size([24])
Unused from pretrain module.Mixed_4d.b2a.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4d.b2b.conv3d.weight, pretrain: torch.Size([64, 24, 3, 3, 3])
Unused from pretrain module.Mixed_4d.b2b.bn.weight, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4d.b2b.bn.bias, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4d.b2b.bn.running_mean, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4d.b2b.bn.running_var, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4d.b2b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4d.b3b.conv3d.weight, pretrain: torch.Size([64, 512, 1, 1, 1])
Unused from pretrain module.Mixed_4d.b3b.bn.weight, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4d.b3b.bn.bias, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4d.b3b.bn.running_mean, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4d.b3b.bn.running_var, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4d.b3b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4e.b0.conv3d.weight, pretrain: torch.Size([112, 512, 1, 1, 1])
Unused from pretrain module.Mixed_4e.b0.bn.weight, pretrain: torch.Size([112])
Unused from pretrain module.Mixed_4e.b0.bn.bias, pretrain: torch.Size([112])
Unused from pretrain module.Mixed_4e.b0.bn.running_mean, pretrain: torch.Size([112])
Unused from pretrain module.Mixed_4e.b0.bn.running_var, pretrain: torch.Size([112])
Unused from pretrain module.Mixed_4e.b0.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4e.b1a.conv3d.weight, pretrain: torch.Size([144, 512, 1, 1, 1])
Unused from pretrain module.Mixed_4e.b1a.bn.weight, pretrain: torch.Size([144])
Unused from pretrain module.Mixed_4e.b1a.bn.bias, pretrain: torch.Size([144])
Unused from pretrain module.Mixed_4e.b1a.bn.running_mean, pretrain: torch.Size([144])
Unused from pretrain module.Mixed_4e.b1a.bn.running_var, pretrain: torch.Size([144])
Unused from pretrain module.Mixed_4e.b1a.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4e.b1b.conv3d.weight, pretrain: torch.Size([288, 144, 3, 3, 3])
Unused from pretrain module.Mixed_4e.b1b.bn.weight, pretrain: torch.Size([288])
Unused from pretrain module.Mixed_4e.b1b.bn.bias, pretrain: torch.Size([288])
Unused from pretrain module.Mixed_4e.b1b.bn.running_mean, pretrain: torch.Size([288])
Unused from pretrain module.Mixed_4e.b1b.bn.running_var, pretrain: torch.Size([288])
Unused from pretrain module.Mixed_4e.b1b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4e.b2a.conv3d.weight, pretrain: torch.Size([32, 512, 1, 1, 1])
Unused from pretrain module.Mixed_4e.b2a.bn.weight, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_4e.b2a.bn.bias, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_4e.b2a.bn.running_mean, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_4e.b2a.bn.running_var, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_4e.b2a.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4e.b2b.conv3d.weight, pretrain: torch.Size([64, 32, 3, 3, 3])
Unused from pretrain module.Mixed_4e.b2b.bn.weight, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4e.b2b.bn.bias, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4e.b2b.bn.running_mean, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4e.b2b.bn.running_var, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4e.b2b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4e.b3b.conv3d.weight, pretrain: torch.Size([64, 512, 1, 1, 1])
Unused from pretrain module.Mixed_4e.b3b.bn.weight, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4e.b3b.bn.bias, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4e.b3b.bn.running_mean, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4e.b3b.bn.running_var, pretrain: torch.Size([64])
Unused from pretrain module.Mixed_4e.b3b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4f.b0.conv3d.weight, pretrain: torch.Size([256, 528, 1, 1, 1])
Unused from pretrain module.Mixed_4f.b0.bn.weight, pretrain: torch.Size([256])
Unused from pretrain module.Mixed_4f.b0.bn.bias, pretrain: torch.Size([256])
Unused from pretrain module.Mixed_4f.b0.bn.running_mean, pretrain: torch.Size([256])
Unused from pretrain module.Mixed_4f.b0.bn.running_var, pretrain: torch.Size([256])
Unused from pretrain module.Mixed_4f.b0.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4f.b1a.conv3d.weight, pretrain: torch.Size([160, 528, 1, 1, 1])
Unused from pretrain module.Mixed_4f.b1a.bn.weight, pretrain: torch.Size([160])
Unused from pretrain module.Mixed_4f.b1a.bn.bias, pretrain: torch.Size([160])
Unused from pretrain module.Mixed_4f.b1a.bn.running_mean, pretrain: torch.Size([160])
Unused from pretrain module.Mixed_4f.b1a.bn.running_var, pretrain: torch.Size([160])
Unused from pretrain module.Mixed_4f.b1a.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4f.b1b.conv3d.weight, pretrain: torch.Size([320, 160, 3, 3, 3])
Unused from pretrain module.Mixed_4f.b1b.bn.weight, pretrain: torch.Size([320])
Unused from pretrain module.Mixed_4f.b1b.bn.bias, pretrain: torch.Size([320])
Unused from pretrain module.Mixed_4f.b1b.bn.running_mean, pretrain: torch.Size([320])
Unused from pretrain module.Mixed_4f.b1b.bn.running_var, pretrain: torch.Size([320])
Unused from pretrain module.Mixed_4f.b1b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4f.b2a.conv3d.weight, pretrain: torch.Size([32, 528, 1, 1, 1])
Unused from pretrain module.Mixed_4f.b2a.bn.weight, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_4f.b2a.bn.bias, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_4f.b2a.bn.running_mean, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_4f.b2a.bn.running_var, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_4f.b2a.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4f.b2b.conv3d.weight, pretrain: torch.Size([128, 32, 3, 3, 3])
Unused from pretrain module.Mixed_4f.b2b.bn.weight, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_4f.b2b.bn.bias, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_4f.b2b.bn.running_mean, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_4f.b2b.bn.running_var, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_4f.b2b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_4f.b3b.conv3d.weight, pretrain: torch.Size([128, 528, 1, 1, 1])
Unused from pretrain module.Mixed_4f.b3b.bn.weight, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_4f.b3b.bn.bias, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_4f.b3b.bn.running_mean, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_4f.b3b.bn.running_var, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_4f.b3b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_5b.b0.conv3d.weight, pretrain: torch.Size([256, 832, 1, 1, 1])
Unused from pretrain module.Mixed_5b.b0.bn.weight, pretrain: torch.Size([256])
Unused from pretrain module.Mixed_5b.b0.bn.bias, pretrain: torch.Size([256])
Unused from pretrain module.Mixed_5b.b0.bn.running_mean, pretrain: torch.Size([256])
Unused from pretrain module.Mixed_5b.b0.bn.running_var, pretrain: torch.Size([256])
Unused from pretrain module.Mixed_5b.b0.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_5b.b1a.conv3d.weight, pretrain: torch.Size([160, 832, 1, 1, 1])
Unused from pretrain module.Mixed_5b.b1a.bn.weight, pretrain: torch.Size([160])
Unused from pretrain module.Mixed_5b.b1a.bn.bias, pretrain: torch.Size([160])
Unused from pretrain module.Mixed_5b.b1a.bn.running_mean, pretrain: torch.Size([160])
Unused from pretrain module.Mixed_5b.b1a.bn.running_var, pretrain: torch.Size([160])
Unused from pretrain module.Mixed_5b.b1a.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_5b.b1b.conv3d.weight, pretrain: torch.Size([320, 160, 3, 3, 3])
Unused from pretrain module.Mixed_5b.b1b.bn.weight, pretrain: torch.Size([320])
Unused from pretrain module.Mixed_5b.b1b.bn.bias, pretrain: torch.Size([320])
Unused from pretrain module.Mixed_5b.b1b.bn.running_mean, pretrain: torch.Size([320])
Unused from pretrain module.Mixed_5b.b1b.bn.running_var, pretrain: torch.Size([320])
Unused from pretrain module.Mixed_5b.b1b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_5b.b2a.conv3d.weight, pretrain: torch.Size([32, 832, 1, 1, 1])
Unused from pretrain module.Mixed_5b.b2a.bn.weight, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_5b.b2a.bn.bias, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_5b.b2a.bn.running_mean, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_5b.b2a.bn.running_var, pretrain: torch.Size([32])
Unused from pretrain module.Mixed_5b.b2a.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_5b.b2b.conv3d.weight, pretrain: torch.Size([128, 32, 3, 3, 3])
Unused from pretrain module.Mixed_5b.b2b.bn.weight, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_5b.b2b.bn.bias, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_5b.b2b.bn.running_mean, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_5b.b2b.bn.running_var, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_5b.b2b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_5b.b3b.conv3d.weight, pretrain: torch.Size([128, 832, 1, 1, 1])
Unused from pretrain module.Mixed_5b.b3b.bn.weight, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_5b.b3b.bn.bias, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_5b.b3b.bn.running_mean, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_5b.b3b.bn.running_var, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_5b.b3b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_5c.b0.conv3d.weight, pretrain: torch.Size([384, 832, 1, 1, 1])
Unused from pretrain module.Mixed_5c.b0.bn.weight, pretrain: torch.Size([384])
Unused from pretrain module.Mixed_5c.b0.bn.bias, pretrain: torch.Size([384])
Unused from pretrain module.Mixed_5c.b0.bn.running_mean, pretrain: torch.Size([384])
Unused from pretrain module.Mixed_5c.b0.bn.running_var, pretrain: torch.Size([384])
Unused from pretrain module.Mixed_5c.b0.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_5c.b1a.conv3d.weight, pretrain: torch.Size([192, 832, 1, 1, 1])
Unused from pretrain module.Mixed_5c.b1a.bn.weight, pretrain: torch.Size([192])
Unused from pretrain module.Mixed_5c.b1a.bn.bias, pretrain: torch.Size([192])
Unused from pretrain module.Mixed_5c.b1a.bn.running_mean, pretrain: torch.Size([192])
Unused from pretrain module.Mixed_5c.b1a.bn.running_var, pretrain: torch.Size([192])
Unused from pretrain module.Mixed_5c.b1a.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_5c.b1b.conv3d.weight, pretrain: torch.Size([384, 192, 3, 3, 3])
Unused from pretrain module.Mixed_5c.b1b.bn.weight, pretrain: torch.Size([384])
Unused from pretrain module.Mixed_5c.b1b.bn.bias, pretrain: torch.Size([384])
Unused from pretrain module.Mixed_5c.b1b.bn.running_mean, pretrain: torch.Size([384])
Unused from pretrain module.Mixed_5c.b1b.bn.running_var, pretrain: torch.Size([384])
Unused from pretrain module.Mixed_5c.b1b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_5c.b2a.conv3d.weight, pretrain: torch.Size([48, 832, 1, 1, 1])
Unused from pretrain module.Mixed_5c.b2a.bn.weight, pretrain: torch.Size([48])
Unused from pretrain module.Mixed_5c.b2a.bn.bias, pretrain: torch.Size([48])
Unused from pretrain module.Mixed_5c.b2a.bn.running_mean, pretrain: torch.Size([48])
Unused from pretrain module.Mixed_5c.b2a.bn.running_var, pretrain: torch.Size([48])
Unused from pretrain module.Mixed_5c.b2a.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_5c.b2b.conv3d.weight, pretrain: torch.Size([128, 48, 3, 3, 3])
Unused from pretrain module.Mixed_5c.b2b.bn.weight, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_5c.b2b.bn.bias, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_5c.b2b.bn.running_mean, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_5c.b2b.bn.running_var, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_5c.b2b.bn.num_batches_tracked, pretrain: torch.Size([])
Unused from pretrain module.Mixed_5c.b3b.conv3d.weight, pretrain: torch.Size([128, 832, 1, 1, 1])
Unused from pretrain module.Mixed_5c.b3b.bn.weight, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_5c.b3b.bn.bias, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_5c.b3b.bn.running_mean, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_5c.b3b.bn.running_var, pretrain: torch.Size([128])
Unused from pretrain module.Mixed_5c.b3b.bn.num_batches_tracked, pretrain: torch.Size([])
Missing in pretrain logits.conv3d.weight
Missing in pretrain logits.conv3d.bias
Missing in pretrain Conv3d_1a_7x7.conv3d.weight
Missing in pretrain Conv3d_1a_7x7.bn.weight
Missing in pretrain Conv3d_1a_7x7.bn.bias
Missing in pretrain Conv3d_1a_7x7.bn.running_mean
Missing in pretrain Conv3d_1a_7x7.bn.running_var
Missing in pretrain Conv3d_1a_7x7.bn.num_batches_tracked
Missing in pretrain Conv3d_2b_1x1.conv3d.weight
Missing in pretrain Conv3d_2b_1x1.bn.weight
Missing in pretrain Conv3d_2b_1x1.bn.bias
Missing in pretrain Conv3d_2b_1x1.bn.running_mean
Missing in pretrain Conv3d_2b_1x1.bn.running_var
Missing in pretrain Conv3d_2b_1x1.bn.num_batches_tracked
Missing in pretrain Conv3d_2c_3x3.conv3d.weight
Missing in pretrain Conv3d_2c_3x3.bn.weight
Missing in pretrain Conv3d_2c_3x3.bn.bias
Missing in pretrain Conv3d_2c_3x3.bn.running_mean
Missing in pretrain Conv3d_2c_3x3.bn.running_var
Missing in pretrain Conv3d_2c_3x3.bn.num_batches_tracked
Missing in pretrain Mixed_3b.b0.conv3d.weight
Missing in pretrain Mixed_3b.b0.bn.weight
Missing in pretrain Mixed_3b.b0.bn.bias
Missing in pretrain Mixed_3b.b0.bn.running_mean
Missing in pretrain Mixed_3b.b0.bn.running_var
Missing in pretrain Mixed_3b.b0.bn.num_batches_tracked
Missing in pretrain Mixed_3b.b1a.conv3d.weight
Missing in pretrain Mixed_3b.b1a.bn.weight
Missing in pretrain Mixed_3b.b1a.bn.bias
Missing in pretrain Mixed_3b.b1a.bn.running_mean
Missing in pretrain Mixed_3b.b1a.bn.running_var
Missing in pretrain Mixed_3b.b1a.bn.num_batches_tracked
Missing in pretrain Mixed_3b.b1b.conv3d.weight
Missing in pretrain Mixed_3b.b1b.bn.weight
Missing in pretrain Mixed_3b.b1b.bn.bias
Missing in pretrain Mixed_3b.b1b.bn.running_mean
Missing in pretrain Mixed_3b.b1b.bn.running_var
Missing in pretrain Mixed_3b.b1b.bn.num_batches_tracked
Missing in pretrain Mixed_3b.b2a.conv3d.weight
Missing in pretrain Mixed_3b.b2a.bn.weight
Missing in pretrain Mixed_3b.b2a.bn.bias
Missing in pretrain Mixed_3b.b2a.bn.running_mean
Missing in pretrain Mixed_3b.b2a.bn.running_var
Missing in pretrain Mixed_3b.b2a.bn.num_batches_tracked
Missing in pretrain Mixed_3b.b2b.conv3d.weight
Missing in pretrain Mixed_3b.b2b.bn.weight
Missing in pretrain Mixed_3b.b2b.bn.bias
Missing in pretrain Mixed_3b.b2b.bn.running_mean
Missing in pretrain Mixed_3b.b2b.bn.running_var
Missing in pretrain Mixed_3b.b2b.bn.num_batches_tracked
Missing in pretrain Mixed_3b.b3b.conv3d.weight
Missing in pretrain Mixed_3b.b3b.bn.weight
Missing in pretrain Mixed_3b.b3b.bn.bias
Missing in pretrain Mixed_3b.b3b.bn.running_mean
Missing in pretrain Mixed_3b.b3b.bn.running_var
Missing in pretrain Mixed_3b.b3b.bn.num_batches_tracked
Missing in pretrain Mixed_3c.b0.conv3d.weight
Missing in pretrain Mixed_3c.b0.bn.weight
Missing in pretrain Mixed_3c.b0.bn.bias
Missing in pretrain Mixed_3c.b0.bn.running_mean
Missing in pretrain Mixed_3c.b0.bn.running_var
Missing in pretrain Mixed_3c.b0.bn.num_batches_tracked
Missing in pretrain Mixed_3c.b1a.conv3d.weight
Missing in pretrain Mixed_3c.b1a.bn.weight
Missing in pretrain Mixed_3c.b1a.bn.bias
Missing in pretrain Mixed_3c.b1a.bn.running_mean
Missing in pretrain Mixed_3c.b1a.bn.running_var
Missing in pretrain Mixed_3c.b1a.bn.num_batches_tracked
Missing in pretrain Mixed_3c.b1b.conv3d.weight
Missing in pretrain Mixed_3c.b1b.bn.weight
Missing in pretrain Mixed_3c.b1b.bn.bias
Missing in pretrain Mixed_3c.b1b.bn.running_mean
Missing in pretrain Mixed_3c.b1b.bn.running_var
Missing in pretrain Mixed_3c.b1b.bn.num_batches_tracked
Missing in pretrain Mixed_3c.b2a.conv3d.weight
Missing in pretrain Mixed_3c.b2a.bn.weight
Missing in pretrain Mixed_3c.b2a.bn.bias
Missing in pretrain Mixed_3c.b2a.bn.running_mean
Missing in pretrain Mixed_3c.b2a.bn.running_var
Missing in pretrain Mixed_3c.b2a.bn.num_batches_tracked
Missing in pretrain Mixed_3c.b2b.conv3d.weight
Missing in pretrain Mixed_3c.b2b.bn.weight
Missing in pretrain Mixed_3c.b2b.bn.bias
Missing in pretrain Mixed_3c.b2b.bn.running_mean
Missing in pretrain Mixed_3c.b2b.bn.running_var
Missing in pretrain Mixed_3c.b2b.bn.num_batches_tracked
Missing in pretrain Mixed_3c.b3b.conv3d.weight
Missing in pretrain Mixed_3c.b3b.bn.weight
Missing in pretrain Mixed_3c.b3b.bn.bias
Missing in pretrain Mixed_3c.b3b.bn.running_mean
Missing in pretrain Mixed_3c.b3b.bn.running_var
Missing in pretrain Mixed_3c.b3b.bn.num_batches_tracked
Missing in pretrain Mixed_4b.b0.conv3d.weight
Missing in pretrain Mixed_4b.b0.bn.weight
Missing in pretrain Mixed_4b.b0.bn.bias
Missing in pretrain Mixed_4b.b0.bn.running_mean
Missing in pretrain Mixed_4b.b0.bn.running_var
Missing in pretrain Mixed_4b.b0.bn.num_batches_tracked
Missing in pretrain Mixed_4b.b1a.conv3d.weight
Missing in pretrain Mixed_4b.b1a.bn.weight
Missing in pretrain Mixed_4b.b1a.bn.bias
Missing in pretrain Mixed_4b.b1a.bn.running_mean
Missing in pretrain Mixed_4b.b1a.bn.running_var
Missing in pretrain Mixed_4b.b1a.bn.num_batches_tracked
Missing in pretrain Mixed_4b.b1b.conv3d.weight
Missing in pretrain Mixed_4b.b1b.bn.weight
Missing in pretrain Mixed_4b.b1b.bn.bias
Missing in pretrain Mixed_4b.b1b.bn.running_mean
Missing in pretrain Mixed_4b.b1b.bn.running_var
Missing in pretrain Mixed_4b.b1b.bn.num_batches_tracked
Missing in pretrain Mixed_4b.b2a.conv3d.weight
Missing in pretrain Mixed_4b.b2a.bn.weight
Missing in pretrain Mixed_4b.b2a.bn.bias
Missing in pretrain Mixed_4b.b2a.bn.running_mean
Missing in pretrain Mixed_4b.b2a.bn.running_var
Missing in pretrain Mixed_4b.b2a.bn.num_batches_tracked
Missing in pretrain Mixed_4b.b2b.conv3d.weight
Missing in pretrain Mixed_4b.b2b.bn.weight
Missing in pretrain Mixed_4b.b2b.bn.bias
Missing in pretrain Mixed_4b.b2b.bn.running_mean
Missing in pretrain Mixed_4b.b2b.bn.running_var
Missing in pretrain Mixed_4b.b2b.bn.num_batches_tracked
Missing in pretrain Mixed_4b.b3b.conv3d.weight
Missing in pretrain Mixed_4b.b3b.bn.weight
Missing in pretrain Mixed_4b.b3b.bn.bias
Missing in pretrain Mixed_4b.b3b.bn.running_mean
Missing in pretrain Mixed_4b.b3b.bn.running_var
Missing in pretrain Mixed_4b.b3b.bn.num_batches_tracked
Missing in pretrain Mixed_4c.b0.conv3d.weight
Missing in pretrain Mixed_4c.b0.bn.weight
Missing in pretrain Mixed_4c.b0.bn.bias
Missing in pretrain Mixed_4c.b0.bn.running_mean
Missing in pretrain Mixed_4c.b0.bn.running_var
Missing in pretrain Mixed_4c.b0.bn.num_batches_tracked
Missing in pretrain Mixed_4c.b1a.conv3d.weight
Missing in pretrain Mixed_4c.b1a.bn.weight
Missing in pretrain Mixed_4c.b1a.bn.bias
Missing in pretrain Mixed_4c.b1a.bn.running_mean
Missing in pretrain Mixed_4c.b1a.bn.running_var
Missing in pretrain Mixed_4c.b1a.bn.num_batches_tracked
Missing in pretrain Mixed_4c.b1b.conv3d.weight
Missing in pretrain Mixed_4c.b1b.bn.weight
Missing in pretrain Mixed_4c.b1b.bn.bias
Missing in pretrain Mixed_4c.b1b.bn.running_mean
Missing in pretrain Mixed_4c.b1b.bn.running_var
Missing in pretrain Mixed_4c.b1b.bn.num_batches_tracked
Missing in pretrain Mixed_4c.b2a.conv3d.weight
Missing in pretrain Mixed_4c.b2a.bn.weight
Missing in pretrain Mixed_4c.b2a.bn.bias
Missing in pretrain Mixed_4c.b2a.bn.running_mean
Missing in pretrain Mixed_4c.b2a.bn.running_var
Missing in pretrain Mixed_4c.b2a.bn.num_batches_tracked
Missing in pretrain Mixed_4c.b2b.conv3d.weight
Missing in pretrain Mixed_4c.b2b.bn.weight
Missing in pretrain Mixed_4c.b2b.bn.bias
Missing in pretrain Mixed_4c.b2b.bn.running_mean
Missing in pretrain Mixed_4c.b2b.bn.running_var
Missing in pretrain Mixed_4c.b2b.bn.num_batches_tracked
Missing in pretrain Mixed_4c.b3b.conv3d.weight
Missing in pretrain Mixed_4c.b3b.bn.weight
Missing in pretrain Mixed_4c.b3b.bn.bias
Missing in pretrain Mixed_4c.b3b.bn.running_mean
Missing in pretrain Mixed_4c.b3b.bn.running_var
Missing in pretrain Mixed_4c.b3b.bn.num_batches_tracked
Missing in pretrain Mixed_4d.b0.conv3d.weight
Missing in pretrain Mixed_4d.b0.bn.weight
Missing in pretrain Mixed_4d.b0.bn.bias
Missing in pretrain Mixed_4d.b0.bn.running_mean
Missing in pretrain Mixed_4d.b0.bn.running_var
Missing in pretrain Mixed_4d.b0.bn.num_batches_tracked
Missing in pretrain Mixed_4d.b1a.conv3d.weight
Missing in pretrain Mixed_4d.b1a.bn.weight
Missing in pretrain Mixed_4d.b1a.bn.bias
Missing in pretrain Mixed_4d.b1a.bn.running_mean
Missing in pretrain Mixed_4d.b1a.bn.running_var
Missing in pretrain Mixed_4d.b1a.bn.num_batches_tracked
Missing in pretrain Mixed_4d.b1b.conv3d.weight
Missing in pretrain Mixed_4d.b1b.bn.weight
Missing in pretrain Mixed_4d.b1b.bn.bias
Missing in pretrain Mixed_4d.b1b.bn.running_mean
Missing in pretrain Mixed_4d.b1b.bn.running_var
Missing in pretrain Mixed_4d.b1b.bn.num_batches_tracked
Missing in pretrain Mixed_4d.b2a.conv3d.weight
Missing in pretrain Mixed_4d.b2a.bn.weight
Missing in pretrain Mixed_4d.b2a.bn.bias
Missing in pretrain Mixed_4d.b2a.bn.running_mean
Missing in pretrain Mixed_4d.b2a.bn.running_var
Missing in pretrain Mixed_4d.b2a.bn.num_batches_tracked
Missing in pretrain Mixed_4d.b2b.conv3d.weight
Missing in pretrain Mixed_4d.b2b.bn.weight
Missing in pretrain Mixed_4d.b2b.bn.bias
Missing in pretrain Mixed_4d.b2b.bn.running_mean
Missing in pretrain Mixed_4d.b2b.bn.running_var
Missing in pretrain Mixed_4d.b2b.bn.num_batches_tracked
Missing in pretrain Mixed_4d.b3b.conv3d.weight
Missing in pretrain Mixed_4d.b3b.bn.weight
Missing in pretrain Mixed_4d.b3b.bn.bias
Missing in pretrain Mixed_4d.b3b.bn.running_mean
Missing in pretrain Mixed_4d.b3b.bn.running_var
Missing in pretrain Mixed_4d.b3b.bn.num_batches_tracked
Missing in pretrain Mixed_4e.b0.conv3d.weight
Missing in pretrain Mixed_4e.b0.bn.weight
Missing in pretrain Mixed_4e.b0.bn.bias
Missing in pretrain Mixed_4e.b0.bn.running_mean
Missing in pretrain Mixed_4e.b0.bn.running_var
Missing in pretrain Mixed_4e.b0.bn.num_batches_tracked
Missing in pretrain Mixed_4e.b1a.conv3d.weight
Missing in pretrain Mixed_4e.b1a.bn.weight
Missing in pretrain Mixed_4e.b1a.bn.bias
Missing in pretrain Mixed_4e.b1a.bn.running_mean
Missing in pretrain Mixed_4e.b1a.bn.running_var
Missing in pretrain Mixed_4e.b1a.bn.num_batches_tracked
Missing in pretrain Mixed_4e.b1b.conv3d.weight
Missing in pretrain Mixed_4e.b1b.bn.weight
Missing in pretrain Mixed_4e.b1b.bn.bias
Missing in pretrain Mixed_4e.b1b.bn.running_mean
Missing in pretrain Mixed_4e.b1b.bn.running_var
Missing in pretrain Mixed_4e.b1b.bn.num_batches_tracked
Missing in pretrain Mixed_4e.b2a.conv3d.weight
Missing in pretrain Mixed_4e.b2a.bn.weight
Missing in pretrain Mixed_4e.b2a.bn.bias
Missing in pretrain Mixed_4e.b2a.bn.running_mean
Missing in pretrain Mixed_4e.b2a.bn.running_var
Missing in pretrain Mixed_4e.b2a.bn.num_batches_tracked
Missing in pretrain Mixed_4e.b2b.conv3d.weight
Missing in pretrain Mixed_4e.b2b.bn.weight
Missing in pretrain Mixed_4e.b2b.bn.bias
Missing in pretrain Mixed_4e.b2b.bn.running_mean
Missing in pretrain Mixed_4e.b2b.bn.running_var
Missing in pretrain Mixed_4e.b2b.bn.num_batches_tracked
Missing in pretrain Mixed_4e.b3b.conv3d.weight
Missing in pretrain Mixed_4e.b3b.bn.weight
Missing in pretrain Mixed_4e.b3b.bn.bias
Missing in pretrain Mixed_4e.b3b.bn.running_mean
Missing in pretrain Mixed_4e.b3b.bn.running_var
Missing in pretrain Mixed_4e.b3b.bn.num_batches_tracked
Missing in pretrain Mixed_4f.b0.conv3d.weight
Missing in pretrain Mixed_4f.b0.bn.weight
Missing in pretrain Mixed_4f.b0.bn.bias
Missing in pretrain Mixed_4f.b0.bn.running_mean
Missing in pretrain Mixed_4f.b0.bn.running_var
Missing in pretrain Mixed_4f.b0.bn.num_batches_tracked
Missing in pretrain Mixed_4f.b1a.conv3d.weight
Missing in pretrain Mixed_4f.b1a.bn.weight
Missing in pretrain Mixed_4f.b1a.bn.bias
Missing in pretrain Mixed_4f.b1a.bn.running_mean
Missing in pretrain Mixed_4f.b1a.bn.running_var
Missing in pretrain Mixed_4f.b1a.bn.num_batches_tracked
Missing in pretrain Mixed_4f.b1b.conv3d.weight
Missing in pretrain Mixed_4f.b1b.bn.weight
Missing in pretrain Mixed_4f.b1b.bn.bias
Missing in pretrain Mixed_4f.b1b.bn.running_mean
Missing in pretrain Mixed_4f.b1b.bn.running_var
Missing in pretrain Mixed_4f.b1b.bn.num_batches_tracked
Missing in pretrain Mixed_4f.b2a.conv3d.weight
Missing in pretrain Mixed_4f.b2a.bn.weight
Missing in pretrain Mixed_4f.b2a.bn.bias
Missing in pretrain Mixed_4f.b2a.bn.running_mean
Missing in pretrain Mixed_4f.b2a.bn.running_var
Missing in pretrain Mixed_4f.b2a.bn.num_batches_tracked
Missing in pretrain Mixed_4f.b2b.conv3d.weight
Missing in pretrain Mixed_4f.b2b.bn.weight
Missing in pretrain Mixed_4f.b2b.bn.bias
Missing in pretrain Mixed_4f.b2b.bn.running_mean
Missing in pretrain Mixed_4f.b2b.bn.running_var
Missing in pretrain Mixed_4f.b2b.bn.num_batches_tracked
Missing in pretrain Mixed_4f.b3b.conv3d.weight
Missing in pretrain Mixed_4f.b3b.bn.weight
Missing in pretrain Mixed_4f.b3b.bn.bias
Missing in pretrain Mixed_4f.b3b.bn.running_mean
Missing in pretrain Mixed_4f.b3b.bn.running_var
Missing in pretrain Mixed_4f.b3b.bn.num_batches_tracked
Missing in pretrain Mixed_5b.b0.conv3d.weight
Missing in pretrain Mixed_5b.b0.bn.weight
Missing in pretrain Mixed_5b.b0.bn.bias
Missing in pretrain Mixed_5b.b0.bn.running_mean
Missing in pretrain Mixed_5b.b0.bn.running_var
Missing in pretrain Mixed_5b.b0.bn.num_batches_tracked
Missing in pretrain Mixed_5b.b1a.conv3d.weight
Missing in pretrain Mixed_5b.b1a.bn.weight
Missing in pretrain Mixed_5b.b1a.bn.bias
Missing in pretrain Mixed_5b.b1a.bn.running_mean
Missing in pretrain Mixed_5b.b1a.bn.running_var
Missing in pretrain Mixed_5b.b1a.bn.num_batches_tracked
Missing in pretrain Mixed_5b.b1b.conv3d.weight
Missing in pretrain Mixed_5b.b1b.bn.weight
Missing in pretrain Mixed_5b.b1b.bn.bias
Missing in pretrain Mixed_5b.b1b.bn.running_mean
Missing in pretrain Mixed_5b.b1b.bn.running_var
Missing in pretrain Mixed_5b.b1b.bn.num_batches_tracked
Missing in pretrain Mixed_5b.b2a.conv3d.weight
Missing in pretrain Mixed_5b.b2a.bn.weight
Missing in pretrain Mixed_5b.b2a.bn.bias
Missing in pretrain Mixed_5b.b2a.bn.running_mean
Missing in pretrain Mixed_5b.b2a.bn.running_var
Missing in pretrain Mixed_5b.b2a.bn.num_batches_tracked
Missing in pretrain Mixed_5b.b2b.conv3d.weight
Missing in pretrain Mixed_5b.b2b.bn.weight
Missing in pretrain Mixed_5b.b2b.bn.bias
Missing in pretrain Mixed_5b.b2b.bn.running_mean
Missing in pretrain Mixed_5b.b2b.bn.running_var
Missing in pretrain Mixed_5b.b2b.bn.num_batches_tracked
Missing in pretrain Mixed_5b.b3b.conv3d.weight
Missing in pretrain Mixed_5b.b3b.bn.weight
Missing in pretrain Mixed_5b.b3b.bn.bias
Missing in pretrain Mixed_5b.b3b.bn.running_mean
Missing in pretrain Mixed_5b.b3b.bn.running_var
Missing in pretrain Mixed_5b.b3b.bn.num_batches_tracked
Missing in pretrain Mixed_5c.b0.conv3d.weight
Missing in pretrain Mixed_5c.b0.bn.weight
Missing in pretrain Mixed_5c.b0.bn.bias
Missing in pretrain Mixed_5c.b0.bn.running_mean
Missing in pretrain Mixed_5c.b0.bn.running_var
Missing in pretrain Mixed_5c.b0.bn.num_batches_tracked
Missing in pretrain Mixed_5c.b1a.conv3d.weight
Missing in pretrain Mixed_5c.b1a.bn.weight
Missing in pretrain Mixed_5c.b1a.bn.bias
Missing in pretrain Mixed_5c.b1a.bn.running_mean
Missing in pretrain Mixed_5c.b1a.bn.running_var
Missing in pretrain Mixed_5c.b1a.bn.num_batches_tracked
Missing in pretrain Mixed_5c.b1b.conv3d.weight
Missing in pretrain Mixed_5c.b1b.bn.weight
Missing in pretrain Mixed_5c.b1b.bn.bias
Missing in pretrain Mixed_5c.b1b.bn.running_mean
Missing in pretrain Mixed_5c.b1b.bn.running_var
Missing in pretrain Mixed_5c.b1b.bn.num_batches_tracked
Missing in pretrain Mixed_5c.b2a.conv3d.weight
Missing in pretrain Mixed_5c.b2a.bn.weight
Missing in pretrain Mixed_5c.b2a.bn.bias
Missing in pretrain Mixed_5c.b2a.bn.running_mean
Missing in pretrain Mixed_5c.b2a.bn.running_var
Missing in pretrain Mixed_5c.b2a.bn.num_batches_tracked
Missing in pretrain Mixed_5c.b2b.conv3d.weight
Missing in pretrain Mixed_5c.b2b.bn.weight
Missing in pretrain Mixed_5c.b2b.bn.bias
Missing in pretrain Mixed_5c.b2b.bn.running_mean
Missing in pretrain Mixed_5c.b2b.bn.running_var
Missing in pretrain Mixed_5c.b2b.bn.num_batches_tracked
Missing in pretrain Mixed_5c.b3b.conv3d.weight
Missing in pretrain Mixed_5c.b3b.bn.weight
Missing in pretrain Mixed_5c.b3b.bn.bias
Missing in pretrain Mixed_5c.b3b.bn.running_mean
Missing in pretrain Mixed_5c.b3b.bn.running_var
Missing in pretrain Mixed_5c.b3b.bn.num_batches_tracked
Removing or not initializing some layers...

```