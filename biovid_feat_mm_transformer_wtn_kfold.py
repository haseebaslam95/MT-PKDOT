from video_dataset_mm import  VideoFrameDataset, ImglistToTensor
from comet_ml import Experiment
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
from biovid_physio_classification import PhysioResNet18

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from torchvision.models.video import r3d_18
from torchvision import models
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from validate import validate_mmtransformer
from models.models import  VIS_PHY_MODEL_CAM, MultimodalTransformer
from models.orig_cam import CAM   
from models.transformation_network import TransformNet  
import copy
from mmd_loss import MMD_loss



"""
Training settings

"""


num_epochs = 100
best_epoch = 0
check_every = 1
b_size = 64
num_classes = 2  # Number of classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_val_acc=0
lr_vis_phy = 0.0001
lr_mmtransformer = 0.001


###### COMET Settings #######
experiment = Experiment(
    api_key="U0t7sSZhwEHvDLko0tJ4kbPH0",
    project_name="dmwl-t",
    workspace="haseebaslam952",
    disabled=False,)

parameters = {'batch_size': b_size,
              'learning_rate bb': lr_vis_phy,
              'learning_rate mmtransformer': lr_mmtransformer,
              'epochs':num_epochs            
              }
experiment.log_parameters(parameters)

criterion = nn.CrossEntropyLoss()

#change the criterion to MMD loss
# criterion_tn_mmd = MMD_loss()
# criterion_tn = nn.MSELoss()

videos_root = '/home/livia/work/Biovid/PartB/biovid_classes'


preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(112),  # image batch, resize smaller edge to 299
        # transforms.CenterCrop(112),  # image batch, center crop to square 299x299
        transforms.RandomHorizontalFlip(p=0.7),  # video batch, each frame horizontally flipped with probability 0.5
        # transforms.RandomCrop(112),  # video batch, each frame randomly cropped to 224x224
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.2, hue=0.1),  # video batch, color jitter with probability 0.1
        # transforms.RandomGrayscale(p=0.1),  # video batch, random grayscale with probability 0.1
        transforms.RandomRotation(10),  # video batch, each frame randomly rotated by -10 to 10 degrees
        # transforms.RandomPerspective(distortion_scale=0.1, p=0.1, interpolation=3),  # video batch, each frame randomly perspectively transformed with probability 0.1
        # transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),  # video batch, each frame randomly erased with probability 0.1
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # video batch, normalize with ImageNet mean and standard deviation
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])





def denormalize(video_tensor):
    inverse_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    return (inverse_normalize(video_tensor) * 255.).type(torch.uint8).permute(0, 2, 3, 1).numpy()






def train_model(train_file_path, val_file_path,fold_num):
    best_transform_loss = 10
    best_transform_epoch = 0    
    train_annotation_file = os.path.join(videos_root, train_file_path)
    val_annotation_file = os.path.join(videos_root, val_file_path)
    model_save_path_bb = os.path.join(os.getcwd(), 'saved_weights_DMWL/all_model_best_feat_concat_fusion_mmtransformer_visphy_m_DMWL'+str(fold_num)+'.pth')
    model_save_path_mmtransformer = os.path.join(os.getcwd(), 'saved_weights_DMWL/all_model_best_feat_concat_fusion_mmtransformer_DMWL'+str(fold_num)+'.pth')
    model_save_path_tn = os.path.join(os.getcwd(), 'saved_weights_DMWL/transform_net_saved_fold_'+str(fold_num)+'.pth')

    vis_phy_model=VIS_PHY_MODEL_CAM().to(device=device)
    mm_transformer = MultimodalTransformer(visual_dim=512, physiological_dim=512, num_heads=2, hidden_dim=512, num_layers=2, num_classes=2)
    mm_transformer = mm_transformer.to(device=device)
    transform_net = TransformNet().to(device=device)

    vis_phy_optimizer = optim.SGD(vis_phy_model.parameters(), lr=lr_vis_phy)
    mmtransformer_optimizer = optim.SGD(mm_transformer.parameters(), lr=lr_mmtransformer)
    # mmtransformer_optimizer = optim.Adam(mm_transformer.parameters(), lr=lr_mmtransformer, weight_decay=0.0001, amsgrad=True, eps=1e-8, betas=(0.9, 0.999))
    optimizer_tn = optim.Adam(transform_net.parameters(), lr=0.00005)
    # scheduler minimum learning rate is 1e-6
    scheduler = ReduceLROnPlateau(mmtransformer_optimizer, mode='max', factor=0.01, patience=10, verbose=True, min_lr=1e-7)

    train_dataset = VideoFrameDataset(
    root_path=videos_root,
    annotationfile_path=train_annotation_file,
    num_segments=10,
    frames_per_segment=1,
    imagefile_template='img_{:05d}.jpg',
    transform=preprocess,
    test_mode=False)

    val_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=val_annotation_file,
        num_segments=10,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        test_mode=True)
    
    train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=b_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True)


    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=b_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)


    best_val_acc = 0
    min_transform_loss = 10
    with experiment.train():
        for epoch in tqdm(range(num_epochs), desc='Epochs'):
            
            vis_phy_model.vis_model.train()
            vis_phy_model.phy_model.train()
            mm_transformer.train()
            # transform_net.train()

            
            # classifier_m.train()
            
            running_loss = 0.0
            correct = 0
            total = 0
            for i,(spec_2d,video_batch, labels) in enumerate(train_dataloader,0):
                mmtransformer_optimizer.zero_grad()
                vis_phy_optimizer.zero_grad()
                # optimizer_tn.zero_grad()

                video_batch=video_batch.permute(0, 2, 1, 3, 4)
                video_batch = video_batch.to(device)
                with torch.no_grad():
                    vis_feats, phy_feats = vis_phy_model.model_out_feats(video_batch,spec_2d)

                # vis_feats_tn = copy.deepcopy(vis_feats)

                #normalize vis_feats and phy_feats
                # vis_feats = torch.nn.functional.normalize(vis_feats, p=2, dim=1)
                # phy_feats = torch.nn.functional.normalize(phy_feats, p=2, dim=1)


                vis_feats = vis_feats.unsqueeze(1)
                phy_feats = phy_feats.unsqueeze(1)

                outs = mm_transformer(vis_feats, phy_feats)
                labels = labels.to(device)
                t_loss = criterion(outs, labels)
                t_loss.backward()
                vis_phy_optimizer.step()
                mmtransformer_optimizer.step()
                

                # detached_vis_feats = vis_feats.detach()
                # detached_phy_feats = phy_feats.detach()

                # recon_phy_feats = transform_net(detached_vis_feats.squeeze(1))
                # transform_loss = criterion_tn_mmd(recon_phy_feats, detached_phy_feats.squeeze(1))
                # transform_loss.backward()
                
                # optimizer_tn.step()

                running_loss += t_loss.item()
                _, predicted = torch.max(outs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                    running_loss = 0.0
            
            train_accuracy= 100 * correct / total
            print("*********************************************\n")
            print(f"Accuracy after epoch {epoch + 1}: {train_accuracy}%")
            # print(f"Transform loss after epoch {epoch + 1}: {transform_loss}")
            train_loss= running_loss / 100
            experiment.log_metric('Loss', train_loss,epoch= epoch)
            experiment.log_metric('Accuracy', train_accuracy ,epoch= epoch)
            # experiment.log_metric('Transform Loss', transform_loss, epoch= epoch)
            # if transform_loss < min_transform_loss:
            #     min_transform_loss = transform_loss
            #     best_transform_loss = transform_loss
            #     best_transform_epoch = epoch+1
            #     # model_save_path_tn = os.path.join(os.getcwd(), 'transform_net_saved_DMWL.pth')
            #     torch.save(transform_net.state_dict(), model_save_path_tn)  
            #     print('Best transform model saved at epoch: ', epoch+1)
                

            # last_lr=scheduler.get_last_lr()
            # experiment.log_metric('Learning Rate', last_lr,epoch= epoch)
            if epoch % check_every == 0:
                val_acc, val_loss = validate_mmtransformer(vis_phy_model,mm_transformer, val_dataloader, criterion, device)
                print( "Validation accuracy: ", val_acc)
                experiment.log_metric('Val Accuracy', val_acc,epoch= epoch)
                experiment.log_metric('Val Loss', val_loss,epoch= epoch)
                # scheduler.step(val_acc)
                current_lr = mmtransformer_optimizer.param_groups[0]['lr']
                experiment.log_metric('Learning Rate', current_lr,epoch= epoch)
                # print('Current learning rate: ', current_lr)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # model_save_path_bb = os.path.join(os.getcwd(), 'model_best_feat_concat_fusion_mmtransformer_visphy_m.pth')
                    # model_save_path_mmtransformer = os.path.join(os.getcwd(), 'model_best_feat_concat_fusion_mmtransformer.pth')

                    torch.save(vis_phy_model.state_dict(), model_save_path_bb)
                    torch.save(mm_transformer.state_dict(), model_save_path_mmtransformer)  
                    print('Best model saved at epoch: ', epoch+1)
                    best_epoch = epoch+1


    print("Finished Training")

    train_accuracy = 100 * correct / total
    avg_train_loss = running_loss / len(train_dataloader)
    print(f'Training accuracy: {train_accuracy}%')
    print(f'Training loss: {avg_train_loss}')

    print("Best model saved at epoch: ", best_epoch)
    print("Best validation accuracy: ", best_val_acc)
    return best_val_acc, best_transform_loss, best_transform_epoch






def train_k_fold():
    k=5
    #create list of size 
    best_acc_list = [] 
    bes_tl_list = []

    for i in range(k):
        print("Fold: ", i+1)
        train_file_path = 'fold_'+ str(i+1) +'_train.txt'
        val_file_path = 'fold_'+ str(i+1) +'_test.txt'
        fold_num = str(i+1)
        best_fold_acc, btl, bte=train_model(train_file_path, val_file_path,fold_num)
        best_acc_list.append(best_fold_acc)
        bes_tl_list.append(btl)
    
    print("Best validation accuracy for each fold: ", best_acc_list)
    print("Best transform loss for each fold: ", bes_tl_list)
    print("Average best validation accuracy: ", sum(best_acc_list)/k)

if __name__ == '__main__':
    train_k_fold()