from video_dataset_mm import  VideoFrameDataset, ImglistToTensor
from comet_ml import Experiment
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import numpy as np
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
from validate import *
from models.models import  VIS_PHY_MODEL_CAM, MultimodalTransformer, VIS_MODEL
from models.orig_cam import CAM 
   




# class CustomLoss(nn.Module):
#     def __init__(self, weight=1.0):
#         super(CustomLoss, self).__init__()
#         self.weight = weight

#     def forward(self, frob_loss):
#         return torch.tensor(frob_loss * self.weight, requires_grad=True)


def cosine_similarity_matrix_rows(matrix):
    similarity_matrix = np.zeros((matrix.shape[0], matrix.shape[0]))
    
    for i, row1 in enumerate(matrix):
        for j, row2 in enumerate(matrix):
            dot_product = np.dot(row1, row2)
            norm_row1 = np.linalg.norm(row1)
            norm_row2 = np.linalg.norm(row2)
            similarity_matrix[i, j] = dot_product / (norm_row1 * norm_row2)
    
    return similarity_matrix


def cosine_similarity_matrix_transpose(matrix):
    transpose_matrix = matrix.T
    similarity_matrix = np.dot(matrix, transpose_matrix)
    norm_matrix = np.linalg.norm(matrix, axis=1)[:, np.newaxis]
    norm_transpose = np.linalg.norm(transpose_matrix, axis=0)[np.newaxis, :]
    similarity_matrix /= np.dot(norm_matrix, norm_transpose)
    return similarity_matrix






"""
Training settings

"""
num_epochs = 10
best_epoch = 0
check_every = 1
b_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_val_acc=0

lr_vis_phy = 0.00001
lr_mmtransformer = 0.001

experiment = Experiment(
  api_key="U0t7sSZhwEHvDLko0tJ4kbPH0",
  project_name="dmwl",
  workspace="haseebaslam952",
  disabled=True
)

parameters = {'batch_size': b_size,
              'learning_rate bb': lr_vis_phy,
              'learning_rate mmtransformer': lr_mmtransformer,
              'epochs':num_epochs            
              }
experiment.log_parameters(parameters)


num_frames = 5  # Number of frames in each video clip
num_channels = 3  # Number of channels (e.g., RGB)
video_length = 112  # Length of the video in each dimension
num_classes = 2  # Number of classes


criterion = nn.CrossEntropyLoss()


student_model = VIS_MODEL().to(device=device)





# params = [{'params': vis_phy_model.parameters(), 'lr': lr_vis_phy},{'params': mm_transformer.parameters(), 'lr': lr_mmtransformer}]

student_optimizer = optim.Adam(student_model.parameters(), lr=lr_vis_phy)




# vis_phy_mmtransformer_optimizer = optim.Adam(params)
# scheduler = optim.lr_scheduler.StepLR(vis_phy_optimizer, step_size=5, gamma=0.01)
# scheduler = ReduceLROnPlateau(mmtransformer_optimizer, mode='min', factor=0.01, patience=3,verbose=True)

videos_root = '/home/livia/work/Biovid/PartB/biovid_classes'
train_annotation_file = os.path.join(videos_root, 'annotations_filtered_peak_2_train.txt')
val_annotation_file = os.path.join(videos_root, 'annotations_filtered_peak_2_val.txt')
test_annotation_file = os.path.join(videos_root, 'annotations_filtered_peak_2_test.txt')

preprocess = transforms.Compose([
    ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
    transforms.Resize(112),  # image batch, resize smaller edge to 299
    transforms.CenterCrop(112),  # image batch, center crop to square 299x299
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

test_dataset = VideoFrameDataset(
    root_path=videos_root,
    annotationfile_path=test_annotation_file,
    num_segments=10,
    frames_per_segment=1,
    imagefile_template='img_{:05d}.jpg',
    transform=preprocess,
    test_mode=True)


def denormalize(video_tensor):
    inverse_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    return (inverse_normalize(video_tensor) * 255.).type(torch.uint8).permute(0, 2, 3, 1).numpy()


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

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=b_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True)


vis_phy_model_teacher = VIS_PHY_MODEL_CAM().to(device=device)
mm_transformer_teacher = MultimodalTransformer(visual_dim=512, physiological_dim=512, num_heads=2, hidden_dim=512, num_layers=2, num_classes=2).to(device=device)
vis_phy_model_teacher.load_state_dict(torch.load('../model_saved_biovid/model_best_feat_concat_fusion_mmtransformer_visphy_m_878.pth'))
mm_transformer_teacher.load_state_dict(torch.load('../model_saved_biovid/model_best_feat_concat_fusion_mmtransformer_878.pth'))
vis_phy_model_teacher.eval()
mm_transformer_teacher.eval()


cos_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)


def train_student():
    best_val_acc = 0
    best_epoch = 1
    with experiment.train():
        for epoch in tqdm(range(num_epochs), desc='Epochs'):
            
            student_model.vis_model.train()
            
            # classifier_m.train()
            
            running_loss = 0.0
            correct = 0
            total = 0
            for i,(spec_2d,video_batch, labels) in enumerate(train_dataloader,0):
                

                student_optimizer.zero_grad()

                #prepare data
                video_batch=video_batch.permute(0, 2, 1, 3, 4)
                video_batch = video_batch.to(device)
                labels = labels.to(device)

                #get teacher embeddings
                with torch.no_grad():
                    vis_feats_teacher, phy_feats_teacher = vis_phy_model_teacher.model_out_feats(video_batch,spec_2d)
                    vis_feats_teacher = vis_feats_teacher.unsqueeze(1)
                    phy_feats_teacher = phy_feats_teacher.unsqueeze(1)
                    _, intermediate_outs_teacher = mm_transformer_teacher(vis_feats_teacher, phy_feats_teacher)



                
                #get student embeddings
                #create zeros of the same size as spec_2d
                vis_feats, outs = student_model.model_out(video_batch)
                # vis_feats = vis_feats.unsqueeze(1)



                cosloss= 1 - cos_similarity(intermediate_outs_teacher.view(-1,1).detach(),vis_feats.view(-1,1))


                #smpkd loss

                cosine_similarity_matrix_teacher = cosine_similarity_matrix_transpose(intermediate_outs_teacher.detach().cpu().numpy())
                cosine_similarity_matrix_student = cosine_similarity_matrix_transpose(vis_feats.detach().cpu().numpy())

                frobenius_norm = np.linalg.norm(cosine_similarity_matrix_teacher- cosine_similarity_matrix_student, ord='fro')



                gt_loss = criterion(outs, labels)

                total_loss = 0.5 * gt_loss + 0.5 * frobenius_norm
                # total_loss = gt_loss

                
                total_loss.backward()
                student_optimizer.step()

                running_loss += total_loss.item()
    
                _, predicted = torch.max(outs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                    running_loss = 0.0
            
            train_accuracy= 100 * correct / total
            print("*********************************************\n")
            print(f"Accuracy after epoch {epoch + 1}: {train_accuracy}%")
            train_loss= running_loss / 100
            print(f"Best Val:  {best_val_acc} at Epoch:  {best_epoch}")   
            experiment.log_metric('Loss', train_loss,epoch= epoch)
            experiment.log_metric('Accuracy', train_accuracy ,epoch= epoch)
            print("total loss: ", total_loss)
            # last_lr=scheduler.get_last_lr()
            # experiment.log_metric('Learning Rate', last_lr,epoch= epoch)
            if epoch % check_every == 0:
                val_acc, val_loss = validate_vis_only_pkd(student_model, val_dataloader, criterion, device)
                # print( "Validation accuracy: ", val_acc)
                # experiment.log_metric('Val Accuracy', val_acc,epoch= epoch)
                # experiment.log_metric('Val Loss', val_loss,epoch= epoch)
                # scheduler.step(val_loss)
                # print('Current learning rate: ', current_lr)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model_save_path_bb = os.path.join(os.getcwd(), 'model_best_dmwl_bb_pkd_visonly.pth')

                    torch.save(student_model.state_dict(), model_save_path_bb)
                    print('Best model saved at epoch: ', epoch+1)
                    best_epoch = epoch+1


    print("Finished Training")

    train_accuracy = 100 * correct / total
    avg_train_loss = running_loss / len(train_dataloader)
    print(f'Training accuracy: {train_accuracy}%')
    print(f'Training loss: {avg_train_loss}')

    print("Best model saved at epoch: ", best_epoch)
    print("Best validation accuracy: ", best_val_acc)



# train_student()
    

"""
Testing
"""

#load models

def test():
    print("Testing")
    student_model.load_state_dict(torch.load('model_best_dmwl_bb_pkd_visonly.pth'))
    student_model.eval()
    test_acc, test_loss = validate_vis_only_pkd(student_model, test_dataloader, criterion, device)
    print("Test accuracy: ", test_acc)


test()


