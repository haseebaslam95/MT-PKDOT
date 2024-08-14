from video_dataset_mm import  VideoFrameDataset, ImglistToTensor
from comet_ml import Experiment
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import numpy as np
from biovid_physio_classification import PhysioResNet18
from matplotlib.colors import ListedColormap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from torchvision.models.video import r3d_18
from torchvision import models
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from validate import validate_mmtransformer_dmwl_wtn, validate_mmtransformer_dmwl_kf, validate_self_distill, validate_mmtransformer_dmwl_self_distill
from models.models import  VIS_PHY_MODEL_CAM, MultimodalTransformer, VIS_MODEL, ClassifierHead
from models.transformation_network import TransformNet, AdapterNetwork
from models.orig_cam import CAM   
import ot
from geomloss import SamplesLoss
from sklearn.manifold import MDS, TSNE
import pandas as pd
import seaborn as sns
from mmd_loss import MMD_loss
from mpl_toolkits.mplot3d import Axes3D
from umap import UMAP

#import mse loss from pytorch
from torch.nn import MSELoss

# from sklearn.metrics.pairwise import rbf_kernel

# class CustomLoss(nn.Module):
#     def __init__(self, weight=1.0):
#         super(CustomLoss, self).__init__()
#         self.weight = weight

#     def forward(self, frob_loss):
#         return torch.tensor(frob_loss * self.weight, requires_grad=True)

# 
def rbf_kernel(X, Y, sigma=1.0):
    """
    Computes the RBF (Gaussian) kernel between X and Y with bandwidth sigma.
    """
    XX = torch.matmul(X, X.T)
    YY = torch.matmul(Y, Y.T)
    XY = torch.matmul(X, Y.T)
    
    X_sqnorms = torch.diag(XX)
    Y_sqnorms = torch.diag(YY)
    
    K_XY = torch.exp(-((X_sqnorms.unsqueeze(1) + Y_sqnorms.unsqueeze(0) - 2 * XY) / (2 * sigma ** 2)))
    K_XX = torch.exp(-((X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0) - 2 * XX) / (2 * sigma ** 2)))
    K_YY = torch.exp(-((Y_sqnorms.unsqueeze(1) + Y_sqnorms.unsqueeze(0) - 2 * YY) / (2 * sigma ** 2)))
    
    return K_XX, K_YY, K_XY


def compute_mmd(X, Y, sigma=1.0):
    """
    Computes the MMD between two sets of samples X and Y using an RBF kernel with bandwidth sigma.
    """
    K_XX, K_YY, K_XY = rbf_kernel(X, Y, sigma)
    
    return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

def cosine_similarity_matrix_rows(matrix):
    similarity_matrix = np.zeros((matrix.shape[0], matrix.shape[0]))
    
    for i, row1 in enumerate(matrix):
        for j, row2 in enumerate(matrix):
            dot_product = np.dot(row1, row2)
            norm_row1 = np.linalg.norm(row1)
            norm_row2 = np.linalg.norm(row2)
            similarity_matrix[i, j] = dot_product / (norm_row1 * norm_row2)
    
    return similarity_matrix
# 
# 
def cosine_similarity_matrix_transpose(matrix):
    transpose_matrix = matrix.T
    similarity_matrix = np.dot(matrix, transpose_matrix)
    norm_matrix = np.linalg.norm(matrix, axis=1)[:, np.newaxis]
    norm_transpose = np.linalg.norm(transpose_matrix, axis=0)[np.newaxis, :]
    similarity_matrix /= np.dot(norm_matrix, norm_transpose)
    return similarity_matrix


def cosine_similarity_matrix_transpose_torch(matrix):
    transpose_matrix = matrix.t()
    similarity_matrix = torch.mm(matrix, transpose_matrix)
    norm_matrix = torch.norm(matrix, dim=1, keepdim=True)
    norm_transpose = torch.norm(transpose_matrix, dim=0, keepdim=True)
    similarity_matrix.div_(torch.mm(norm_matrix, norm_transpose))
    return similarity_matrix




def plot_simmat(sim_mat_teacher, sim_mat_student, epoch, fold_num):
    # plot the similarity matrices
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(sim_mat_teacher)
    axs[1].imshow(sim_mat_student)   
    # plt.show()

    #save the similarity matrices as images
    fig.savefig(f'sim_mat_{epoch}_{fold_num}.png')




def plot_mds(matrix, title):
    embedding = MDS(n_components=2)
    mds = embedding.fit_transform(matrix)
    plt.figure(figsize=(10, 8))
    plt.scatter(mds[:, 0], mds[:, 1])
    plt.title(title)
    plt.show()



"""
Training settings

"""
num_epochs = 10
tmp_i=0

#set best epoch as global variable
best_epoch = 0

check_every = 1
b_size = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_val_acc=0

lr_vis_phy = 0.000005  #0.00001
lr_mmtransformer = 0.0005 #0.0001




experiment = Experiment(
  api_key="--------------------",
  project_name="MT-PKDOT",
  workspace="---------------",
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


videos_root = 'Path to video folder'


preprocess_train = transforms.Compose([
    ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
    transforms.Resize(112),  # image batch, resize smaller edge to 299
    # transforms.CenterCrop(112),  # image batch, center crop to square 299x299
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
])


preprocess_test = transforms.Compose([
    ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
    transforms.Resize(112),  # image batch, resize smaller edge to 299
    # transforms.CenterCrop(112),  # image batch, center crop to square 299x299
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])





def denormalize(video_tensor):
    inverse_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    return (inverse_normalize(video_tensor) * 255.).type(torch.uint8).permute(0, 2, 3, 1).numpy()




def train_self_distil_wadn(train_file_path, val_file_path,fold_num):
    train_annotation_file = os.path.join(videos_root, train_file_path)
    val_annotation_file = os.path.join(videos_root, val_file_path)
    model_save_path_self_distil = os.path.join(os.getcwd(), 'self_distil_from_teacher_vis_'+str(fold_num)+'.pth')


    best_val_acc = 0
    best_epoch = 0
    min_align_loss = 10
    criterion = nn.CrossEntropyLoss()


    # criterion_align = MMD_loss()
    # criterion_align = MSELoss()
    # criterion_align = nn.CosineSimilarity(dim=1, eps=1e-6)  
    criterion_align = nn.CosineEmbeddingLoss()





    vis_phy_model_teacher = VIS_PHY_MODEL_CAM().to(device=device)
    mm_transformer_teacher = MultimodalTransformer(visual_dim=512, physiological_dim=512, num_heads=2, hidden_dim=512, num_layers=2, num_classes=2).to(device=device)

    

    #freeze the weights of the teacher
    for param in vis_phy_model_teacher.parameters():
        param.requires_grad = False

    
    for param in mm_transformer_teacher.parameters():
        param.requires_grad = True

    
    adapter_net_vis = AdapterNetwork().to(device=device)
    adapter_net_phy = AdapterNetwork().to(device=device)
    c_head_vis = ClassifierHead(input_dim=512, num_classes=2).to(device=device)
    c_head_phy = ClassifierHead(input_dim=512, num_classes=2).to(device=device)

    # add parameters of the classifier heads to the optimizer anv optimizer


    optimizer_anv = optim.Adam(adapter_net_vis.parameters() , lr=0.005)
    optimizer_anp = optim.Adam(adapter_net_phy.parameters(), lr=0.005)
    c_head_phy_optimizer = optim.Adam(c_head_phy.parameters(), lr=0.005)
    c_head_vis_optimizer = optim.Adam(c_head_vis.parameters(), lr=0.005)
    optimizer_mmtransformer = optim.Adam(mm_transformer_teacher.parameters(), lr=0.0001)


    vis_phy_model_teacher.eval()
    mm_transformer_teacher.train()
    c_head_phy.train()
    c_head_vis.train()
    adapter_net_vis.train()
    adapter_net_phy.train()

    train_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=train_annotation_file,
        num_segments=10,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess_train,
        test_mode=False)

    val_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=val_annotation_file,
        num_segments=10,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess_test,
        test_mode=True)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=b_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True)


    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=b_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True)


    with experiment.train():

        for epoch in tqdm(range(num_epochs), desc='Epochs'):
            
            
            running_loss = 0.0
            for i,(spec_2d,video_batch, labels) in enumerate(train_dataloader,0):
                
                
                optimizer_mmtransformer.zero_grad()
                # vis_optimizer.zero_grad()
                adapter_net_vis.zero_grad()
                adapter_net_phy.zero_grad()
                c_head_phy.zero_grad()
                c_head_vis.zero_grad()

                #prepare data
                video_batch=video_batch.permute(0, 2, 1, 3, 4)
                video_batch = video_batch.to(device)
                labels = labels.to(device)

                
                #get teacher embeddings
                # with torch.no_grad():
                vis_feats_teacher, phy_feats_teacher = vis_phy_model_teacher.model_out_feats(video_batch,spec_2d)
                vis_feats_teacher = vis_feats_teacher.unsqueeze(1)
                phy_feats_teacher = phy_feats_teacher.unsqueeze(1)
                outs, intermediate_outs_teacher = mm_transformer_teacher(vis_feats_teacher, phy_feats_teacher)

                vis_feats_adapted = adapter_net_vis(vis_feats_teacher)
                vis_feats_adapted = vis_feats_adapted.squeeze(1)
                vis_feats_teacher = vis_feats_teacher.squeeze(1)

                phy_feats_adapted = adapter_net_phy(phy_feats_teacher)
                phy_feats_adapted = phy_feats_adapted.squeeze(1)
                phy_feats_teacher = phy_feats_teacher.squeeze(1)



                vis_outs_final = c_head_vis(vis_feats_adapted)
                phy_outs_final = c_head_phy(phy_feats_adapted)

                #calculate the loss with vis and phy heads using criterion 
                vis_loss = criterion(vis_outs_final, labels)
                phy_loss = criterion(phy_outs_final, labels)

                head_loss = 0.5*vis_loss + 0.5*phy_loss 





 

                'align loss with adapters'

                align_loss = criterion_align(vis_feats_adapted, intermediate_outs_teacher, torch.ones(vis_feats_adapted.shape[0]).to(device)) + criterion_align(phy_feats_adapted, intermediate_outs_teacher, torch.ones(phy_feats_adapted.shape[0]).to(device))


                
                'align loss for directly distilling the features without adaptor net'
                # align_loss = criterion_align(vis_feats_teacher, intermediate_outs_teacher, torch.ones(vis_feats_teacher.shape[0]).to(device)) + criterion_align(phy_feats_teacher, intermediate_outs_teacher, torch.ones(phy_feats_teacher.shape[0]).to(device))
                # align_loss = criterion_align(vis_feats_teacher, intermediate_outs_teacher) + criterion_align(phy_feats_teacher, intermediate_outs_teacher)
                # align_loss = criterion_align(vis_feats_teacher, intermediate_outs_teacher) + criterion_align(phy_feats_teacher, intermediate_outs_teacher)

                
                gt_loss = criterion(outs, labels) 

                total_loss= 0.5*align_loss + 0.5*(gt_loss + head_loss)
                       
                
                total_loss.backward()
       
                optimizer_anv.step()
                optimizer_anp.step()
                c_head_phy_optimizer.step()
                c_head_vis_optimizer.step()
                optimizer_mmtransformer.step()
                # optimizer_vis_phy.step()


                running_loss += total_loss.item()
       
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f"[{epoch + 1}, {i + 1}] align_loss_vis: {running_loss / 100:.3f}")
                    running_loss = 0.0
            

            #print alignment loss for the batch
            print("Alignment loss: ", running_loss/i) 
            if epoch % check_every == 0:
                val_acc, vis_acc, phy_acc, val_loss, vis_loss, phy_loss = validate_mmtransformer_dmwl_self_distill(vis_phy_model_teacher,mm_transformer_teacher,adapter_net_vis,adapter_net_phy,c_head_vis,c_head_vis,criterion,val_dataloader, device)
                print( "Validation accuracy: ", val_acc)

                acc_values= [vis_acc, phy_acc, val_acc]
                max_val_acc = max(acc_values)
                max_val_acc_index = acc_values.index(max_val_acc)
                print('Max Validation accuracy: ', max_val_acc)
                print('Max Validation accuracy index: ', max_val_acc_index)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    #add fold number to the model save path

                    model_save_name_bb= 'teacher_after_self_distill_bb_fold'+str(fold_num)+'.pth'  
                    model_save_name_mmtransformer= 'teacher_after_self_distill_mm_fold'+str(fold_num)+'.pth'
                    model_save_name_adn_vis= 'teacher_after_self_distill_adn_vis_fold'+str(fold_num)+'.pth'
                    model_save_name_adn_phy= 'teacher_after_self_distill_adn_phy_fold'+str(fold_num)+'.pth'
                    model_save_name_c_head_vis= 'teacher_after_self_distill_c_head_vis_fold'+str(fold_num)+'.pth'
                    model_save_name_c_head_phy= 'teacher_after_self_distill_c_head_phy_fold'+str(fold_num)+'.pth'

                    model_save_path_bb = os.path.join(os.getcwd(), model_save_name_bb)
                    model_save_path_mmtransformer = os.path.join(os.getcwd(), model_save_name_mmtransformer)
                    model_save_path_adn_vis = os.path.join(os.getcwd(), model_save_name_adn_vis)
                    model_save_path_adn_phy = os.path.join(os.getcwd(), model_save_name_adn_phy)
                    model_save_path_c_head_vis = os.path.join(os.getcwd(), model_save_name_c_head_vis)
                    model_save_path_c_head_phy = os.path.join(os.getcwd(), model_save_name_c_head_phy)


                    torch.save(vis_phy_model_teacher.state_dict(), model_save_path_bb)
                    torch.save(mm_transformer_teacher.state_dict(), model_save_path_mmtransformer)
                    torch.save(adapter_net_vis.state_dict(), model_save_path_adn_vis)
                    torch.save(adapter_net_phy.state_dict(), model_save_path_adn_phy)
                    torch.save(c_head_vis.state_dict(), model_save_path_c_head_vis)
                    torch.save(c_head_phy.state_dict(), model_save_path_c_head_phy)

                    print('Validation_accuracy: ', val_acc)
                    print('Best model saved at epoch: ', epoch+1)
                    best_epoch = epoch+1

            
            




    print("Finished Alignment with Self Distillation")

    avg_train_loss = running_loss / len(train_dataloader)

    print(f'Training_Align_Loss: {avg_train_loss}')

    return avg_train_loss, best_val_acc






#*************************************************************************************************

def train_student_mt(train_file_path, val_file_path,fold_num):
    vis_count=0
    phy_count=0
    joint_count=0
    train_annotation_file = os.path.join(videos_root, train_file_path)
    val_annotation_file = os.path.join(videos_root, val_file_path)
    model_save_path_bb = os.path.join(os.getcwd(), 'model_best_bb_ot_visonly_student_mtpkdot_wocent'+str(fold_num)+'.pth')
    model_save_path_mmtransformer = os.path.join(os.getcwd(), 'model_best_mmtrans_ot_visonly_student_mtpkdot_wocent'+str(fold_num)+'.pth')




    best_val_acc = 0
    best_epoch = 0
    sinkhorn_loss_func = SamplesLoss("sinkhorn", p=2, blur=0.1)


    criterion = nn.CrossEntropyLoss()


    vis_model_student=VIS_MODEL(fold_num).to(device=device)
    
    saved_path_root='/home/livia/work/Biovid/PartB/dmwl_ot/model_saved_otpi_771'

        
    # vis_model_student.load_state_dict(torch.load(vis_model_student_load_path))
    

    # vis_model_student = VIS_PHY_MODEL_CAM().to(device=device)
    mm_transformer_student = MultimodalTransformer(visual_dim=512, physiological_dim=512, num_heads=2, hidden_dim=512, num_layers=2, num_classes=2)

    mm_transformer_student = mm_transformer_student.to(device=device)



    transform_net = TransformNet().to(device=device)

    #freeze the weights of the transformation network
    for param in transform_net.parameters():
        param.requires_grad = False


        
    adapter_net_vis = AdapterNetwork().to(device=device)
    adapter_net_phy = AdapterNetwork().to(device=device)
    c_head_vis = ClassifierHead(input_dim=512, num_classes=2).to(device=device)
    c_head_phy = ClassifierHead(input_dim=512, num_classes=2).to(device=device)
    

    vis_optimizer = optim.Adam(vis_model_student.parameters(), lr=lr_vis_phy)
    mmtransformer_optimizer = optim.Adam(mm_transformer_student.parameters(), lr=lr_mmtransformer)

    mmtransformer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(mmtransformer_optimizer, mode='max', factor=0.01, patience=5,verbose=True)



    vis_phy_model_teacher = VIS_PHY_MODEL_CAM().to(device=device)
    mm_transformer_teacher = MultimodalTransformer(visual_dim=512, physiological_dim=512, num_heads=2, hidden_dim=512, num_layers=2, num_classes=2).to(device=device)


    '''
    Load all modules of teacher models per fold
    
    '''

    model_load_path_bb = os.path.join(os.getcwd(), 'teacher_after_self_distill_bb_fold'+str(fold_num)+'.pth')
    model_load_path_mmtransformer = os.path.join(os.getcwd(), 'teacher_after_self_distill_mm_fold'+str(fold_num)+'.pth')
    model_load_path_adn_vis = os.path.join(os.getcwd(), 'teacher_after_self_distill_adn_vis_fold'+str(fold_num)+'.pth')
    model_load_path_adn_phy = os.path.join(os.getcwd(), 'teacher_after_self_distill_adn_phy_fold'+str(fold_num)+'.pth')
    model_load_path_c_head_vis = os.path.join(os.getcwd(), 'teacher_after_self_distill_c_head_vis_fold'+str(fold_num)+'.pth')
    model_load_path_c_head_phy = os.path.join(os.getcwd(), 'teacher_after_self_distill_c_head_phy_fold'+str(fold_num)+'.pth')

    transform_net_load_path = os.path.join(os.getcwd(), 'saved_weights_DMWL/transform_net_saved_fold_'+str(fold_num)+'.pth')
    
    vis_phy_model_teacher.load_state_dict(torch.load(model_load_path_bb))
    mm_transformer_teacher.load_state_dict(torch.load(model_load_path_mmtransformer))
    adapter_net_vis.load_state_dict(torch.load(model_load_path_adn_vis))
    adapter_net_phy.load_state_dict(torch.load(model_load_path_adn_phy))
    c_head_vis.load_state_dict(torch.load(model_load_path_c_head_vis))
    c_head_phy.load_state_dict(torch.load(model_load_path_c_head_phy))
    transform_net.load_state_dict(torch.load(transform_net_load_path))

    


    #freeze the weights of the teacher
    for param in vis_phy_model_teacher.parameters():
        param.requires_grad = False
    
    for param in mm_transformer_teacher.parameters():
        param.requires_grad = False

    for param in c_head_vis.parameters():
        param.requires_grad = False

    for param in c_head_phy.parameters():
        param.requires_grad = False

    for param in adapter_net_vis.parameters():
        param.requires_grad = False

    for param in adapter_net_phy.parameters():
        param.requires_grad = False
    
    vis_phy_model_teacher.eval()
    mm_transformer_teacher.eval()
    transform_net.eval()
    c_head_vis.eval()
    c_head_phy.eval()
    adapter_net_vis.eval()
    adapter_net_phy.eval()


    train_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=train_annotation_file,
        num_segments=5,
        frames_per_segment=2,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess_train,
        test_mode=False)

    val_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=val_annotation_file,
        num_segments=10,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess_test,
        test_mode=True)



    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=b_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True)


    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=b_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True)







    with experiment.train():
        for epoch in tqdm(range(num_epochs), desc='Epochs'):
            
            vis_model_student.vis_model.eval()
            mm_transformer_student.train()
            
            # classifier_m.train()
            
            running_loss = 0.0
            correct = 0
            total = 0
            for i,(spec_2d,video_batch, labels) in enumerate(train_dataloader,0):
                
                
                mmtransformer_optimizer.zero_grad()
                vis_optimizer.zero_grad()

                #prepare data
                video_batch=video_batch.permute(0, 2, 1, 3, 4)
                video_batch = video_batch.to(device)
                labels = labels.to(device)



                '''
                
                Get teacher embeddings
                1) Get multimodal teacher
                2) Get get vis adapted features, and phy adapted features
                3) Get the outputs from the classifier heads to calculate confidence
                
                '''
                
                #get teacher embeddings and classifier outputs for selection
                with torch.no_grad():
                    vis_feats_teacher, phy_feats_teacher = vis_phy_model_teacher.model_out_feats(video_batch,spec_2d)
                    vis_feats_teacher = vis_feats_teacher.unsqueeze(1)
                    phy_feats_teacher = phy_feats_teacher.unsqueeze(1)
                    joint_out, joint_outs_teacher = mm_transformer_teacher(vis_feats_teacher, phy_feats_teacher)


                    vis_feats_adapted = adapter_net_vis(vis_feats_teacher)
                    vis_feats_adapted = vis_feats_adapted.squeeze(1)

                    phy_feats_adapted = adapter_net_phy(phy_feats_teacher)
                    phy_feats_adapted = phy_feats_adapted.squeeze(1)

                    vis_outs_final = c_head_vis(vis_feats_adapted)
                    phy_outs_final = c_head_phy(phy_feats_adapted)

                    #calculate the loss with joint and vis and phy heads using criterion
                    vis_loss = criterion(vis_outs_final, labels)
                    phy_loss = criterion(phy_outs_final, labels)
                    joint_out_loss = criterion(joint_out, labels)

                    #select the output with minimum loss
                    loss_values = [vis_loss, phy_loss, joint_out_loss]
                    min_loss = min(loss_values)
                    min_loss_index = loss_values.index(min_loss)
                    if min_loss_index == 0:
                        vis_count= vis_count+1
                        intermediate_outs_teacher = vis_feats_adapted
                    elif min_loss_index == 1:
                        intermediate_outs_teacher = phy_feats_adapted
                        phy_count = phy_count+1
                    elif min_loss_index == 2:
                        intermediate_outs_teacher = joint_outs_teacher
                        joint_count = joint_count+1





                    #sort intermediate_outs_teacher based on class labels where all the samples of the same class are together
                    # intermediate_outs_teacher = intermediate_outs_teacher[np.argsort(labels.detach().cpu().numpy())]


                
                '''
                Get student intermediate embeddings

                
                '''



                #get student embeddings

                vis_feats, outs_vo = vis_model_student.model_out(video_batch)


                recon_phy_feats = transform_net(vis_feats.detach())
                vis_feats = vis_feats.unsqueeze(1)
                recon_phy_feats = recon_phy_feats.unsqueeze(1)


                outs, intermediate_outs_student = mm_transformer_student(vis_feats, recon_phy_feats)

                cosine_similarity_matrix_teacher = cosine_similarity_matrix_transpose_torch(intermediate_outs_teacher)

                
                '''
                Uncomment the following to plot the similarity matrices 
                '''


                # #plot similarity matrices where i is 1
                # if i == 1 and fold_num == '3':
                #     if epoch == 0 or epoch == 10:
                #         plot_simmat(cosine_similarity_matrix_teacher.detach().cpu().numpy(), cosine_similarity_matrix_student.detach().cpu().numpy(),epoch,fold_num)


                
                # cosine_similarity_matrix_student = cosine_similarity_matrix_transpose(intermediate_outs_student.detach().cpu().numpy())
                cosine_similarity_matrix_student = cosine_similarity_matrix_transpose_torch(intermediate_outs_student)

                # ensure that the similarity matrices are non negative
                cosine_similarity_matrix_teacher_nonneg = cosine_similarity_matrix_teacher - cosine_similarity_matrix_teacher.min()
                cosine_similarity_matrix_student_nonneg = cosine_similarity_matrix_student - cosine_similarity_matrix_student.min()

            


                #select topk values from the similiarity matrix where the similarity is minimum from the teacher
                topk = 30
                topk_indices = np.argpartition(cosine_similarity_matrix_teacher.detach().cpu().numpy(), topk, axis=1)[:, :topk]
                cosine_similarity_matrix_teacher_tk = cosine_similarity_matrix_teacher_nonneg[np.arange(cosine_similarity_matrix_teacher_nonneg.shape[0])[:, None], topk_indices]
                cosine_similarity_matrix_student_tk = cosine_similarity_matrix_student_nonneg[np.arange(cosine_similarity_matrix_student_nonneg.shape[0])[:, None], topk_indices]
                
                
                #calculate the centroid of the cosine similarity matrix_teacher_tk and cosine_similarity_matrix_student_tk
                
                centroid_teacher = torch.mean(cosine_similarity_matrix_teacher, axis=0)
                centroid_student = torch.mean(cosine_similarity_matrix_student, axis=0)

                #calculate l2 norm between the centroids
                l2_norm_centroid = torch.norm(centroid_teacher - centroid_student, p=2)


  
                

                '''
                OT between two simalrity matrices (cosine sim matrices)
                ''' 
   
                sinkhorn_loss = sinkhorn_loss_func(cosine_similarity_matrix_teacher_tk, cosine_similarity_matrix_student_tk)
                


                gt_loss = criterion(outs, labels)

                # print(sinkhorn_loss)

                #make alpha learnable parameter
                alpha = 0.8
                beta = 0.2
                gamma = 0.2

                # total_loss= (alpha*gt_loss) + (beta * sinkhorn_loss) + (gamma * l2_norm_centroid)   #with centroid loss

                total_loss= (alpha*gt_loss) + (beta * sinkhorn_loss)  #without centroid loss


                
                
                total_loss.backward()

                mmtransformer_optimizer.step()

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
            experiment.log_metric('Loss', train_loss,epoch= epoch)
            experiment.log_metric('Accuracy', train_accuracy ,epoch= epoch)
            print("Sinkhorn loss: ", sinkhorn_loss.item())
            # print("Frobenius norm: ", frobenius_norm)
            print("total loss: ", total_loss.item())
            print("Best validation accuracy: ", best_val_acc, "at epoch: ", best_epoch)
            # last_lr=scheduler.get_last_lr()
            # experiment.log_metric('Learning Rate', last_lr,epoch= epoch)





            if epoch % check_every == 0:
                val_acc, val_loss = validate_mmtransformer_dmwl_wtn(vis_model_student,mm_transformer_student,transform_net, val_dataloader, criterion, device)
                print( "Validation accuracy: ", val_acc)
                # experiment.log_metric('Val Accuracy', val_acc,epoch= epoch)
                # experiment.log_metric('Val Loss', val_loss,epoch= epoch)
                # mmtransformer_scheduler.step(val_acc)   
                # current_lr = mmtransformer_optimizer.param_groups[0]['lr']
                # experiment.log_metric('Learning Rate', current_lr,epoch= epoch)
                # print('Current learning rate: ', current_lr)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # model_save_path_bb = os.path.join(os.getcwd(), 'model_best_dmwl_bb_wtn_ot.pth')
                    # model_save_path_mmtransformer = os.path.join(os.getcwd(), 'model_best_dmwl_mmtrans_wtn_ot.pth')

                    torch.save(vis_model_student.state_dict(), model_save_path_bb)
                    torch.save(mm_transformer_student.state_dict(), model_save_path_mmtransformer)  
                    print('Validation_accuracy: ', val_acc)
                    print('Best model saved at epoch: ', epoch+1)
                    best_epoch = epoch+1


    print("Finished Training")

    train_accuracy = 100 * correct / total
    avg_train_loss = running_loss / len(train_dataloader)
    print(f'Training accuracy: {train_accuracy}%')
    print(f'Training loss: {avg_train_loss}')

    print("Best model saved at epoch: ", best_epoch)
    print("Best validation accuracy: ", best_val_acc)
    return best_val_acc



def get_tsne_data(train_file_path, val_file_path,fold_num):
    train_annotation_file = os.path.join(videos_root, train_file_path)
    val_annotation_file = os.path.join(videos_root, val_file_path)
    


    vis_phy_model_teacher = VIS_PHY_MODEL_CAM().to(device=device)
    mm_transformer_teacher = MultimodalTransformer(visual_dim=512, physiological_dim=512, num_heads=2, hidden_dim=512, num_layers=2, num_classes=2).to(device=device)


    #freeze the weights of the teacher
    for param in vis_phy_model_teacher.parameters():
        param.requires_grad = False

    
    for param in mm_transformer_teacher.parameters():
        param.requires_grad = False

    
    adapter_net_vis = AdapterNetwork().to(device=device)
    adapter_net_phy = AdapterNetwork().to(device=device)
    c_head_vis = ClassifierHead(input_dim=512, num_classes=2).to(device=device)
    c_head_phy = ClassifierHead(input_dim=512, num_classes=2).to(device=device)

    #load the teacher models
    model_load_path_bb = os.path.join(os.getcwd(), 'teacher_after_self_distill_bb_fold'+str(fold_num)+'.pth')
    model_load_path_mmtransformer = os.path.join(os.getcwd(), 'teacher_after_self_distill_mm_fold'+str(fold_num)+'.pth')
    model_load_path_adn_vis = os.path.join(os.getcwd(), 'teacher_after_self_distill_adn_vis_fold'+str(fold_num)+'.pth')
    model_load_path_adn_phy = os.path.join(os.getcwd(), 'teacher_after_self_distill_adn_phy_fold'+str(fold_num)+'.pth')
    model_load_path_c_head_vis = os.path.join(os.getcwd(), 'teacher_after_self_distill_c_head_vis_fold'+str(fold_num)+'.pth')
    model_load_path_c_head_phy = os.path.join(os.getcwd(), 'teacher_after_self_distill_c_head_phy_fold'+str(fold_num)+'.pth')

    vis_phy_model_teacher.load_state_dict(torch.load(model_load_path_bb))
    mm_transformer_teacher.load_state_dict(torch.load(model_load_path_mmtransformer))
    adapter_net_vis.load_state_dict(torch.load(model_load_path_adn_vis))
    adapter_net_phy.load_state_dict(torch.load(model_load_path_adn_phy))
    c_head_vis.load_state_dict(torch.load(model_load_path_c_head_vis))
    c_head_phy.load_state_dict(torch.load(model_load_path_c_head_phy))






    vis_phy_model_teacher.eval()
    mm_transformer_teacher.train()
    c_head_phy.eval()
    c_head_vis.eval()
    adapter_net_vis.eval()
    adapter_net_phy.eval()

    train_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=train_annotation_file,
        num_segments=10,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess_train,
        test_mode=False)

    val_dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=val_annotation_file,
        num_segments=10,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess_test,
        test_mode=True)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=b_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True)


    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=b_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True)

    phy_feats_teacher_stack = torch.tensor([]).to(device)
    vis_feats_teacher_stack = torch.tensor([]).to(device)
    intermediate_outs_teacher_stack = torch.tensor([]).to(device)
    phy_feats_teacher_stack_adapted = torch.tensor([]).to(device)
    vis_feats_teacher_stack_adapted = torch.tensor([]).to(device)
    labels_stack = torch.tensor([]).to(device)

    with experiment.train():


        for i,(spec_2d,video_batch, labels) in enumerate(train_dataloader,0):
            # original_order = np.argsort(labels)
            
            

            #prepare data
            video_batch=video_batch.permute(0, 2, 1, 3, 4)
            video_batch = video_batch.to(device)
            labels = labels.to(device)



            
            
            #get teacher embeddings
            # with torch.no_grad():
            vis_feats_teacher, phy_feats_teacher = vis_phy_model_teacher.model_out_feats(video_batch,spec_2d)
            vis_feats_teacher = vis_feats_teacher.unsqueeze(1)
            phy_feats_teacher = phy_feats_teacher.unsqueeze(1)
            outs, intermediate_outs_teacher = mm_transformer_teacher(vis_feats_teacher, phy_feats_teacher)

            


            #stack the intermediate_outs_teacher and vis_feats_teacher and phy_feats_teacher before the adapter
            intermediate_outs_teacher_stack = torch.cat((intermediate_outs_teacher_stack, intermediate_outs_teacher), dim=0)
            vis_feats_teacher_stack = torch.cat((vis_feats_teacher_stack, vis_feats_teacher), dim=0)
            phy_feats_teacher_stack = torch.cat((phy_feats_teacher_stack, phy_feats_teacher), dim=0)
            labels_stack = torch.cat((labels_stack, labels), dim=0)












            vis_feats_adapted = adapter_net_vis(vis_feats_teacher)
            vis_feats_adapted = vis_feats_adapted.squeeze(1)
            vis_feats_teacher = vis_feats_teacher.squeeze(1)

            phy_feats_adapted = adapter_net_phy(phy_feats_teacher)
            phy_feats_adapted = phy_feats_adapted.squeeze(1)
            phy_feats_teacher = phy_feats_teacher.squeeze(1)


            #stack the intermediate_outs_teacher and vis_feats_teacher and phy_feats_teacher after the adapter
            vis_feats_teacher_stack_adapted = torch.cat((vis_feats_teacher_stack_adapted, vis_feats_adapted), dim=0)
            phy_feats_teacher_stack_adapted = torch.cat((phy_feats_teacher_stack_adapted, phy_feats_adapted), dim=0)

            vis_outs_final = c_head_vis(vis_feats_adapted)
            phy_outs_final = c_head_phy(phy_feats_adapted)
    
    vis_feats_teacher_stack=vis_feats_teacher_stack.squeeze(1)
    phy_feats_teacher_stack=phy_feats_teacher_stack.squeeze(1)

    #save the stacked features in files
    vis_feats_teacher_stack_path = os.path.join(os.getcwd(), 'vis_feats_teacher_stack_fold'+str(fold_num)+'.npy')
    phy_feats_teacher_stack_path = os.path.join(os.getcwd(), 'phy_feats_teacher_stack_fold'+str(fold_num)+'.npy')
    intermediate_outs_teacher_stack_path = os.path.join(os.getcwd(), 'intermediate_outs_teacher_stack_fold'+str(fold_num)+'.npy')
    vis_feats_teacher_stack_adapted_path = os.path.join(os.getcwd(), 'vis_feats_teacher_stack_adapted_fold'+str(fold_num)+'.npy')
    phy_feats_teacher_stack_adapted_path = os.path.join(os.getcwd(), 'phy_feats_teacher_stack_adapted_fold'+str(fold_num)+'.npy')
    labels_stack_path = os.path.join(os.getcwd(), 'labels_stack_fold'+str(fold_num)+'.npy')

    np.save(vis_feats_teacher_stack_path, vis_feats_teacher_stack.cpu().detach().numpy())
    np.save(phy_feats_teacher_stack_path, phy_feats_teacher_stack.cpu().detach().numpy())
    np.save(intermediate_outs_teacher_stack_path, intermediate_outs_teacher_stack.cpu().detach().numpy())
    np.save(vis_feats_teacher_stack_adapted_path, vis_feats_teacher_stack_adapted.cpu().detach().numpy())   
    np.save(phy_feats_teacher_stack_adapted_path, phy_feats_teacher_stack_adapted.cpu().detach().numpy())
    np.save(labels_stack_path, labels_stack.cpu().detach().numpy())





    # return vis_feats_teacher_stack, phy_feats_teacher_stack, intermediate_outs_teacher_stack, vis_feats_teacher_stack_adapted, phy_feats_teacher_stack_adapted, labels_stack



 

def plot_tsne_graphs_with_labels():
    # Load the features and labels
    visf = np.load('vis_feats_teacher_stack_fold1.npy')
    phyf = np.load('phy_feats_teacher_stack_fold1.npy')
    jointf = np.load('intermediate_outs_teacher_stack_fold1.npy')   
    visfa = np.load('vis_feats_teacher_stack_adapted_fold1.npy')
    phyfa = np.load('phy_feats_teacher_stack_adapted_fold1.npy')
    labels = np.load('labels_stack_fold1.npy')

    #select only 1000 samples

    total_samples = 1000
    visf = visf[:total_samples]
    phyf = phyf[:total_samples]
    jointf = jointf[:total_samples]
    visfa = visfa[:total_samples]
    phyfa = phyfa[:total_samples]
    labels = labels[:total_samples]

    features1 = np.vstack([visf, phyf, jointf])
    markers = ['o', '^', 's', '*', 'x', '+']
    unique_labels = np.unique(labels)



    # Compute t-SNE embeddings for the first set of features
    tsne1 = TSNE(n_components=3, random_state=42)
    X_tsne1 = tsne1.fit_transform(features1)

    # Create a 3D plot for the first set of features
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    labels1 = np.tile(labels, 3)  # Adjust labels size if necessary

    # Plot each point in the t-SNE embeddings
    scatter1 = ax1.scatter(X_tsne1[:, 0], X_tsne1[:, 1], X_tsne1[:, 2], marker = '*', c=labels1, cmap='viridis', alpha=0.6)

    # Adding color bar to show the mapping of colors
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Labels')

    # Set labels for axes
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.set_zlabel('t-SNE 3')

    plt.show()




    # Stack the feature vectors for the second set of features
    features2 = np.vstack([visfa, phyfa, jointf])

    # Compute t-SNE embeddings for the second set of features
    tsne2 = TSNE(n_components=3, random_state=42)
    X_tsne2 = tsne2.fit_transform(features2)

    # Create a 3D plot for the second set of features
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    labels2 = np.tile(labels, 3)  # Adjust labels size if necessary

    # Plot each point in the t-SNE embeddings
    scatter2 = ax2.scatter(X_tsne2[:, 0], X_tsne2[:, 1], X_tsne2[:, 2], c=labels2, cmap='viridis', alpha=0.6)

    # Adding color bar to show the mapping of colors
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Labels')

    # Set labels for axes
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_zlabel('t-SNE 3')

    plt.show()






    # # Stack the feature vectors for the first graph
    # features1 = np.vstack([visf, phyf, jointf])
    # labels1 = np.concatenate([labels for _ in range(3)])  # Assuming labels are the same for all features
    
    # # Stack the feature vectors for the second graph
    # features2 = np.vstack([visfa, phyfa, jointf])
    # labels2 = np.concatenate([labels for _ in range(3)])  # Assuming labels are the same for all features
    
    # # Compute t-SNE for the first set of features
    # tsne_features1 = TSNE(n_components=2, random_state=42).fit_transform(features1)
    
    # # Compute t-SNE for the second set of features
    # tsne_features2 = TSNE(n_components=2, random_state=42).fit_transform(features2)




    
    # # Plotting the first graph
    # plt.figure(figsize=(14, 6))
    # plt.subplot(1, 2, 1)
    # scatter = plt.scatter(tsne_features1[:, 0], tsne_features1[:, 1], c=labels1, cmap='viridis', alpha=0.6)
    # plt.title('t-SNE: visf, phyf, jointf')
    # plt.colorbar(scatter)
    
    # # Plotting the second graph
    # plt.subplot(1, 2, 2)
    # scatter = plt.scatter(tsne_features2[:, 0], tsne_features2[:, 1], c=labels2, cmap='viridis', alpha=0.6)
    # plt.title('t-SNE: visfa, phyfa, jointf')
    # plt.colorbar(scatter)
    
    plt.show()


    


def train_self_distil_kfold():
    k=5
    #create list of size 
    align_loss_list = []
    best_val_acc_list = []


    for i in range(k):
        print("Fold: ", i+1)
        train_file_path = 'fold_'+ str(i+1) +'_train.txt'
        val_file_path = 'fold_'+ str(i+1) +'_test.txt'
        fold_num = str(i+1)
        #call only for fold 3 and 4

        align_loss, best_val_acc=train_self_distil_wadn(train_file_path, val_file_path,fold_num)
        align_loss_list.append(align_loss)
        best_val_acc_list.append(best_val_acc)

    print("Best align loss for each fold: ", align_loss_list)
    print("Average align loss: ", sum(align_loss_list)/k)
    print("Best validation accuracy for each fold: ", best_val_acc_list)
    print("Average best validation accuracy: ", sum(best_val_acc_list)/k)




def train_student_mt_k_fold():
    k=5
    #create list of size 
    best_acc_list = []


    for i in range(k):
        print("Fold: ", i+1)
        train_file_path = 'fold_'+ str(i+1) +'_train.txt'
        val_file_path = 'fold_'+ str(i+1) +'_test.txt'
        fold_num = str(i+1)
        #call only for fold 3 and 4
        best_fold_acc=train_student_mt(train_file_path, val_file_path,fold_num)
        best_acc_list.append(best_fold_acc)

    print("Best validation accuracy for each fold: ", best_acc_list)
    print("Average best validation accuracy: ", sum(best_acc_list)/k)


def train_student_mt_1_fold():
    k=2
    print("Fold: ", k+1)
    train_file_path = 'fold_'+ str(k+1) +'_train.txt'
    val_file_path = 'fold_'+ str(k+1) +'_test.txt'
    fold_num = str(k+1)
    #call only for fold 3 and 4
    best_fold_acc=train_student_mt(train_file_path, val_file_path,fold_num)
    # best_acc_list.append(best_fold_acc)


'''
Extra Check functions

'''
def test_k_fold():
    saved_path_root='/home/livia/work/Biovid/PartB/DMWL_TBIOM/MT-PKDOT_saved_weights_final'
    k=5
    #create list of size 
    best_acc_list = [] 

    for i in range(k):
        print("Fold: ", i+1)
        train_file_path = 'fold_'+ str(i+1) +'_train.txt'
        val_file_path = 'fold_'+ str(i+1) +'_test.txt'
        fold_num = str(i+1)

        transform_net = TransformNet().to(device=device)
        val_annotation_file = os.path.join(videos_root, val_file_path)

        #freeze the weights of the transformation network
        for param in transform_net.parameters():
            param.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        val_dataset = VideoFrameDataset(
            root_path=videos_root,
            annotationfile_path=val_annotation_file,
            num_segments=10,
            frames_per_segment=1,
            imagefile_template='img_{:05d}.jpg',
            transform=preprocess_test,
            test_mode=True)

        val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=b_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)


        #load vis_model and mmtransformer

        vis_model_student=VIS_MODEL(fold_num).to(device=device)

        mm_transformer_student = MultimodalTransformer(visual_dim=512, physiological_dim=512, num_heads=2, hidden_dim=512, num_layers=2, num_classes=2).to(device=device)
        vis_model_student_load_path= os.path.join(saved_path_root,'model_best_bb_ot_visonly_student_mtpkdot'+str(fold_num)+'.pth')
        mm_transformer_student_load_path = os.path.join(saved_path_root,'model_best_mmtrans_ot_visonly_student_mtpkdot'+str(fold_num)+'.pth')
        
        vis_model_student.load_state_dict(torch.load(vis_model_student_load_path))
        mm_transformer_student.load_state_dict(torch.load(mm_transformer_student_load_path))
        best_fold_acc, val_loss = validate_mmtransformer_dmwl_wtn(vis_model_student,mm_transformer_student,transform_net, val_dataloader, criterion, device)
        best_acc_list.append(best_fold_acc)


        print("Best validation accuracy for each fold: ", best_acc_list)
        print("Average best validation accuracy: ", sum(best_acc_list)/k)
        mean_accuracy = np.mean(best_acc_list)
        std_error = np.std(best_acc_list) / np.sqrt(k)

        print("Mean accuracy: ", mean_accuracy)
        print("Standard error: ", std_error)


def train_1_fold():
    fold_num = 1
    train_file_path = 'annotations_filtered_peak_2_train.txt'
    val_file_path = 'annotations_filtered_peak_2_test.txt'
    best_fold_acc=train_student_mt(train_file_path, val_file_path,fold_num)  



if __name__ == '__main__':
    # train_align_kfold()
    train_student_mt_k_fold()
    # test_k_fold()
    # train_1_fold()
    # train_self_distil_kfold()   


