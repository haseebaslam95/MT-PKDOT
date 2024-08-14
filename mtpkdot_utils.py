import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns





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

class CustomLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(CustomLoss, self).__init__()
        self.weight = weight

    def forward(self, frob_loss):
        return torch.tensor(frob_loss * self.weight, requires_grad=True)




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


def plot_tsne(data, labels, random_state=0):
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=random_state)
    tsne_results = tsne.fit_transform(data)

    # Create a DataFrame to make plotting easier
    df = pd.DataFrame(data = {
        'x': tsne_results[:,0],
        'y': tsne_results[:,1],
        'label': labels
    })

    # Set the figure size
    plt.figure(figsize=(10,10))

    # Use seaborn to create a scatterplot with different colors for each label
    sns.scatterplot(
        x="x", y="y",
        hue="label",
        palette=sns.color_palette("hsv", len(df['label'].unique())),
        data=df,
        legend="full",
        alpha=0.9
    )

    # Set the title of the plot
    plt.title('t-SNE visualization of the data')

    # Show the plot
    plt.show()



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

