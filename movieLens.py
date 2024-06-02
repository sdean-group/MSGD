import numpy as np
import argparse
import pickle
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def square_loss(x, theta, y, mask):
    batched_loss = np.mean(np.multiply(((np.matmul(theta, x.T).transpose(0, 2, 1) - y)**2), mask), axis = -1)
    batched_grad = np.einsum('ijk,jl->ijkl', np.multiply(2 * (np.matmul(theta, x.T).transpose(0, 2, 1) - y), mask), x)
    return  batched_loss, batched_grad
def MSGD_movieLens(X, Theta, n, Y, Mask, T, zeta, loss_func, eta, num_sample =1):
    num_movies = Y.shape[1] 
    num_emb = X.shape[1] 
    Theta_traj = np.zeros((n, num_movies, num_emb, T))
    Update_all_Theta_traj = np.zeros((n, num_movies, num_emb, T))
    Update_all_Theta = Theta
    for t in range(T):
        user = [t]
        x = X[user, :] 
        y = Y[user, :]
        mask = Mask[user, :] 
        p = np.random.uniform(size = (num_sample, )) 
        batch_loss,  batch_grad = loss_func(x, Theta, y, mask) 
        _,  batch_grad_Update_all = loss_func(x, Update_all_Theta, y, mask) 
        model_id = np.argmin(batch_loss, axis = 0) 
        model_id[p<zeta] = np.random.randint(low=0, high=n, size = (model_id[p<zeta].shape[0], ))
        grad_model_to_update  = batch_grad[model_id, range(num_sample),:,:] 
        grad_Theta = np.zeros_like(Theta)
        
        values, _ = np.unique(model_id, return_counts=True)
        for model_index in values:
            grad_Theta[model_index,:,:] = np.mean(grad_model_to_update[model_id == model_index, :, :], axis = 0)
        Update_all_grad_Theta = np.mean(batch_grad_Update_all, axis = 1)
        Theta = Theta - eta * grad_Theta / (t+1)
        Update_all_Theta = Update_all_Theta -eta * Update_all_grad_Theta/ (t+1)
        Theta_traj[:, :, :, t] = Theta
        Update_all_Theta_traj[:, :, :, t] = Update_all_Theta
    result = {'Theta_traj': Theta_traj, 'Update_all_Theta_traj': Update_all_Theta_traj}
    return result
def population_loss_movieLens(X_test, n, zeta, Theta_traj, y_test, mask_test, loss_func):
    num_user = X_test.shape[0]
    T = Theta_traj.shape[-1]
    total_loss = np.zeros((T))
    for t in range(T):
        batched_loss, _ = loss_func(X_test, Theta_traj[:, :, :, t], y_test, mask_test) #(n, user_size)
        model_id = np.argmin(batched_loss, axis = 0)
        total_loss[t] = (1-zeta) * np.mean(batched_loss[model_id, range(num_user)]) + zeta * np.mean(batched_loss)
    return total_loss
def main():
    parser = argparse.ArgumentParser(description='experiments on MovieLens dataset')
    parser.add_argument('--eta', type=float, default=1, help='The constant parameter related to learning rate')
    parser.add_argument('--init_seed', type=int, default=0, help='Random seed to control initial Theta_0')
    parser.add_argument('--zeta_list', type=str, default='0,0.2,0.5,0.8,1')
    parser.add_argument('--n', type=int, default=3, help = 'The number of models when varying zeta')
    parser.add_argument('--T', type=int, default=2000, help = 'Total number of rounds')
    parser.add_argument('--data_source_file', type=str, default="./dataset/MovieLens10M_200_5.pkl", help = 'The preprocessed .pickle file for movieLens dataset')
    args = parser.parse_args()
    print(args)
    with open(args.data_source_file, 'rb') as file:
        loaded_data = pickle.load(file)

    test_size= 600 # Number of samples used during computing the population loss
    user_embeddings = loaded_data["user_embeddings"]
    item_embeddings = loaded_data["item_embeddings"]
    ratings = loaded_data["ratings"] 
    masks = loaded_data["mask"]
    num_movies = item_embeddings.shape[0] 
    num_emb = user_embeddings.shape[1]
    T = args.T
    n = args.n
    eta = args.eta
    X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(user_embeddings, ratings, masks, test_size=test_size, random_state=42)


    _, X_train, _, y_train, _, mask_train = train_test_split(X_train, y_train, mask_train, test_size=T, random_state=0)
    init_seed = args.init_seed
    random.seed(init_seed)
    np.random.seed(init_seed)
    results = []
    Theta_init = np.random.rand(n, num_movies, num_emb)
    zeta_list = [float(i) for i in args.zeta_list.split(',')]
    for zeta in zeta_list:
        print('running zeta=', zeta)
        result = MSGD_movieLens(X_train, Theta_init, n, y_train, mask_train, T, zeta, square_loss, eta)
        Theta = result['Theta_traj']
        Theta = Theta.reshape((Theta.shape[0], Theta.shape[1]* Theta.shape[2], Theta.shape[3]))
        Theta_full = result['Update_all_Theta_traj']
        Theta_full =Theta_full.reshape((Theta_full.shape[0], Theta_full.shape[1]* Theta_full.shape[2], Theta_full.shape[3]))
        total_loss = population_loss_movieLens(X_test, n, zeta, result['Theta_traj'], y_test, mask_test, square_loss)
        total_loss_full = population_loss_movieLens(X_test, n, zeta, result['Update_all_Theta_traj'], y_test, mask_test, square_loss)
        results.append({
                'n': n,
                'zeta': zeta,
                'total_loss': total_loss,
                'total_loss_full': total_loss_full,
                'Theta_full': Theta_full,
                'Theta': Theta,
                
            })

    # Create a DataFrame from the results
    with open('./results/movieLens_n{}_eta{}.pickle'.format(n, eta), 'wb') as f:
            pickle.dump( results, f, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == "__main__":
    main()