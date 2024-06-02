import numpy as np
import argparse
import pickle
import random
from sklearn.model_selection import train_test_split
from folktables import ACSDataSource, ACSEmployment
from sklearn.preprocessing import StandardScaler
def logistic_loss(x, theta, y):
    predict_f = lambda x,theta: np.matmul(theta, x.T) 
    sigmoid_f = lambda yhat: 1/(1+np.exp(-yhat))  
    loss_f = lambda y, sigmoid: -(y.T * np.log(np.clip(sigmoid, 1e-7, None))+(1-y).T * np.log(np.clip(1- sigmoid, 1e-7, None)))
    grad_f = lambda x,y, sigmoid: np.einsum('ij,ki->kij', x, sigmoid -y)
    loss = loss_f(y, sigmoid_f(predict_f(x,theta)))
    gradient = grad_f(x,y, sigmoid_f(predict_f(x,theta)))
    return  loss, gradient
def MSGD_census(X, Theta, n, Y, T, zeta, loss_func, eta, num_sample = 1):
    num_emb = X.shape[1] 
    Theta_traj = np.zeros((n, num_emb, T))
    Update_all_Theta_traj = np.zeros((n, num_emb, T))
    Update_all_Theta = Theta
    for t in range(T):
        user = list(range(num_sample * t, num_sample *(t+1)))
        x = X[user, :] 
        y = Y[user]
       
        p = np.random.uniform(size = (num_sample, )) 
        batch_loss,  batch_grad = loss_func(x, Theta, y) # x.shape: (1, 16); Theta.shape: (3, 16), y.shape: (1, )
        _,  batch_grad_Update_all = loss_func(x, Update_all_Theta, y) 
        model_id = np.argmin(batch_loss, axis = 0) 
        model_id[p<zeta] = np.random.randint(low=0, high=n, size = (model_id[p<zeta].shape[0], ))
        grad_model_to_update  = batch_grad[model_id, range(num_sample),:] 
        grad_Theta = np.zeros_like(Theta)
        values, _ = np.unique(model_id, return_counts=True)
        for model_index in values:
            grad_Theta[model_index,:] = np.mean(grad_model_to_update[model_id == model_index, :], axis = 0)
        Update_all_grad_Theta = np.mean(batch_grad_Update_all, axis = 1)
        Theta = Theta - eta * grad_Theta / (t+1)
        Update_all_Theta = Update_all_Theta -eta * Update_all_grad_Theta/ (t+1)
        Theta_traj[:, :, t] = Theta
        Update_all_Theta_traj[:, :, t] = Update_all_Theta
    result = {'Theta_traj': Theta_traj, 'Update_all_Theta_traj': Update_all_Theta_traj}
    return result
def population_loss_census(X_test, n, zeta, Theta_traj, y_test, loss_func):
    num_user = X_test.shape[0]
    T = Theta_traj.shape[-1]
    total_loss = np.zeros((T))
    total_acc = np.zeros((T))
    for t in range(T):
        Correct = (Theta_traj[:, :, t].dot(X_test.T)>0) == y_test
        batched_loss, _ = loss_func(X_test, Theta_traj[:, :, t], y_test) #(n, user_size)
        model_id = np.argmin(batched_loss, axis = 0)
        total_loss[t] = (1-zeta) * np.mean(batched_loss[model_id, range(num_user)]) + zeta * np.mean(batched_loss)
        total_acc[t] = np.mean(Correct[model_id, range(num_user)])
    return total_loss, total_acc

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

    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["AL"], download=True)
    features, label, group = ACSEmployment.df_to_numpy(acs_data)

    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)

    X_train, X_test, y_train, y_test, _, _ = train_test_split(
        features, label, group, test_size=0.2, random_state=0)
    n_features = X_train.shape[1]


    T = args.T
    n = args.n
    eta = args.eta
    init_seed = args.init_seed
    random.seed(init_seed)
    np.random.seed(init_seed)
    results = []
    zeta_list = [float(i) for i in args.zeta_list.split(',')]







    for zeta in zeta_list:
        print('running zeta=', zeta)
        Theta_init = np.random.rand(n, n_features)
        result = MSGD_census(X_train, Theta_init, n, y_train, T, zeta, logistic_loss, eta)
        Theta = result['Theta_traj']
        Theta_full = result['Update_all_Theta_traj']
        total_loss,  total_acc = population_loss_census(X_test, n, zeta, result['Theta_traj'], y_test, logistic_loss)
        total_loss_full, total_acc_full = population_loss_census(X_test, n, zeta, result['Update_all_Theta_traj'], y_test, logistic_loss)
        
        results.append({
                'n': n,
                'zeta': zeta, 
                'total_loss': total_loss,
                'total_loss_full': total_loss_full,
                'Theta_full': Theta_full,
                'Theta': Theta,
                'total_acc': total_acc,
                "total_acc_full": total_acc_full
            })

    # Create a DataFrame from the results
    with open('./results/census_n{}_eta{}.pickle'.format(n, eta, init_seed), 'wb') as f:
            pickle.dump( results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()